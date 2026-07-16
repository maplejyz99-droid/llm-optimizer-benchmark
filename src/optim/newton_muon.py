import math

import torch

from .muon import zeropower_via_newtonschulz5


class NewtonMuon(torch.optim.Optimizer):
    """
    Single-device Newton-Muon v1.

    This follows the local Muon baseline's parameter split, then adds a pure
    PyTorch right-preconditioner for mapped dense Llama matrices before the
    usual momentum + Newton-Schulz Muon update.
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0.0,
        precond_every=32,
        precond_ewma=0.95,
        precond_init_diag=1e-3,
        precond_ridge_mult=0.2,
        precond_eps=1e-8,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if adamw_lr <= 0:
            raise ValueError(f"Invalid AdamW learning rate: {adamw_lr}")
        if precond_every <= 0:
            raise ValueError(f"Invalid preconditioner interval: {precond_every}")
        if not 0.0 <= precond_ewma < 1.0:
            raise ValueError(f"Invalid preconditioner EWMA: {precond_ewma}")
        if precond_init_diag <= 0:
            raise ValueError(f"Invalid preconditioner init diag: {precond_init_diag}")
        if precond_ridge_mult < 0:
            raise ValueError(f"Invalid preconditioner ridge multiplier: {precond_ridge_mult}")
        if precond_eps <= 0:
            raise ValueError(f"Invalid preconditioner epsilon: {precond_eps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        self.precond_every = int(precond_every)
        self.precond_ewma = float(precond_ewma)
        self.precond_init_diag = float(precond_init_diag)
        self.precond_ridge_mult = float(precond_ridge_mult)
        self.precond_eps = float(precond_eps)
        self.global_step = 0
        self._precond_attached = False
        self._precond_map = {}

        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def attach_preconditioner(self, model):
        """Map dense Llama matrices to their activation covariance buffers."""
        raw_model = model.module if hasattr(model, "module") else model
        if not hasattr(raw_model, "transformer") or not hasattr(raw_model.transformer, "h"):
            raise ValueError("Newton-Muon v1 expects a dense Llama-style model.")

        self._precond_map = {}
        for block in raw_model.transformer.h:
            attn = block.attn
            mlp = block.mlp
            self._register_precond_param(
                attn.c_attn.weight,
                kind="full",
                accum=attn.newton_muon_qkv_accum,
                count=attn.newton_muon_qkv_count,
            )
            self._register_precond_param(
                attn.c_proj.weight,
                kind="full",
                accum=attn.newton_muon_o_accum,
                count=attn.newton_muon_o_count,
            )
            if not all(hasattr(mlp, name) for name in ("w1", "w2", "c_proj")):
                raise ValueError("Newton-Muon v1 only supports dense Llama MLP blocks.")
            self._register_precond_param(
                mlp.w1.weight,
                kind="full",
                accum=mlp.newton_muon_fc_accum,
                count=mlp.newton_muon_fc_count,
            )
            self._register_precond_param(
                mlp.w2.weight,
                kind="full",
                accum=mlp.newton_muon_fc_accum,
                count=mlp.newton_muon_fc_count,
            )
            self._register_precond_param(
                mlp.c_proj.weight,
                kind="blocks4",
                accum=mlp.newton_muon_proj_accum,
                count=mlp.newton_muon_proj_count,
            )

        self._precond_attached = True

    def precond_flag_for_step(self, step=None):
        step = self.global_step if step is None else step
        return self._precond_attached and step % self.precond_every == self.precond_every - 1

    def state_dict(self):
        state = super().state_dict()
        state["newton_muon_global_step"] = self.global_step
        return state

    def load_state_dict(self, state_dict):
        optimizer_state = dict(state_dict)
        self.global_step = optimizer_state.pop("newton_muon_global_step", 0)
        return super().load_state_dict(optimizer_state)

    def _register_precond_param(self, p, kind, accum, count):
        if p not in self.state or not self.state[p].get("use_muon", False):
            return
        expected_width = accum.shape[-1] if kind == "full" else accum.shape[0] * accum.shape[-1]
        if p.shape[-1] != expected_width:
            raise ValueError(
                f"Preconditioner dimension mismatch for shape {tuple(p.shape)} "
                f"and accum {tuple(accum.shape)}."
            )
        self._precond_map[p] = {"kind": kind, "accum": accum, "count": count}
        state = self.state[p]
        if "precond_cov" not in state:
            state["precond_cov"] = self._init_cov(accum, p.device)
        if "precond_inv" not in state:
            state["precond_inv"] = torch.empty_like(state["precond_cov"])
        if "precond_ready" not in state:
            state["precond_ready"] = torch.tensor(False, device=p.device)

    def _init_cov(self, accum, device):
        cov = torch.zeros_like(accum, dtype=torch.float32, device=device)
        if cov.ndim == 2:
            cov.diagonal().fill_(self.precond_init_diag)
        elif cov.ndim == 3:
            cov.diagonal(dim1=-2, dim2=-1).fill_(self.precond_init_diag)
        else:
            raise ValueError(f"Unsupported covariance rank: {cov.ndim}")
        return cov

    def _refresh_preconditioners(self):
        for p, ref in self._precond_map.items():
            state = self.state[p]
            accum = ref["accum"].to(device=p.device, dtype=torch.float32)
            count = ref["count"].to(device=p.device, dtype=torch.float32).clamp_min(1.0)
            observed = accum / count
            cov = state["precond_cov"]
            cov.mul_(self.precond_ewma).add_(observed, alpha=1.0 - self.precond_ewma)
            inv = self._regularized_inverse(cov)
            state["precond_inv"].copy_(inv)
            state["precond_ready"].fill_(True)
            ref["accum"].zero_()
            ref["count"].zero_()

    def _regularized_inverse(self, cov):
        if cov.device.type == "mps":
            return self._regularized_inverse(cov.cpu()).to(cov.device)
        mat = cov.clone()
        diag = mat.diagonal(dim1=-2, dim2=-1)
        ridge = diag.mean(dim=-1).clamp_min(0.0) * self.precond_ridge_mult
        ridge = ridge + self.precond_eps
        eye = torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)
        if mat.ndim == 2:
            mat = mat + ridge * eye
        else:
            mat = mat + ridge.view(-1, 1, 1) * eye
        for _ in range(3):
            chol, info = torch.linalg.cholesky_ex(mat, upper=False, check_errors=False)
            if torch.all(info == 0):
                return torch.cholesky_inverse(chol, upper=False)
            mat = mat + self.precond_eps * 10.0 * eye
        return torch.linalg.pinv(mat)

    def _precondition_grad(self, p, grad):
        ref = self._precond_map.get(p)
        if ref is None:
            return grad
        state = self.state[p]
        ready = state.get("precond_ready")
        if ready is None or not bool(ready.item()):
            return grad
        inv = state["precond_inv"].to(device=grad.device, dtype=grad.dtype)
        if ref["kind"] == "full":
            return grad.reshape(grad.shape[0], -1).matmul(inv).view_as(grad)
        if ref["kind"] == "blocks4":
            if grad.shape[-1] % 4 != 0:
                raise ValueError("Block preconditioner expects input dimension divisible by 4.")
            block = grad.shape[-1] // 4
            pieces = grad.reshape(grad.shape[0], 4, block).transpose(0, 1)
            pieces = torch.bmm(pieces, inv)
            return pieces.transpose(0, 1).reshape_as(grad)
        raise ValueError(f"Unknown Newton-Muon preconditioner kind: {ref['kind']}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.precond_flag_for_step():
            self._refresh_preconditioners()

        for group in self.param_groups:
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            momentum = group["momentum"]

            for p in params:
                grad = p.grad
                if grad is None:
                    continue
                if grad.ndim > 2:
                    grad = grad.view(grad.size(0), -1)

                grad = self._precondition_grad(p, grad)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                update *= max(1, update.size(0) / update.size(1)) ** 0.5
                p.data.add_(update.view_as(p.data).type_as(p.data), alpha=-lr)

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["adamw_lr_ratio"] * group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(grad)
                    state["moment2"] = torch.zeros_like(grad)
                state["step"] += 1
                step = state["step"]
                moment1 = state["moment1"]
                moment2 = state["moment2"]
                moment1.lerp_(grad, 1 - beta1)
                moment2.lerp_(grad.square(), 1 - beta2)
                update = moment1 / (eps + moment2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr / scale)

        self.global_step += 1
        return loss
