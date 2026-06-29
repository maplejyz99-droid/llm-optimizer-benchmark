import os

import torch
import torch.distributed as dist


SOFTEQ_ALPHA = 0.5
SOFTEQ_STEPS = 2000
SOFTEQ_NS_STEPS = 12


def zeropower_via_newtonschulz12(update, steps=SOFTEQ_NS_STEPS, eps=1e-7):
    """Newton-Schulz orthogonalization used by the SoftEq K=2000 variant."""
    if update.ndim != 2:
        raise ValueError(f"Expected a matrix update, got shape {tuple(update.shape)}.")

    # Keep the Track 3 GPU path in bf16, but allow CPU tests on machines where
    # bf16 matmul is unavailable or very limited.
    x = update.bfloat16() if update.device.type == "cuda" else update.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    x = x / (x.norm() + eps)
    a, b, c = 2.0, -1.5, 0.5
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x

    if transposed:
        x = x.T
    return x


def soft_row_equilibrate(update, alpha=SOFTEQ_ALPHA, eps=1e-8):
    """Apply row-wise SoftEq: row / max(||row||_2, eps)**alpha."""
    row_norm = update.norm(dim=-1, keepdim=True).clamp_min(eps)
    return update / row_norm.pow(alpha)


class SoftEqK2000Muon(torch.optim.Optimizer):
    """
    Muon variant with SoftEq-0.5 enabled for the first 2000 optimizer steps.

    Matrix parameters follow the user-provided SoftEq K=2000 Muon formula:
    M_t = mu M_{t-1} + (1 - mu) G_t,
    U_t = (1 - mu) G_t + mu M_t, then SoftEq-0.5(U_t) before Orth_12
    while global_step < 2000. Non-matrix parameters use the same AdamW backup
    contract as the local Muon baseline.
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.1,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0.0,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if adamw_lr <= 0:
            raise ValueError(f"Invalid AdamW learning rate: {adamw_lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            softeq_alpha=SOFTEQ_ALPHA,
            softeq_steps=SOFTEQ_STEPS,
            softeq_ns_steps=SOFTEQ_NS_STEPS,
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

        self.global_step = 0
        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
        for p in adamw_params:
            self.state[p]["use_muon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def state_dict(self):
        state = super().state_dict()
        state["softeq_global_step"] = self.global_step
        return state

    def load_state_dict(self, state_dict):
        self.global_step = state_dict.get("softeq_global_step", 0)
        optimizer_state = dict(state_dict)
        optimizer_state.pop("softeq_global_step", None)
        return super().load_state_dict(optimizer_state)

    def _matrix_update(self, grad, state, group):
        momentum = group["momentum"]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1.0 - momentum)
        update = grad.lerp(buf, momentum)

        if self.global_step < group["softeq_steps"]:
            update = soft_row_equilibrate(
                update,
                alpha=group["softeq_alpha"],
            )

        update = zeropower_via_newtonschulz12(update, steps=group["softeq_ns_steps"])
        update *= max(1.0, update.size(0) / update.size(1)) ** 0.5
        return update

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            total_params = sum(p.numel() for p in params)
            if total_params:
                first_param = params[0]
                updates_flat = torch.zeros(
                    total_params,
                    device=first_param.device,
                    dtype=torch.bfloat16 if first_param.device.type == "cuda" else torch.float32,
                )
            else:
                updates_flat = None

            curr_idx = 0
            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    grad = p.grad
                    if grad is None:
                        raise RuntimeError("Missing gradient for a SoftEq Muon matrix parameter.")
                    if grad.ndim > 2:
                        grad = grad.view(grad.size(0), -1)
                    update = self._matrix_update(grad, self.state[p], group)
                    updates_flat[curr_idx : curr_idx + p.numel()] = update.flatten()
                curr_idx += p.numel()

            if updates_flat is not None:
                if self.world_size > 1:
                    dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

                curr_idx = 0
                for p in params:
                    update = updates_flat[curr_idx : curr_idx + p.numel()]
                    update = update.view_as(p.data).type_as(p.data)
                    p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                    p.data.add_(update, alpha=-group["lr"])
                    curr_idx += p.numel()

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
