import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .muon import Muon, zeropower_via_newtonschulz5


def _compute_magma_factor(
    state,
    param,
    grad,
    moment,
    magma_param_ids,
    survival_p,
    tau,
    ema_beta,
):
    if magma_param_ids is not None and id(param) not in magma_param_ids:
        return None

    if "magma_scale_ema" not in state:
        state["magma_scale_ema"] = torch.tensor(
            1.0, device=grad.device, dtype=torch.float32
        )

    # Use fp32 for the alignment score to avoid noisy low-precision cosine values.
    cos = F.cosine_similarity(moment.float().flatten(), grad.float().flatten(), dim=0)
    s_tilde = torch.sigmoid(cos / tau)
    state["magma_scale_ema"].mul_(ema_beta).add_(s_tilde, alpha=1.0 - ema_beta)
    mask = torch.bernoulli(
        torch.full((), survival_p, device=grad.device, dtype=torch.float32)
    )
    return (state["magma_scale_ema"] * mask).to(grad.dtype)


class MagmaAdamW(torch.optim.Optimizer):
    """
    AdamW + Magma wrapper.
    Keep AdamW moments dense, then apply block-level (tensor-level) Magma scaling
    and Bernoulli masking to the already-computed update direction.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        magma_survival_p=0.5,
        magma_tau=2.0,
        magma_beta=0.9,
        magma_param_ids=None,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 < magma_survival_p <= 1.0:
            raise ValueError(f"Invalid magma_survival_p: {magma_survival_p}")
        if magma_tau <= 0.0:
            raise ValueError(f"Invalid magma_tau: {magma_tau}")
        if not 0.0 <= magma_beta < 1.0:
            raise ValueError(f"Invalid magma_beta: {magma_beta}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.magma_survival_p = magma_survival_p
        self.magma_tau = magma_tau
        self.magma_beta = magma_beta
        self.magma_param_ids = magma_param_ids

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("MagmaAdamW does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg / denom

                magma_factor = _compute_magma_factor(
                    state=state,
                    param=p,
                    grad=grad,
                    moment=exp_avg,
                    magma_param_ids=self.magma_param_ids,
                    survival_p=self.magma_survival_p,
                    tau=self.magma_tau,
                    ema_beta=self.magma_beta,
                )
                if magma_factor is not None:
                    update = update * magma_factor

                p.data.add_(update, alpha=-(lr / bias_correction1))

        return loss


class MagmaMuon(Muon):
    """
    Muon + Magma wrapper (new class, original Muon remains untouched).
    """

    def __init__(
        self,
        *args,
        magma_survival_p=0.5,
        magma_tau=2.0,
        magma_beta=0.9,
        magma_param_ids=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 < magma_survival_p <= 1.0:
            raise ValueError(f"Invalid magma_survival_p: {magma_survival_p}")
        if magma_tau <= 0.0:
            raise ValueError(f"Invalid magma_tau: {magma_tau}")
        if not 0.0 <= magma_beta < 1.0:
            raise ValueError(f"Invalid magma_beta: {magma_beta}")
        self.magma_survival_p = magma_survival_p
        self.magma_tau = magma_tau
        self.magma_beta = magma_beta
        self.magma_param_ids = magma_param_ids

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            ############################
            #       Muon branch        #
            ############################
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            momentum = group["momentum"]

            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    grad = p.grad
                    assert grad is not None
                    if grad.ndim > 2:
                        grad = grad.view(grad.size(0), -1)
                    state = self.state[p]

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)

                    ns_input = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                    update = zeropower_via_newtonschulz5(
                        ns_input, steps=group["ns_steps"]
                    )
                    update *= max(1, update.size(0) / update.size(1)) ** 0.5

                    magma_factor = _compute_magma_factor(
                        state=state,
                        param=p,
                        grad=grad,
                        moment=buf,
                        magma_param_ids=self.magma_param_ids,
                        survival_p=self.magma_survival_p,
                        tau=self.magma_tau,
                        ema_beta=self.magma_beta,
                    )
                    if magma_factor is not None:
                        update = update * magma_factor

                    updates_flat[curr_idx : curr_idx + p.numel()] = update.flatten()
                curr_idx += p.numel()

            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in params:
                update = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                p.data.add_(update, alpha=-lr)
                curr_idx += p.numel()

            ############################
            #    AdamW backup branch   #
            ############################
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["adamw_lr_ratio"] * group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                grad = p.grad
                assert grad is not None
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
                magma_factor = _compute_magma_factor(
                    state=state,
                    param=p,
                    grad=grad,
                    moment=moment1,
                    magma_param_ids=self.magma_param_ids,
                    survival_p=self.magma_survival_p,
                    tau=self.magma_tau,
                    ema_beta=self.magma_beta,
                )
                if magma_factor is not None:
                    update = update * magma_factor

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr / scale)
