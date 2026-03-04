import torch


class CAdamW(torch.optim.Optimizer):
    """
    Cautious AdamW (C-AdamW).

    The update direction from AdamW is element-wise masked by alignment with
    current gradients and then rescaled by d / (nnz(mask) + xi), where d is the
    number of parameters in the tensor.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        cautious_xi=1.0,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if cautious_xi <= 0:
            raise ValueError(f"Invalid cautious_xi: {cautious_xi}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            cautious_xi=cautious_xi,
        )
        super().__init__(params, defaults)

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
            cautious_xi = group["cautious_xi"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("CAdamW does not support sparse gradients.")

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

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                update = m_hat / (v_hat.sqrt() + eps)

                # Cautious masking: keep coordinates where update aligns with gradient.
                mask = (update.float() * grad.float() > 0).to(update.dtype)
                scale = mask.numel() / (mask.sum() + cautious_xi)
                step_size = lr * scale

                p.data.add_(update * mask, alpha=-step_size)

                # Match the C-AdamW pseudocode: apply decoupled decay with scaled lr.
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-step_size * weight_decay)

        return loss
