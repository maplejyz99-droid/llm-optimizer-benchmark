"""
Llama style Language Model that is 
compilable (avoids torch complex)
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import CausalSelfAttention, GPTBase
from models.moe import MoE


@torch.no_grad()
def _accumulate_xtx(x, accum, count, blocks=1):
    x2d = x.detach().float().reshape(-1, x.shape[-1])
    if blocks == 1:
        accum.add_(x2d.transpose(0, 1).matmul(x2d).div_(x2d.size(0)))
    else:
        width = x2d.shape[-1]
        if width % blocks != 0:
            raise ValueError("Newton-Muon block covariance requires divisible width.")
        block = width // blocks
        x_blocks = x2d.reshape(-1, blocks, block).permute(1, 0, 2)
        accum.add_(torch.bmm(x_blocks.transpose(1, 2), x_blocks).div_(x2d.size(0)))
    count.add_(1.0)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    # Stack the cos and sin parts in the last dimension to simulate complex numbers
    return torch.stack((cos_freqs, sin_freqs), dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (x.shape[1], x.shape[-2])
    # New shape for broadcasting
    shape = [
        1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])
    ] + [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    # q, k: (B, T, nh, hs)
    # freq_cis: (T, hs)
    # return: (B, T, nh, hs), (B, T, nh, hs)
    q = q.float().reshape(*q.shape[:-1], -1, 2)
    k = k.float().reshape(*k.shape[:-1], -1, 2)

    freqs_cis = _reshape_for_broadcast(freqs_cis, q)

    # Perform manual "complex" multiplication
    q_cos = q[..., 0] * freqs_cis[..., 0] - q[..., 1] * freqs_cis[..., 1]
    q_sin = q[..., 0] * freqs_cis[..., 1] + q[..., 1] * freqs_cis[..., 0]
    k_cos = k[..., 0] * freqs_cis[..., 0] - k[..., 1] * freqs_cis[..., 1]
    k_sin = k[..., 0] * freqs_cis[..., 1] + k[..., 1] * freqs_cis[..., 0]

    # Combine the results back into the interleaved format expected by q and k
    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k.shape).flatten(3)

    return q_out, k_out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.n_embd * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.register_buffer(
            "newton_muon_fc_accum",
            torch.zeros(config.n_embd, config.n_embd, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "newton_muon_fc_count", torch.zeros((), dtype=torch.float32), persistent=False
        )
        if hidden_dim % 4 != 0:
            raise ValueError("Newton-Muon expects Llama MLP hidden_dim divisible by 4.")
        proj_block = hidden_dim // 4
        self.register_buffer(
            "newton_muon_proj_accum",
            torch.zeros(4, proj_block, proj_block, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "newton_muon_proj_count",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x, precond_flag=False):
        if precond_flag:
            _accumulate_xtx(x, self.newton_muon_fc_accum, self.newton_muon_fc_count)
        # tuple form because of aux loss from MoE
        x = nn.functional.silu(self.w1(x)) * self.w2(x)
        if precond_flag:
            _accumulate_xtx(
                x, self.newton_muon_proj_accum, self.newton_muon_proj_count, blocks=4
            )
        return self.c_proj(x), {}


class LlamaAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer(
            "newton_muon_qkv_accum",
            torch.zeros(config.n_embd, config.n_embd, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "newton_muon_qkv_count",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "newton_muon_o_accum",
            torch.zeros(config.n_embd, config.n_embd, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "newton_muon_o_count", torch.zeros((), dtype=torch.float32), persistent=False
        )

    def forward(self, x, freqs_cis, precond_flag=False):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()
        if precond_flag:
            _accumulate_xtx(x, self.newton_muon_qkv_accum, self.newton_muon_qkv_count)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # (B, nh, T, hs)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        if precond_flag:
            _accumulate_xtx(y, self.newton_muon_o_accum, self.newton_muon_o_count)
        y = self.resid_dropout(self.c_proj(y))
        return y


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = LlamaAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)

        if config.moe:
            self.mlp = MoE(config, LlamaMLP)
        else:
            self.mlp = LlamaMLP(config)

    def forward(self, x, freqs_cis, precond_flag=False):
        x = x + self.attn(self.ln_1(x), freqs_cis, precond_flag=precond_flag)
        x_, logits_and_experts = self.mlp(self.ln_2(x), precond_flag=precond_flag)
        x = x + x_
        return x, logits_and_experts


class Llama(GPTBase):
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if not config.untied_embeds:
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=self.config.init_std / math.sqrt(2 * config.n_layer),
                )
            if pn.endswith("router.weight"):
                # special scaled init to moe router?
                with torch.no_grad():
                    std = p.std()
                    p.div_(p.sum(dim=1, keepdim=True))
                    p.mul_(std / p.std())

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default)
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(
        self,
        idx,
        targets=None,
        get_logits=False,
        moe=False,
        full_logits=False,
        precond_flag=False,
    ):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        # router logits is a list for each layer's routing, each of shape (b * seq_len, n_experts)
        router_logits = []
        # experts is a list for each layer's selected experts, shape (b * seq_len, topk)
        experts = []

        precond_flag = bool(precond_flag) and self.training
        for block in self.transformer.h:
            x, logits_and_experts = block(x, freqs_cis=freqs_cis, precond_flag=precond_flag)
            if len(logits_and_experts) > 0:
                router_logits.append(logits_and_experts["router_logits"])
                experts.append(logits_and_experts["selected_experts"])
        x = self.transformer.ln_f(x)

        # aux_losses is a dict with keys for different auxiliary losses
        aux_losses = {}
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            if moe and self.config.moe_routing == "standard_gating":
                # calculate the router losses per layer
                for logit, expert_choice in zip(router_logits, experts):
                    router_losses = self.get_router_losses(
                        logit, expert_choice, eval=not self.training
                    )
                    for k, v in router_losses.items():
                        aux_losses[k] = aux_losses.get(k, 0.0) + v
                        if self.training:
                            loss += (
                                v
                                * getattr(self.config, k + "_factor")
                                / self.config.n_layer
                            )
        elif get_logits and full_logits:
            # Return full-sequence logits without forcing CE computation.
            logits = self.lm_head(x)
            loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        logits = logits if get_logits else None

        router_logits = (
            torch.stack(router_logits, dim=0) if len(router_logits) > 0 else None
        )

        return {
            "logits": logits,
            "loss": loss,
            "aux_losses": aux_losses,
            "router_logits": router_logits,
        }
