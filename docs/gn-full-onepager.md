# GN-Full One Pager

基于当前仓库真实实现整理，目标是把 `--opt gn-full` 的调用链、公式映射、line search 与 scheduler 关系压在一页里。

```mermaid
flowchart TD
  subgraph A["1) 入口与优化器装配"]
    A1["CLI<br/>--opt gn-full<br/>--gn_inner_iters / lr / b1 / b2 / wd<br/>--gn_linesearch --gn_ls_range"] --> A2["src/config/base.py<br/>注册 gn-full 与全部 GN 超参数"]
    A2 --> A3["src/main.py<br/>GN mode -> force math SDP backend<br/>关闭 Flash/MemEff SDP"]
    A3 --> A4["raw_model.get_parameter_group_specs(config)<br/>先在 raw model 上生成 decay / no_decay 分组"]
    A4 --> A5["translate_model_parameter_name_for_node()<br/>把参数名翻译到当前节点包装后的参数"]
    A5 --> A6["inner torch.optim.AdamW(group_specs,<br/>lr=gn_inner_lr,<br/>betas=(gn_inner_b1, gn_inner_b2),<br/>weight_decay=0.0)"]
    A6 --> A7["scheduler 挂到 inner AdamW<br/>sched_base_lr = gn_inner_lr<br/>不是 args.lr"]
  end

  subgraph B["2) outer iteration in train()"]
    B1["src/optim/base.py<br/>use_gn = cfg.opt in {gn-prox, gn-full}<br/>gn_mode = 'full'"]
    B2["raw_model = distributed_backend.get_raw_model(model)<br/>params = current_param_dict(raw_model)<br/>params0 = clone_param_dict(raw_model)"]
    B3["repeat gn_inner_iters"]
    B4["compute_gn_step(model=raw_model,<br/>params0=params0,<br/>mode='full',<br/>prox_weight_decay=gn_inner_wd)"]
    B5["for param, grad in zip(params.values(), grads):<br/>param.grad = grad.detach()"]
    B6["opt.step()<br/>inner AdamW 真正更新参数"]
    B7["opt.zero_grad(set_to_none=True)<br/>substep += 1"]
    B8["optional:<br/>line_search_over_direction(anchor=params0,<br/>direction=current_params - params0)"]
    B9["scheduler.step()<br/>每个 outer iter 只调一次"]
    B10["log loss = gn_metrics.base_loss<br/>不是 gn_metrics.loss"]
    B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B3
    B3 -->|inner loop done| B8 --> B9 --> B10
  end

  subgraph C["3) compute_gn_step(..., mode='full') 公式对照"]
    C1["delta = params - params0"]
    C2["logits_fn(pdict) = functional_call(model, (pdict, buffers), x,<br/>targets=None, get_logits=True, moe=cfg.moe, full_logits=True)"]
    C3["logits0, J_delta = jvp(logits_fn, (params0,), (delta,))"]
    C4["logits0_for_grad = logits0.detach().requires_grad_(True)<br/>base_loss = CE(logits0_for_grad, y)"]
    C5["g0 = d CE(logits0, y) / d logits0"]
    C6["hv = H_logits * J_delta<br/>hvp(base_loss_on_logits, logits0_for_grad, J_delta)[1]"]
    C7["pullback = VJP(logits_fn, params0)(g0 + hv)<br/>把 logits 空间信号拉回参数空间"]
    C8["if gn_inner_wd > 0:<br/>grad[name] += (2 * wd / numel) * delta[name]<br/>显式 prox 梯度, 不是 AdamW decoupled decay"]
    C9["surrogate = base_loss + g0^T J_delta + 0.5 * J_delta^T H_logits J_delta"]
    C10["return grads, metrics<br/>metrics.loss = surrogate<br/>metrics.base_loss = base_loss"]
    C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7 --> C8 --> C9 --> C10
  end

  A7 --> B1
  B4 --> C1
  C10 --> B5
  A4 -. "同一套 parameter groups<br/>决定谁被更新" .-> A6
```

## Formula Map

| 公式项 | 代码变量 | 代码位置 | 含义 |
| --- | --- | --- | --- |
| `delta = params - params0` | `delta` | `src/optim/gn.py` | 当前 inner 参数相对 outer anchor 的位移 |
| `J_delta` | `jvp_delta` | `src/optim/gn.py` | logits 对参数位移的雅可比方向项 |
| `logits0` | `logits0` | `src/optim/gn.py` | anchor `params0` 下的 full logits |
| `g0 = d CE(logits0, y) / d logits0` | `g0` | `src/optim/gn.py` | logits 空间的一阶梯度 |
| `hv = H_logits * J_delta` | `hv` | `src/optim/gn.py` | logits 空间二阶修正项 |
| `pullback = J^T (g0 + hv)` | `pullback = vjp_fn((g0 + hv).detach())[0]` | `src/optim/gn.py` | 把 logits 空间信号拉回参数空间 |
| `grad_param = pullback + d prox / d params` | `grad` | `src/optim/gn.py` | 供 inner AdamW 消费的最终参数梯度 |
| `surrogate = base_loss + g0^T J_delta + 0.5 J_delta^T H_logits J_delta` | `surrogate` | `src/optim/gn.py` | 只用于 metrics / logging，不直接输入 `opt.step()` |

## Key Distinctions

- `gn-full` 不是独立 `Optimizer` 子类；真正执行更新的是 inner `torch.optim.AdamW`。
- GN 负责“构造梯度”，AdamW 负责“带动量状态地更新参数”。
- `gn_inner_wd` 进入 `compute_gn_step(..., prox_weight_decay=...)`，不是 `AdamW(weight_decay=...)`。
- `scheduler.step()` 调的是 inner AdamW 的学习率；它按 outer iteration 调一次，不按 inner step 调。
- `line_search_over_direction()` 发生在全部 inner updates 之后，目标是 full model cross-entropy，不是 `surrogate`。
- 训练日志里的主 `loss` 取 `gn_metrics.base_loss`，而 `gn_metrics.loss` 在 `gn-full` 下是 `surrogate`。

