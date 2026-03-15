# GN-Prox 分析提示词

请你阅读并分析这个项目中 `GN-Prox` 优化路径的完整工作流。

我的目标不是只看表面定义，而是要你“沿着真实代码调用链”把 `--opt gn-prox` 在这个仓库里到底如何工作讲清楚。不要只复述 Gauss-Newton 的一般定义，必须以当前 repo 的真实实现为准。

【分析目标】
请围绕 repo 中 `gn-prox` 的实现，回答：
1. 从命令行传入 `--opt gn-prox` 开始，程序如何一步步走到 `compute_gn_step(..., mode="prox")`？
2. 这个仓库里的 GN-Prox 和普通 optimizer 的关系到底是什么？谁负责“构造梯度”，谁负责“真正更新参数”？
3. GN-Prox 在这个项目里究竟优化哪些参数？parameter groups 和 weight decay / no_decay 分组是怎么进入这条链路的？
4. 一次 outer iteration 内部究竟发生了什么？`gn_inner_iters` 次 inner step、可选 `line search`、`scheduler.step()` 的相对顺序是什么？
5. 请把代码流程和 GN-Prox 的公式逐项对齐解释，不要只讲抽象概念。
6. 请说明这个 repo 里的 GN-Prox 和“纯 AdamW”相比，工作流上的最大差异是什么。
7. 如果这个仓库里实现的是“GN + inner AdamW”的混合逻辑，请明确指出混合发生在什么地方、依据什么代码路径完成。

【必须追踪的文件】
请优先分析这些文件，并在回答中说明每个文件负责什么：
- `src/config/base.py`
- `src/main.py`
- `src/optim/base.py`
- `src/optim/gn.py`
- `src/models/base.py`
- `src/distributed/backend.py`
- `src/distributed/single.py`
- `src/distributed/ddp.py`

如果你发现实际运行模型覆写了 `get_parameter_group_specs`，也要说明“当前分析是以哪个模型实现为准”。

【必须回答的技术点】
请重点解释下面这些代码点：

1. `src/config/base.py` 中：
   - `--opt` 的可选值里为什么会有 `gn-prox`
   - `--gn_inner_iters`
   - `--gn_inner_lr`
   - `--gn_inner_b1`
   - `--gn_inner_b2`
   - `--gn_inner_wd`
   - `--gn_linesearch`
   - `--gn_ls_range`
   - `--gn_log_inner_steps`

2. `src/main.py` 中：
   - `args.opt in {"gn-prox", "gn-full"}` 分支
   - 为什么 GN 模式下要 `force math SDP backend`
   - `torch.optim.AdamW(...)` 在 GN 分支里的初始化参数来源
   - 为什么这里 `weight_decay=0.0`
   - `lr=args.gn_inner_lr`
   - `betas=(args.gn_inner_b1, args.gn_inner_b2)`
   - `sched_base_lr = args.gn_inner_lr if args.opt in {"gn-prox", "gn-full"} else args.lr`
   - GN 模式下 scheduler 是如何挂到 inner AdamW 上的

3. `src/optim/base.py` 中：
   - `use_gn = cfg.opt in {"gn-prox", "gn-full"}`
   - `gn_mode = "full" if cfg.opt == "gn-full" else "prox"`
   - `params = current_param_dict(raw_model)`
   - `params0 = clone_param_dict(raw_model)`
   - inner loop 里如何调用 `compute_gn_step`
   - `for param, grad in zip(params.values(), grads): param.grad = grad.detach()`
   - `opt.step()`、`opt.zero_grad(set_to_none=True)`、`substep += 1`
   - `cfg.gn_log_inner_steps`
   - `gn_linesearch` 分支
   - `direction = sub_param_dict(current_params, params0)`
   - 为什么 `scheduler.step()` 在 GN 路径里发生在 inner loop 之后，而不是每个 inner step 后
   - 为什么 GN 路径里 `loss = torch.tensor(gn_metrics.base_loss, device=cfg.device)`，而不是直接记录 `gn_metrics.loss`

4. `src/optim/gn.py` 中：
   - `current_param_dict`
   - `clone_param_dict`
   - `sub_param_dict`
   - `add_scaled_param_dict`
   - `apply_param_dict_`
   - `compute_gn_step`
   - `mode == "prox"` 分支
   - `logits_fn`
   - `functional_call`
   - `jvp`
   - `delta = sub_param_dict(params, params0)`
   - `logits0, jvp_delta = jvp(logits_fn, (params0,), (delta,))`
   - `logits_linearized = logits0.detach() + jvp_delta`
   - `base_loss = _cross_entropy_from_logits(logits_linearized, y)`
   - `prox = prox_weight_decay * _mean_squared_delta(delta)`
   - `loss = base_loss + prox`
   - `grads = torch.autograd.grad(loss, tuple(params.values()))`
   - `GNStepMetrics`
   - `line_search_over_direction`

5. `src/models/base.py` 中：
   - `get_parameter_group_specs`
   - `decay` / `no_decay`
   - `whitelist_weight_modules`
   - `BLACKLIST_WEIGHT_MODULES`
   - `{"params": sorted(list(decay))}` 和 `{"params": sorted(list(no_decay)), "weight_decay": 0.0}`
   - 这些 parameter groups 最终如何进入 GN 分支里的 inner AdamW

6. distributed 相关：
   - `distributed_backend.transform_model(model)`
   - `distributed_backend.get_raw_model(model)`
   - `translate_model_parameter_name_for_node`
   - DDP 下 GN 为什么取 `raw_model`
   - 单机和 DDP 对 GN 调用链的影响是什么

【公式对齐要求】
请你一定要把代码和下面这类 `GN-Prox` 公式对齐起来解释：

- 参数位移：
  `delta = params - params0`

- 线性化 logits：
  `logits0, J_delta = jvp(logits_fn, params0, delta)`

- 线性化目标：
  `logits_linearized = logits0 + J_delta`

- prox 目标：
  `L_prox(delta) = CE(logits_linearized, y) + lambda * mean(delta^2)`

- 对当前参数的梯度：
  `grads = d L_prox / d params`

并且请你明确说明：
1. 上面每个公式项在代码里对应哪个变量。
2. 为什么 `delta` 是 `params - params0`，也就是“当前 inner 参数相对 anchor 参数的位移”。
3. 为什么 `jvp` 是在 `params0` 处对 `delta` 做前向模式线性化。
4. `prox_weight_decay` 在这里为什么不是 AdamW 式 decoupled weight decay，而是 GN objective 里的显式惩罚项。
5. 这个仓库里并没有写一个“GN 自己的闭式参数更新公式”，真正的参数更新为什么仍然是 `opt.step()` 的 inner AdamW。

如果你写参数更新关系，请把“GN 构造出来的梯度”和“AdamW 实际使用这些梯度更新参数”分开写清楚，不能混为一谈。

【scheduler 与 line search 必须讲清楚】
请明确解释：
1. 普通 optimizer 的 scheduler 和 GN 这里的 scheduler 有什么相同与不同。
2. 为什么 GN 的 `scheduler.step()` 是按 outer iteration 调一次，但 `opt.step()` 在一个 outer iteration 内会跑 `gn_inner_iters` 次。
3. `sched_base_lr` 为什么取 `args.gn_inner_lr` 而不是 `args.lr`。
4. `line_search_over_direction` 是在什么时机运行的。
5. `line_search_over_direction` 用的损失到底是：
   - 线性化损失
   - 还是 full model cross-entropy
   你必须根据真实代码回答。
6. `line_search` 为什么是沿着 `direction = current_params - params0` 从 `anchor_params=params0` 出发做候选步长搜索。

【必须主动指出的易错点】
看到下面这些容易误解的地方，请你主动指出并纠正：
- GN 在这个仓库里不是一个独立的 `Optimizer` 子类。
- `src/main.py` 里创建的是 inner `torch.optim.AdamW`，不是 “GN optimizer”。
- `gn_inner_wd` 不是 `AdamW(weight_decay=...)` 的那个 weight decay。
- GN 路径里训练日志记录的 `loss` 与 `gn_metrics.loss` 不是同一个量。
- GN 路径下 `scheduler` 调整的是 inner AdamW 学习率。
- `line_search` 发生在 inner updates 之后，不在 `compute_gn_step` 里面。

【输出格式】
请严格按下面结构输出：

# 1. 调用链总览
用“文件 -> 函数/类 -> 下一步”的方式梳理从 `--opt gn-prox` 到 `compute_gn_step(..., mode="prox")` 再到 `opt.step()` 的调用链

# 2. GN 参数与超参数来源
说明 GN 相关超参数从 CLI 如何进入 `main.py`、`train()`、`compute_gn_step()` 和 inner AdamW

# 3. GN-Prox 单步工作流的代码级拆解
按一次 outer iteration 内部真实顺序分成若干小步骤讲
每一步都要包含：
- 代码位置
- 核心变量
- 数学含义
- 对应公式

# 4. `compute_gn_step(..., mode="prox")` 的公式对齐
把 `delta`、`jvp`、线性化 logits、prox penalty、`autograd.grad` 逐项对齐解释

# 5. line search 与 scheduler 如何接入 GN-Prox
重点讲它们在调用顺序、目标函数、lr 粒度上的作用

# 6. GN-Prox vs 纯 AdamW 在这个仓库里的真正区别
从“梯度来源”“参数更新执行者”“step 粒度”“weight decay 处理”“日志口径”“distributed 兼容处理”“scheduler 接入方式”几个角度对比

# 7. 我最该关注的代码片段
列出最关键的 5~10 段代码，并说明为什么它们关键

【风格要求】
- 不要只给摘要，要像高级代码审阅一样讲清楚
- 不要跳步
- 不要只讲论文定义，必须以 repo 实现为准
- 用中文输出
- 数学公式、代码变量名、文件路径都保留原样
- 如果某个行为是你根据代码推断出来的，请明确标注“这是根据代码推断，不是代码直接注释”
- 看到可能让人误解的地方，请主动指出
