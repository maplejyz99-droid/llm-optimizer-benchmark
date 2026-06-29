# Optimizer 实现审计

日期：2026-06-29

## 范围

这是 optimizer benchmark 分支的第一轮实现审计。目标是检查本地 optimizer 和实验 wiring 是否真实实现，记录实现与声明的 upstream / paper-style formula 之间的差异，并判断这些差异在 benchmark 中是否可接受。

本轮证据来源：

- `src/optim/`、`src/main.py`、`src/optim/base.py` 下的本地源码
- `tests/behavior/` 下的本地行为测试
- optimizer 文件中的声明来源注释
- staging 分支已有 CodeStable notes
- 用户提供的 `softeq_k2000.py` 和公式截图

本轮限制：当前环境没有完成外部官方来源的完整逐行核验。下表中的“声明来源”应理解为本地代码注释声明的来源，不等同于新做过的官方 diff。

目录约定：自研、用户提供、未完成公开官方来源核验的实验 optimizer 放在 `src/optim/experimental/`；第三方 baseline、论文 baseline 或已有正式实现继续放在 `src/optim/` 根目录。

## 总结

当前 benchmark 中大多数 optimizer 是真实实现，不是 CLI 空壳。最重要的 caveat 是：不少变体是 benchmark-adapted implementation，不是严格 paper replica。只要命名和结果解释中写清楚“本地 benchmark baseline / variant”，这些差异可以接受；但在完成官方来源 diff 前，不能宣称它们是 exact official reproduction。

`softeq-k2000-muon` 可以作为独立 experimental variant 接入 benchmark。它已经接入 CLI、optimizer builder、scheduler contract、124M script manifest、行为测试和接入文档。在确认公开官方实现或论文来源前，应继续标注为 user-provided / experimental。

同日后续审查已加入以下加固：

- SoftEq 训练前拒绝 MoE 和非 `llama` / `mup_llama` 模型。
- SoftEq optimizer state loading 不再修改调用方传入的 `state_dict`。
- Optimizer group logging 对 AdamW backup 参数打印实际 backup learning rate，而不是 matrix learning rate。
- Schedule-free optimizer 现在拒绝外部 scheduler，要求使用 `--scheduler none`。

## Optimizer 审计表

| Optimizer | 本地实现 | 声明来源 / 依据 | 差异与可接受性 |
| --- | --- | --- | --- |
| `adamw` | 在 `src/main.py` 中使用原生 `torch.optim.AdamW`；仅在 CUDA 可用时启用 fused mode。 | PyTorch implementation。 | 可接受的 baseline。fused / non-fused 是硬件相关实现细节。 |
| `cadamw` | `src/optim/cadamw.py` 实现 cautious masking、scaled update、decoupled weight decay。 | 文件内无 upstream URL。 | 可作为本地 C-AdamW variant；若要称 official，需要额外做来源核验。 |
| `soap` | `src/optim/soap.py` 有完整 Shampoo-style preconditioner / eigenbasis 逻辑，以及 GaLore-derived projection pieces。 | 声明 `nikhilvyas/SOAP`、arXiv `2409.11321`、GaLore projector code。 | 已实现。`merge_dims`、`precondition_1d`、`normalize_grads` 等是本地实验配置；报告结果时需要列出。 |
| `muon` | `src/optim/muon.py` 本地 `Muon` class 使用 matrix / non-matrix partition、5-step quintic Newton-Schulz、AdamW backup。 | 声明 KellerJordan Muon / modded-nanogpt lineage。 | 已实现，但文件自身说明旧 `Muon` class 不含 matrix weight decay。可作为历史本地 Muon baseline；不要称为 latest Muon with matrix WD。 |
| `d-muon` | `src/optim/muon.py` 中的 `DistributedMuon` 包含 distributed metadata support 和 matrix weight decay。 | 声明 KellerJordan Muon 和 Megatron-LM distributed Muon implementation。 | 已实现。可接受性取决于 distributed smoke；本地 CPU 测试只保护 construction / order，不保护多 GPU 数值行为。 |
| `muon-pytorch` | 使用 `torch.optim.Muon`，显式设置 coefficients `(3.4445, -4.775, 2.0315)`、`eps=1e-7`、`adjust_lr_fn=None`。 | PyTorch 2.9 optimizer。 | 可作为 PyTorch-native comparison。它与本地 `Muon` 不完全相同，因为参数分组和 LR policy 不同。 |
| `newton-muon` | `src/optim/newton_muon.py`，仅绑定 dense single-device Llama，并在 `main.py` 中加 guard。 | 本地分支 feature。 | 已作为 experimental 实现。guard 可以减少错误 benchmark claim。正式比较前仍需 GPU training evidence。 |
| `muon-magma` / `adamw-magma` | `src/optim/magma.py` 在 AdamW / Muon-style update 上包装 MAGMA 参数选择。 | 本地分支 feature。 | 已实现。可作为 local intervention variant；结果必须报告 MAGMA selection scope、survival、tau、beta。 |
| `softeq-k2000-muon` | `src/optim/experimental/softeq_muon.py` 实现 `M_t`、`U_t`、step 2000 前 SoftEq-0.5、12-step Newton-Schulz、matrix WD、AdamW backup、state-dict cutoff resume。 | 用户提供的文件和公式截图。 | 可作为独立 experimental benchmark variant。公开官方 parity 未验证；不能称为 paper-equivalent。正式比较前需要 GPU smoke 和 step 2000 cutoff 附近的 checkpoint-resume 验证。 |
| `ademamix` | `src/optim/ademamix.py`；`main.py` 传入 benchmark CLI 的 alpha / beta3 / warmup 值。 | 声明 `apple/ml-ademamix`。 | 已实现。超参是 benchmark choice，可能不同于 paper default；脚本和报告必须记录。 |
| `lion` | `src/optim/lion.py` 实现 sign-based update 和 decoupled weight decay。 | 声明 Google AutoML Lion。 | 已实现。若要严格 official claim，需要额外 source diff。 |
| `sf-adamw` / `sf-sgd` | `src/optim/schedulefree.py`；training / eval loop 会按需调用 `train()` / `eval()`。 | 声明 `facebookresearch/schedule_free`。 | 已实现。该 optimizer 自带 schedule；`main.py` 现在拒绝外部 scheduler，要求 `--scheduler none`。 |
| `signsgd` / `signum` | 通过 `src/optim/sign.py` 的 `Signum` 实现；`signsgd` 强制 zero momentum。 | 本地实现。 | 已实现。可作为简单 baseline。 |
| `prodigy` | `src/optim/prodigy.py`，带 beta3、decouple、bias correction、safeguard warmup、FSDP flag。 | 声明 `konstmish/prodigy`。 | 已实现。严格 parity 需要 source diff。 |
| `sophiag` | `src/optim/sophia.py`；training loop 每 `precondition_frequency` 做 sampled Hessian update。 | 声明 `Liuhong99/Sophia`。 | 已实现，但 Hessian sampling / scaling 绑定本地 training loop。可作为现有本地 Sophia baseline；改动应单独做 ablation。 |
| `adopt` | `src/optim/adopt.py`；首步初始化 second moment 并 return，后续使用 clipped normalized gradient 和 AdamW-style moments。 | 声明 `iShohei220/adopt`。 | 已实现。严格 parity 需要 source diff。 |
| `mars` | `src/optim/mars.py`；`main.py` 默认 approximate MARS，`optimize_1d=False`，1D 参数走 AdamW fallback。 | 声明 `AGI-Arena/MARS`。 | 已实现。与 all-parameter MARS 的差异是有意的 benchmark policy；需要报告 `mars_is_approx`、`mars_type`、1D fallback。 |
| `adafactor` | `src/optim/adafactor.py`；shape 允许时使用 factored second moment，可选 `beta1` first moment。 | 声明 jettify/pytorch-optimizer、fairseq reference、paper arXiv `1804.04235`。 | 已实现。其内部 relative-step / scale-parameter 行为会影响外部 `--lr` 和 scheduler 的解释；只有显式报告或单独 ablation 后才适合直接比较。 |
| `lamb` | `src/optim/lamb.py`；trust ratio 和可选 bias correction。 | 声明 official `cybertronai/pytorch-lamb`，paper arXiv `1904.00962`。 | 已实现。文件说明 paper v3 不用 debiasing；CLI flag 决定行为。只要记录 flag 即可接受。 |
| `scion` / `scion-light` | `src/optim/scion.py` 实现本地 partitioning 和 update logic。 | Source 标记为 `TBD`。 | 已实现但 provenance 未解决。补来源和 intended contract 前，只能作为 local baseline。 |
| `gn-prox` / `gn-full` | `src/optim/gn.py` 通过 training loop 选择，不是普通 optimizer step。 | 本地分支 feature。 | 已作为 experiment path 实现。应归类为 training-procedure variant，而不是 drop-in optimizer。 |

## 实验 Wiring 审计

实验实现是真实的，并且关键位置已有行为测试保护：

- CLI parsing 暴露 optimizer-specific arguments，并拒绝未知 `--opt`。
- `src/main.py` 会根据 optimizer name 构造不同 optimizer object。
- 需要 matrix / AdamW backup scheduler coordination 的 Muon-family optimizer 使用 `CombinedScheduler`。
- SophiaG 和 MARS 在 `src/optim/base.py` 中有显式 training-loop branch。
- 脚本命令由 `scripts/script_manifest.json` 跟踪；测试会捕获 command count 和 parseability drift。
- `softeq-k2000-muon` 已有 124M script 和 manifest entry；210M / 720M / MoE entry 暂时不加是有意设计。

## 最高风险差异

1. 现有本地 `muon` 是旧版 matrix-no-weight-decay class；`d-muon` 和 `softeq-k2000-muon` 会对 matrix 参数应用 weight decay。
2. `muon-pytorch` 是 PyTorch-native baseline，它的 grouping 和 LR policy 与本地 Muon 不同。
3. SophiaG Hessian update 行为属于本地 training loop 的一部分；比较既有 runs 时不应悄悄改动。
4. MARS 默认 approximate mode，并对 1D 参数使用 AdamW fallback。
5. GN variants 是 optimizer-procedure experiments，不是普通 optimizer replacement。
6. Scion provenance 仍是 `TBD`。
7. SoftEq K=2000 Muon 已有本地公式测试和 2-step Mac CPU CLI smoke，但仍需 CUDA smoke、数值稳定性检查、step 2000 附近 checkpoint-resume 验证。
8. SoftEq 参数拆分故意沿用本地 Muon shape heuristic：`p.ndim >= 2 and p.size(0) < 10000`，不是完整继承 model `group_specs` 的 no-decay partition。这可以作为 Muon-family benchmark policy 接受，但必须写进结果说明。
9. Adafactor 的 internal relative-step learning-rate policy 需要单独 review；否则不应把外部 scheduler 曲线直接视为与 AdamW-style optimizer 可比。

## 建议

当前分支适合继续做本地 CPU / CLI validation，并准备 GPU smoke jobs；benchmark 解释需要保持保守：

- 将 `softeq-k2000-muon` 称为 experimental user-provided variant；
- 新增自研或用户提供 optimizer 时优先放入 `src/optim/experimental/`，待来源、接口和实验语义稳定后再评估是否提升到 `src/optim/` 根目录；
- 未做 external source diff 的 optimizer 不要写成 official reproduction；
- full 124M 前先跑 tiny CUDA smoke；
- 124M 路径稳定后，再添加更大规模 SoftEq 脚本。
