# SoftEq K=2000 Muon 接入说明

## 范围

本分支新增 `--opt softeq-k2000-muon`，作为 benchmark 中的一个 optimizer variant。
它基于用户提供的 `softeq_k2000.py` 和公式截图实现，不是已验证过的公开官方实现。
因此代码放在 `src/optim/experimental/`，用于区分自研 / 用户提供的实验 optimizer 与 `src/optim/` 根目录下的第三方或正式 baseline。

当前支持：

- dense `llama` / `mup_llama` 运行
- 单进程运行，以及现有 Muon-style environment-rank update sharding 代码路径
- 对被本地 Muon heuristic 排除的参数使用 AdamW backup
- 通过 optimizer state dict 保存 / 恢复 SoftEq cutoff step

当前非目标：

- 在未验证公开来源前宣称 paper-equivalent
- 替换现有 `muon`、`d-muon` 或 `muon-pytorch` baseline
- 在 124M smoke 经 GPU 验证前添加 210M / 720M / MoE 脚本
- 第一版接入中支持 MoE 或非 Llama 模型族

## 算法

对于矩阵参数，每个 optimizer step 先计算：

```text
M_t = mu M_{t-1} + (1 - mu) G_t
U_t = (1 - mu) G_t + mu M_t
```

前 `2000` 个 optimizer step：

```text
U_tilde[i, :] = U_t[i, :] / max(||U_t[i, :]||_2, eps)^0.5
D_t = sqrt(max(1, m / n)) * Orth_12(U_tilde)
```

从 step `2000` 开始：

```text
D_t = sqrt(max(1, m / n)) * Orth_12(U_t)
```

矩阵参数更新使用 decoupled weight decay：

```text
W_{t+1} = (1 - lr * weight_decay) W_t - lr * D_t
```

非 Muon 参数沿用本地 Muon baseline 的 AdamW backup 契约，使用 `adamw_lr=args.lr`，AdamW betas 来自 `--beta1/--beta2`。

## 代码路径

- `src/optim/experimental/softeq_muon.py`：optimizer 实现和 SoftEq helpers
- `src/config/base.py`：将 `softeq-k2000-muon` 加入 `--opt`
- `src/main.py`：构造 `SoftEqK2000Muon`，并注册为 Muon-family scheduler 使用者
- `scripts/124m/softeq-k2000-muon.sh`：124M benchmark 入口
- `scripts/script_manifest.json`：受行为测试保护的脚本 manifest 条目
- `tests/behavior/test_softeq_muon.py`：公式、cutoff、state dict 行为测试
- `tests/behavior/test_optimizer_assembly.py`：main assembly 和 scheduler contract 测试

## Benchmark 语义

`softeq-k2000-muon` 故意继承以下本地 benchmark 语义：

- 矩阵参数选择沿用本地 Muon：`p.ndim >= 2 and p.size(0) < 10000`
- norm 和 1D 参数使用 AdamW backup
- 大 embedding 和 head 通过本地 Muon shape heuristic 进入 AdamW backup；tiny 小词表 embedding 仍可能被归为 matrix 参数，应只视为 smoke-test-only 行为
- `mup_llama` 会按 `n_embd / scale_base_model` 缩放 matrix learning rate，匹配现有本地 Muon 工程策略
- 非 `none` scheduler 使用 `CombinedScheduler`，与 `muon`、`muon-magma`、`newton-muon` 一致
- `main.py` 会拒绝 MoE 和非 `llama` / `mup_llama` 模型

与现有本地 `Muon` 的关键差异：该变体按用户给定公式处理 momentum：

```text
M_t = mu M_{t-1} + (1 - mu) G_t
U_t = (1 - mu) G_t + mu M_t
```

现有本地 `Muon` class 使用历史 Nesterov-style buffer update 和 5-step quintic Newton-Schulz map。该 baseline 保持不变。

## 124M 入口

在 CUDA 机器和 benchmark 环境中，从仓库根目录运行：

```bash
bash scripts/124m/softeq-k2000-muon.sh
```

脚本使用 124M Muon-family 形状：

- `n_embd=768`
- `n_head=12`
- `n_layer=12`
- `sequence_length=512`
- `batch_size=64`
- `acc_steps=4`
- `iterations=128000`
- `warmup_steps=2000`
- `muon_lr_factor=1e-2`
- `momentum=0.95`

脚本保留 benchmark W&B placeholders。真实运行前需要替换 `YOUR_WANDB-PROJECT` 和 `YOUR-WANDB-ENTITY`。

## 本地验证

本分支在 Mac / CPU 上跑过：

```bash
python3 -m py_compile src/optim/experimental/softeq_muon.py src/main.py src/config/base.py
bash -n scripts/124m/softeq-k2000-muon.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
```

观察结果：

```text
Ran 109 tests ... OK
```

额外真实 CLI smoke：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ./src/main.py \
  --config_format base --model llama --dataset shakespeare-char \
  --datasets_dir /tmp/llmopt-codex-datasets \
  --results_base_folder /tmp/llmopt-codex-results \
  --experiment_name newton-muon-softeq-cpu-smoke \
  --device cpu --dtype float32 \
  --n_layer 1 --n_head 2 --n_embd 32 --sequence_length 16 \
  --batch_size 2 --acc_steps 1 --iterations 2 --warmup_steps 1 \
  --eval_interval 2 --eval_batches 1 \
  --latest_ckpt_interval 0 --permanent_ckpt_interval 0 \
  --opt softeq-k2000-muon --scheduler none \
  --lr 1e-3 --muon_lr_factor 1e-2 \
  --momentum 0.95 --weight_decay 0.1 --grad_clip 1.0
```

观察结果：完成 `2` 个 optimizer iterations，生成 `summary.json`，并在 iteration `0` 和 `2` 跑了 validation。

迁移到 `llm-optimizer-benchmark-newton-muon` 后已再次运行同等 Mac CPU smoke，实验名为 `newton-muon-softeq-cpu-smoke`，确认新 import path 能完成 dataset load、model build、optimizer build、2 step training 和 `summary.json` 生成。

## 仍需 GPU 验证

在把它当作真实 benchmark 结果前，需要跑：

1. Tiny CUDA smoke：关闭 W&B，`scheduler none`，使用很小模型。
2. 124M single-GPU BF16 smoke：20-100 steps。
3. 124M single-GPU BF16 smoke：启用 W&B 和 checkpoint。
4. 可选 2-GPU functional smoke：只在单 GPU 行为稳定后再跑。

在 124M GPU smoke 确认无 NaN、无 OOM、checkpoint 能正确跨 `2000` step cutoff resume、W&B keys 符合预期之前，不要把速度或最终 loss 拿去和 `muon` 做正式比较。
