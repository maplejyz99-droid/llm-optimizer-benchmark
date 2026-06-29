# Optimizer Benchmark 隔夜报告

日期：2026-06-29

## 范围

仓库：
`/Users/jiujiujiu/Documents/New-project/llm-optimizer-benchmark-newton-muon`

本轮工作聚焦在新增 `softeq-k2000-muon` 之后，检查 optimizer benchmark 是否已经具备继续实验的基本条件。本轮没有 push、merge，也没有修改父目录下的无关项目。

## 已完成

- 新增独立优化器变体：`--opt softeq-k2000-muon`。
- 实现 SoftEq-0.5 前 2000 个 optimizer step 生效、12 步 Newton-Schulz 正交化、矩阵参数 weight decay、AdamW backup，以及 optimizer state 中的 cutoff step 持久化。
- 新增 124M shell 入口和 `scripts/script_manifest.json` 条目。
- 新增行为测试，覆盖 CLI 解析、optimizer 装配、scheduler 路由、SoftEq 公式、cutoff 行为、state dict roundtrip、script manifest 计数。
- 增加 dense-only guard：SoftEq v1 在训练前拒绝 MoE 和非 `llama` / `mup_llama` 模型族。
- 修复 tied embedding 下参数日志统计：改用模型自身 `get_num_params(non_embedding=True)` 口径，避免把 tied 的 `wte` 和 `lm_head` weight 重复扣除。
- 修复 `SoftEqK2000Muon.load_state_dict()`：不再修改调用方传入的 state dict。
- 修复 optimizer group 日志：AdamW backup 参数行现在显示实际 backup learning rate，不再显示矩阵 learning rate。
- 增加启动 guard：schedule-free optimizer 必须使用 `--scheduler none`，与其内部调度契约一致。
- 更新 README 和 SoftEq 接入文档。

## 已验证事实

本轮本地平台：

- macOS / Darwin arm64
- shell：`zsh`
- 本地 PyTorch 可用；CUDA 不可用

以下命令已成功执行：

```bash
python3 -m py_compile src/optim/experimental/softeq_muon.py src/main.py src/config/base.py
bash -n scripts/124m/softeq-k2000-muon.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
git diff --check
```

全量测试结果：

```text
Ran 109 tests ... OK
```

Mac CPU 上真实 CLI smoke 已跑通：

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

观察结果：dataset load、model build、optimizer build、2 个 optimizer iterations、iteration 0 和 2 的 validation、`summary.json` 生成都已完成。

负向验证也已通过：`--model base --opt softeq-k2000-muon` 会在训练前明确报 unsupported model，不会进入训练流程。

## 合理推断

- 新 optimizer 不是 CLI 空壳；它已经在 CPU 上走过真实训练路径。
- 124M shell 入口可解析，并被 manifest 测试保护。
- CUDA smoke 通过后，SoftEq 可以作为本地 experimental Muon-family variant 参与对比。
- 当前还不能把它描述为 official paper reproduction，因为尚未验证到这个精确变体的公开官方实现或论文来源。

## 剩余风险

| 风险 | 状态 | 说明 |
| --- | --- | --- |
| CUDA / BF16 行为 | 未验证 | 这台 Mac 没有 GPU。 |
| NaN / OOM 行为 | 未验证 | 需要 tiny CUDA 和 124M CUDA smoke。 |
| 跨 step 2000 的 checkpoint resume | 未验证 | optimizer state roundtrip 已测，但完整训练 checkpoint resume 仍需要 CUDA 或更长 smoke。 |
| 2-GPU update sharding | 未验证 | 代码有 environment-rank sharding，但本地没有多 GPU 验证条件。 |
| SoftEq 官方来源 | 未验证 | 当前来源是用户提供的公式和文件，公开来源未确认。 |
| 小词表 embedding 分支 | 已知行为 | 本地 Muon heuristic 会把 tiny embedding 归为 matrix 参数；正式 Llama / FineWeb 大词表形状可避开该 smoke-only 行为。 |
| `--muon_ns_steps` 对 SoftEq 的影响 | 已知行为 | SoftEq 故意固定使用 `Orth_12`；共享 CLI 参数不会改变该变体。 |
| Adafactor LR 语义 | 待决 | Adafactor 有内部 relative-step / scale-parameter 行为；外部 scheduler 的可比性需要单独决策。 |

## 远程 GPU Smoke 计划

先用小数据或缓存数据跑通；optimizer 路径稳定前保持 W&B disabled。

```bash
WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 ./src/main.py \
  --config_format base --model llama --distributed_backend nccl \
  --dataset shakespeare-char --datasets_dir /tmp/llmopt-datasets \
  --results_base_folder /tmp/llmopt-exps \
  --experiment_name cuda_tiny_softeq \
  --device cuda:0 --dtype bfloat16 \
  --n_embd 128 --n_head 4 --n_layer 2 \
  --batch_size 4 --sequence_length 128 --acc_steps 1 \
  --iterations 20 --warmup_steps 2 \
  --eval_interval 10 --eval_batches 2 --latest_ckpt_interval 10 \
  --opt softeq-k2000-muon --scheduler none \
  --lr 1e-3 --muon_lr_factor 1e-2 \
  --weight_decay 0.1 --beta1 0.8 --beta2 0.999 \
  --momentum 0.95 --grad_clip 0.5 --log_interval 1
```

然后跑 124M BF16 checkpoint / resume smoke：

```bash
WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 ./src/main.py \
  --config_format base --model llama --distributed_backend nccl \
  --dataset fineweb --datasets_dir ./src/data/datasets \
  --results_base_folder /tmp/llmopt-exps \
  --experiment_name cuda_124m_softeq_resume \
  --device cuda:0 --dtype bfloat16 \
  --n_embd 768 --n_head 12 --n_layer 12 \
  --batch_size 16 --sequence_length 512 --acc_steps 1 \
  --iterations 100 --warmup_steps 10 \
  --eval_interval 50 --eval_batches 2 --latest_ckpt_interval 50 \
  --opt softeq-k2000-muon --scheduler cos \
  --lr 1e-3 --muon_lr_factor 1e-2 \
  --weight_decay 0.1 --beta1 0.8 --beta2 0.999 \
  --momentum 0.95 --grad_clip 0.5 --log_interval 10

WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 ./src/main.py \
  --config_format base --model llama --distributed_backend nccl \
  --dataset fineweb --datasets_dir ./src/data/datasets \
  --results_base_folder /tmp/llmopt-exps \
  --experiment_name cuda_124m_softeq_resume \
  --device cuda:0 --dtype bfloat16 \
  --n_embd 768 --n_head 12 --n_layer 12 \
  --batch_size 16 --sequence_length 512 --acc_steps 1 \
  --iterations 150 --warmup_steps 10 \
  --eval_interval 50 --eval_batches 2 --latest_ckpt_interval 50 \
  --opt softeq-k2000-muon --scheduler cos \
  --lr 1e-3 --muon_lr_factor 1e-2 \
  --weight_decay 0.1 --beta1 0.8 --beta2 0.999 \
  --momentum 0.95 --grad_clip 0.5 --log_interval 10
```

## 实验矩阵

| Gate | 模型 | Optimizer | 目的 |
| --- | --- | --- | --- |
| 0 | tiny CUDA | SoftEq | 隔离 CUDA / BF16 / NaN / OOM 问题。 |
| 1 | 124M smoke | SoftEq | 检查 BF16、checkpoint、resume、logging。 |
| 2 | 124M full | AdamW, Muon, SoftEq | 主对照实验。 |
| 3 | 124M LR sweep | SoftEq only | sweep `muon_lr_factor`：`5e-3`, `1e-2`, `2e-2`。 |
| 4 | 124M seeds | AdamW, Muon, best SoftEq | 用 seeds `0`, `1`, `2` 做确认。 |
| 5 | 210M | AdamW, Muon, best SoftEq | 规模迁移检查。 |
| 6 | 720M | AdamW, Muon, best SoftEq | 前面 gate 稳定后再跑大模型。 |

## 失败排查顺序

1. 先用 tiny CUDA 和 disabled W&B 复现。
2. 关闭 scheduler：`--scheduler none`。
3. 降低 `--muon_lr_factor`：`1e-2 -> 5e-3 -> 2e-3`。
4. OOM 时先降 batch size，再降 sequence length。
5. 使用新的 `--experiment_name`，避免误触 auto-resume。
6. 对比同 shape 的 `muon`；如果两者都失败，先查数据、环境、脚本；如果只有 SoftEq 失败，再查 SoftEq update norm 和 cutoff state。
