# LLM Optimizer Benchmark

**面向大语言模型预训练的优化器研究与可复现实验平台。**

**简体中文** · [English](README.en.md)

[实验配置](#实验配置) · [快速开始](#快速开始) · [运行与结果](#运行与结果) · [文档与开发](#文档与开发)

我们在统一的 Llama 训练框架中维护优化器基线、研究变体和实验配方，围绕验证损失、训练稳定性、计算开销与跨规模行为开展比较。当前工作涵盖 Muon 系列、Gauss–Newton 方法、预条件与参数选择实验，以及从 124M 到 1B 的 dense Llama 配置。

这个仓库把算法实现和实验管理放在一起：每次运行记录代码版本、数据身份、有效优化配置、参数更新分组和评估协议，让曲线能够追溯到具体的训练过程。

## 当前维护重点

- **优化器研究**：GN-Prox / GN-Full、Newton-Muon、SoftEq K=2000 Muon，以及 AdamW / Muon 的 Magma 变体。
- **跨规模比较**：124M、210M、720M 和 1B dense Llama 配方，另有 520M MoE 实验入口。
- **参数化实验**：标准参数化与 muP 的 Llama 路径，配套参数组检查、坐标检查和学习率扫描工具。
- **可追溯运行**：FineWeb-30B 数据预检、统一启动器、版本化运行清单、评估记录、checkpoint 状态和行为测试。

实现入口、脚本配置与完整训练结果是不同层次的证据。各方法的适用范围见下文；性能结论应以对应的运行产物为准。

## 优化器与研究路径

### 比较基线

当前 CLI 提供 AdamW、Muon、Distributed Muon、SOAP、SophiaG、Lion、AdEMAMix、ADOPT、MARS、Prodigy、Schedule-Free、Adafactor、LAMB 等选项。完整列表与参数以 [`src/config/base.py`](src/config/base.py) 为准，实际构造与参数分组见 [`src/main.py`](src/main.py)。

同名方法在不同实现中的参数分组、回退更新器和学习率策略可能不同。比较时应记录本仓库的实际实现；已有差异整理在[优化器实现审计](docs/optimizer-implementation-audit.md)中。

### 研究变体

- **GN-Prox / GN-Full**：通过线性化或 Gauss–Newton 近似构造内层更新，使用 `--opt gn-prox` / `--opt gn-full`。当前训练入口限制为单设备。见 [GN-Full 实现说明](docs/gn-full-onepager.md)。
- **Newton-Muon**：在 Muon 路径中加入激活协方差右预条件，使用 `--opt newton-muon`。当前仅支持单设备 dense `llama`。见 [Newton-Muon 说明](docs/newton-muon-onepager.md)。
- **SoftEq K=2000 Muon**：使用 `--opt softeq-k2000-muon`，支持 dense `llama` / `mup_llama`。作为实验变体维护，正式比较前需完成 GPU 与 cutoff 附近的恢复验证。见 [SoftEq 接入说明](docs/softeq-k2000-muon-integration.md)。
- **Magma**：使用 `--opt adamw-magma` / `--opt muon-magma` 研究参数选择干预。当前训练入口限制为单设备。

## 实验配置

- [124M](scripts/124m/)：基线比较、Newton-Muon / SoftEq 入口及显存探测脚本。
- [210M](scripts/210m/) 与 [720M](scripts/720m/)：更大模型的优化器配方。
- [1B](scripts/1b/)：17 个优化器变体的 dense Llama 配方，与 720M 配方集合对应。
- [520M MoE](scripts/moe-520m/)：MoE 模型的实验配方；使用前仍需核对对应优化器的模型与分布式限制。
- [muP 学习率扫描](scripts/repro/mup_readme_lr_sweep/README.md)：基于本仓库 Llama 训练栈的标准参数化 / muP 比较。

**1B 比较配置**采用 24 层、隐藏维度 1792、14 个注意力头与 tied embeddings，共 **1,026,086,656 个可训练参数**。序列长度为 512，全局 batch 为 1984 条序列，每次更新处理 **1,015,808 个 token**。

1B 配方沿用 720M 超参数作为起点，尚不能据此宣称完成 1B 调优或获得优化器排名。脚本中的 48,000 次更新是可编辑的默认值；8,000、16,000 与 20,203 次更新的比较约定、训练预算及协作接入要求见 [1B 实验指南](docs/1b-benchmark.md)。

## 快速开始

### 1. 安装

训练以 Python 3.10 和 Linux / CUDA 环境为主；CPU 可用于小模型功能检查。

```bash
git clone https://github.com/maplejyz99-droid/llm-optimizer-benchmark.git
cd llm-optimizer-benchmark

conda create -n llmopt python=3.10 -y
conda activate llmopt
python -m pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) 是基础依赖列表，并非完整的 CUDA 环境锁定文件。请按机器配置匹配的 PyTorch / CUDA；[`requirements-ci.lock`](requirements-ci.lock) 专用于 Linux x86_64 / Python 3.10 的 CPU 行为测试。环境记录与复现边界见[复现说明](docs/reproducibility.md)。

### 2. 跑通一个小模型

以下示例在 CPU 上执行 2 次 AdamW 更新，检查数据加载、模型构建、训练与评估链路。首次使用会下载 Tiny Shakespeare 数据，不需要 W&B。

```bash
python ./src/main.py \
  --config_format base --model llama --opt adamw \
  --dataset shakespeare-char --device cpu --dtype float32 \
  --n_layer 1 --n_head 2 --n_embd 64 --vocab_size 96 \
  --sequence_length 32 --batch_size 2 --acc_steps 1 \
  --iterations 2 --warmup_steps 1 --scheduler none \
  --eval_interval 2 --eval_batches 1 --final_eval_batches 1 \
  --latest_ckpt_interval 0 --permanent_ckpt_interval 0 \
  --results_base_folder ./exps/readme-smoke \
  --experiment_name adamw-cpu-smoke
```

重复运行时请使用新的 `--experiment_name`。这个示例用于功能检查，正式性能比较应在相同的 CUDA 环境和实验协议下进行。

### 3. 准备比较数据

当前 FineWeb 比较使用版本化的 **FineWeb-30B** 数据：GPT-2 tokenizer、`uint16` token 文件、30B 训练 token 与 100M 验证 token。数据目录需要包含 `train.bin`、`val.bin` 和 `meta.json`。

```bash
export LLMOPT_DATASETS_DIR=/path/to/fineweb-30B
python ./scripts/data/check_fineweb_30b.py --json
```

将示例路径替换为实际数据目录，也可以指向包含 `fineweb-30B/` 的父目录。训练参数 `--datasets_dir` 优先于环境变量。预检核对元数据与文件大小，不扫描整个 token 文件；训练默认不会自动下载缺失的 FineWeb 数据。需要构建数据时，先查看 [`scripts/data/build_fineweb_30b.py`](scripts/data/build_fineweb_30b.py) 的参数并单独准备。

## 运行与结果

推荐通过启动器管理新实验，把原始运行产物与整理后的结果分别放在指定目录中：

```bash
python ./scripts/setup_runtime_paths.py \
  --runs-root /path/to/large-disk/llmopt/runs \
  --results-root /path/to/curated/llmopt-results
```

该命令创建被 Git 忽略的 `runs/`、`results/` 本地链接。完成数据预检后，可先预览一个 124M AdamW 的 20 步 CUDA 功能检查命令：

```bash
python ./scripts/launch_run.py \
  --runs-root /path/to/large-disk/llmopt/runs \
  --suite smoke/fineweb30b \
  --run-id adamw-124m-20steps-seed0 \
  --gpu-ids 0 --monitor-gpu --dry-run \
  -- \
  --config_format base --model llama --opt adamw \
  --dataset fineweb --device cuda:0 --dtype bfloat16 \
  --n_layer 12 --n_head 12 --n_embd 768 \
  --sequence_length 512 --batch_size 1 --acc_steps 1 \
  --iterations 20 --warmup_steps 2 --scheduler cos \
  --lr 1e-3 --weight_decay 0.1 --seed 0 --data_seed 1337 \
  --eval_interval 10 --eval_batches 1 --final_eval_batches 1
```

`--dry-run` 只打印命令，不启动训练或创建运行目录。确认路径、环境和显存后，移除该选项即可执行。此处 batch、训练长度和评估上限仅用于功能检查；正式比较请采用统一的[实验配方](#实验配置)与评估协议。

每次运行使用新的 `--run-id`。启动器管理 `--results_base_folder` 和 `--experiment_name`，不要在分隔符 `--` 后重复传入这两个参数。

运行产物包括：

- `run_manifest.json`：配置、代码与数据身份、有效优化计划、实际参数更新分组。
- `summary.json` 与 `evaluations/`：训练指标、验证指标及版本化评估记录。
- `launch/` 与 `logs/`：启动命令、生命周期状态、终端日志和可选 GPU 采样。
- `ckpts/`：在启用 checkpoint 间隔时保存的训练状态。

启动器会核对运行清单与结果文件的一致性后再标记完成。后台运行、多 GPU、输出结构和恢复限制见[运行指南](docs/running-benchmark.md)。

## 比较约定

- 固定数据版本、模型配置、训练 token 预算、seed 集合与评估协议，同时报告实际执行配置和计算开销。
- 本仓库 CLI 的 `--batch_size` 与 `--acc_steps` 的乘积已表示全局序列 batch；DDP 会在进程间分配，**不要再乘 GPU 数量**。该乘积需能被进程数整除。
- GN、Newton-Muon 与 Magma 当前走单设备入口；通过启动器使用时，省略 `--nproc-per-node` 和 `--distributed_backend`。配方使用 `torchrun` 本身不代表分布式语义已全面验证。
- Schedule-Free 方法使用 `--scheduler none`。不同 Muon 变体的矩阵更新、weight decay 和 AdamW 回退策略应分别记录。
- 现有 shell 配方含 W&B 占位参数，运行前需填写或移除相应选项。它们不会转发附加位置参数；修改训练长度时应编辑命令，不能仅在 `bash scripts/1b/adamw.sh` 后追加选项。
- 断点恢复需满足数据、配置、运行身份和 checkpoint 状态契约。启动器每次创建新运行，直接训练入口的恢复用法见[复现说明](docs/reproducibility.md)。

## 文档与开发

- [1B 实验指南](docs/1b-benchmark.md)：模型配置、训练预算、启动示例与新增优化器接入清单。
- [运行指南](docs/running-benchmark.md)：启动器、日志、输出目录、GPU 监控与 checkpoint 边界。
- [复现说明](docs/reproducibility.md)：环境记录、CPU 依赖锁定、运行身份、恢复与 CI。
- [优化器实现审计](docs/optimizer-implementation-audit.md)：本地实现与来源声明、已知差异及验证范围。
- [muP 学习率扫描](scripts/repro/mup_readme_lr_sweep/README.md)：参数化比较与绘图流程。

实现主要位于 [`src/models/`](src/models/)、[`src/optim/`](src/optim/)、[`src/config/`](src/config/) 与 [`src/main.py`](src/main.py)；运行身份由 [`src/run_manifest.py`](src/run_manifest.py) 管理。新增优化器时，请同时更新 CLI、构造逻辑、有效优化计划、实验配方与行为测试，接入细节见 1B 实验指南。

在已按复现说明安装依赖的 **Linux CPU 测试环境**中运行：

```bash
python repro/verify_cpu_environment.py
python repro/run_behavior_tests.py full
python repro/run_behavior_tests.py isolated
python repro/run_behavior_tests.py reverse
```

测试分别覆盖整体运行、模块隔离与逆序执行。分布式集成测试和 CI 配置见 [`tests/integration/`](tests/integration/) 与 [GitHub Actions](https://github.com/maplejyz99-droid/llm-optimizer-benchmark/actions)。CPU 行为测试通过不等同于完成 CUDA 性能或收敛验证。

代码与文档改进请通过[本仓库 Pull Requests](https://github.com/maplejyz99-droid/llm-optimizer-benchmark/pulls)提交与讨论。

## 来源、许可与引用

本项目在 [EPFL 的 llm-optimizer-benchmark](https://github.com/epfml/llm-optimizer-benchmark) 基础上持续开发，继承了其训练框架与基线实现，并在此基础上维护本仓库的研究变体、实验配置和复现工具。感谢上游作者，以及 [llm-baselines](https://github.com/epfml/llm-baselines) 和 [nanoGPT](https://github.com/karpathy/nanoGPT) 的贡献。

项目采用 [Apache-2.0 许可证](LICENSE)，保留上游与第三方实现的来源说明。使用上游论文的基准设计或结果时，请引用原论文；使用本仓库新增实现时，请同时注明仓库地址与具体 commit。

上游论文：[Benchmarking Optimizers for Large Language Model Pretraining](https://arxiv.org/abs/2509.01440)，Andrei Semenov、Matteo Pagliardini、Martin Jaggi，2025。

```bibtex
@article{semenov2025benchmarking,
  title={Benchmarking {O}ptimizers for {L}arge {L}anguage {M}odel {P}retraining},
  author={Semenov, Andrei and Pagliardini, Matteo and Jaggi, Martin},
  journal={arXiv preprint arXiv:2509.01440},
  url={https://arxiv.org/abs/2509.01440},
  year={2025}
}
```
