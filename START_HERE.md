# Ascend 集群 Codex 执行入口

> **公开指南与私有完整包。** 完整 v3 ZIP 已上传 ModelScope；固定版本、下载命令和权限说明见 [DOWNLOADS.md](DOWNLOADS.md)。当前接手阻塞项及解除条件见 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)。GitHub 可公开阅读，完整包与训练数据仍需各自仓库的读取权限。

目标：复现 ECR 最佳 Xiaomi XR‑1 第二轮 Continue s1，并与原 GPU 的 **455/500（91.0%）** 比较效果和耗时。默认只迁移最后一阶段：加载固定的 **A3 s1 续训 20k 父模型**，再训练 **10,000 optimizer steps**。完整科学配方和历史证据见 [README.md](README.md)。

先按 [DOWNLOADS.md](DOWNLOADS.md) 下载云端完整包，将**整个 ZIP 解压目录**放到集群；不能只转发这一个 Markdown，否则会缺源码、配置、下载清单和校验工具。v1/v2 保留，本版为 v3。

## 给接手 Codex 的指令

> 请执行这份交接包。先读 START_HERE.md、README.md 和 UPLOAD_REPORT.md，再依次进行本机环境探测、下载已就绪输入、逐文件核验、NPU 适配、数值检查、20 步冒烟、600 步吞吐试跑、正式 10k、dev200 选点和 test500 对比。原始 GPU 参考源码保持原样，适配写入独立工作目录。通过前一阶段后自主继续；普通路径选择、安装隔离环境、适配和修复无需反复询问。使用平台已分配的设备，遵守该集群的调度和访问规则。遇到缺失数据、私有仓库读取权限或缺少评测节点等外部条件时，记录准确缺项并推进不依赖它的工作；不要虚构输入，也不要把候选代码当成已验证移植。每个阶段保存日志、配置、代码哈希、起止时间和结果。不得静默更换训练起点、比例、loss 或测试 seed，也不得把 Fresh 对照当作 Continue 复现。

这里的“立即开始”指接手后立即执行准备和适配；**当前已核对的上传报告不支持直接启动完整 10k**：R2 recovery 和 20k 父模型权重尚无云路径。详见 [缺失输入交付单](MISSING_INPUTS.md)。这不是要求重新批准实验，而是缺少实际文件。

## 只有 GitHub 链接时先做什么

先按 [DOWNLOADS.md 第 0 节](DOWNLOADS.md#0-先取得完整包并验证读取权限) 验证 12 个私有仓库的小文件访问，再按第 5 节下载并校验完整 ZIP。GitHub 根目录记为 `$ECR_GUIDE`，完整包根目录记为 `$ECR_BUNDLE`；以下 `scripts/`、`manifests/` 和 `reference/` 命令全部在 `$ECR_BUNDLE` 内执行。无需先取得私有训练源码仓库。开始前阅读 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)，逐项记录当前机器的权限、输入和环境状态。

## 0 先核对包和本机环境

先 `cd "$ECR_BUNDLE"`，在交接包根目录执行；`ECR_ROOT` 选为用户有权限的高速存储目录，可由集群 Codex 自动选择。下面默认复用下载阶段的 `ECR_ROOT`；未设置时使用 `$HOME/ecr-ascend`。这是当前机器的恢复目录，不是历史服务器挂载。

```bash
set -euo pipefail
export ECR_ROOT="${ECR_ROOT:-$HOME/ecr-ascend}"
sha256sum -c SHA256SUMS > bundle-check.log
python3 scripts/cluster_preflight.py --root "$ECR_ROOT" --output cluster-preflight.json
```

读取 `cluster-preflight.json`，确认 CPU 架构、NPU 型号、设备可见性、PyTorch / torch-npu / CANN 和空间。预检查不会申请卡，也不会开训。优先使用平台已配套的镜像；不要覆盖系统驱动或全局 Python。训练环境和下载环境分开。R1 单独 49.79 GB，R2 约 5.56 GB，父模型约 10.12 GB；还要计入 Human、SDK 缓存、3 个导出及导出临时副本。Human 下载清单没有可靠体积，不能仅凭上述三项之和预留磁盘。

## 1 下载已经可以定位的输入

```bash
python3.12 -m venv .venv-download
.venv-download/bin/python -m pip install 'modelscope-hub==0.6.1'
read -rsp 'ModelScope read token: ' MODELSCOPE_API_TOKEN
export MODELSCOPE_API_TOKEN

.venv-download/bin/python scripts/download_files.py \
  manifests/human10-download.jsonl --root "$ECR_ROOT" --workers 4
.venv-download/bin/python scripts/unpack_human.py --root "$ECR_ROOT"

.venv-download/bin/python scripts/download_files.py \
  manifests/round1-download.jsonl.gz --root "$ECR_ROOT" --workers 8
.venv-download/bin/python scripts/verify_recovery.py \
  --root "$ECR_ROOT" --round round1 --output "$ECR_ROOT/checks/round1-integrity.json"
```

R1 下载清单含 **205,003 个文件**，已从冻结云映射完整提取。它使用 `master` 加固定逐文件 SHA256 和字节数：这是内容锁定，不是整个仓库的不可变 revision。SDK 缓存外仍会核验实际文件，遇到内容不一致立即报错；不会退而接受最新内容。重新执行会核验已有文件，并继续缺失项。私有仓库 token 由集群的正常凭据机制提供；不写入文档、命令行参数或训练日志。

需要原版权重做算子诊断或完整 30k 路线时，使用已冻结的清单：

```bash
.venv-download/bin/python scripts/download_files.py \
  manifests/xiaomi-base-download.jsonl --root "$ECR_ROOT" --workers 4
```

原版可以用于适配诊断；正式 Continue 的父模型仍必须是 20k 那一份。

## 2 接入第二轮数据和父模型

当前上传报告只证明旧归档完成；**不能从旧的 3,000 局 rollout 重新“推导”出 R2 recovery，也不能拿最终 30k best 权重替代 20k 父模型。**

取得新上传的 repo / 文件映射后，按 [MISSING_INPUTS.md](MISSING_INPUTS.md) 生成 `manifests/round2-parent-download.jsonl`。本文件必须包含 R2 整个源目录（包括原始 `MANIFEST.sha256`、`meta.json`、README）和父模型全部 22 文件。用本文提供的 `download_files.py` 执行，并用冻结 manifest 和父模型 pins 验收；不要人工复制一个不完整子集。

```bash
.venv-download/bin/python scripts/download_files.py \
  manifests/round2-parent-download.jsonl --root "$ECR_ROOT" --check-only
.venv-download/bin/python scripts/download_files.py \
  manifests/round2-parent-download.jsonl --root "$ECR_ROOT" --workers 8
.venv-download/bin/python scripts/verify_recovery.py \
  --root "$ECR_ROOT" --round round2 --output "$ECR_ROOT/checks/round2-integrity.json"
.venv-download/bin/python scripts/prepare_config.py \
  --root "$ECR_ROOT" --out "$ECR_ROOT/configs/round2-502525.json"
unset MODELSCOPE_API_TOKEN
```

`prepare_config.py` 会核验 22 个父模型文件，并把固定的 41 个数据根目录改为本机路径。已有结果文件不覆盖：复查已有回执，或用独立 attempt 目录保存新结果；不要删除历史回执来使命令通过。

## 3 获取固定上游并生成 NPU 适配候选

取得配套私有交接包后，可使用其中保存的冻结训练源码，无需访问历史训练节点。另取公开 XR‑1 辅助代码：

```bash
mkdir -p "$ECR_ROOT/code"
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1 \
  "$ECR_ROOT/code/Xiaomi-Robotics-1"
git -C "$ECR_ROOT/code/Xiaomi-Robotics-1" checkout --detach \
  0dd7aef8dc87296246aae812a1f59ccb708e5546
test "$(git -C "$ECR_ROOT/code/Xiaomi-Robotics-1" rev-parse HEAD)" = \
  0dd7aef8dc87296246aae812a1f59ccb708e5546
python3 scripts/prepare_npu_candidate.py --root "$ECR_ROOT"
```

若目标 checkout 已存在，只核验 commit 和工作树，不重新 clone 或覆盖。候选生成器拒绝覆盖已存在的适配目录。

`code/ecr-ascend/npu-port.diff` 清楚列出 CUDA→NPU、NCCL→HCCL、NPU device mesh、attention 参数和受限步数运行的改动；原 flow objective、LR、采样和参数组保持不变。**它不是已通过实机验证的 NPU trainer。** Codex 应检查下载模型中的自定义 attention，处理 SDPA / eager 支持、FFT 前后向、FSDP2 / DTensor、AdamW foreach 及 HF 导出问题。源码中存在 SDPA 参数不等于内层已没有 CUDA/FlashAttention 调用。不得删掉 FFT loss 来“跑通”。

依赖按 README 第 5 节安装到平台训练环境。除配套 TorchNPU 外，保留 Transformers 4.57.1、NumPy 2.1.3、Pillow 11.3.0、safetensors 0.6.2、einops 0.8.1，以及 av / pandas / pyarrow 版本；不要在 Ascend 安装 CUDA flash-attn wheel。若平台依赖需要变更，记录差异并验证。

以下命令中的 `python` 必须是**训练环境**的 Python：

```bash
python "$ECR_ROOT/code/ecr-ascend/scripts/round2_input_check.py" \
  --config "$ECR_ROOT/configs/round2-502525.json" \
  --output "$ECR_ROOT/checks/input-check.json"
```

该检查核对 41 个根目录、帧数/episode 数、三相机、state/action 形状，并解码每源一个 episode；完整内容哈希由前面的两个 integrity 检查负责。

## 4 冒烟与正式训练

以下命令在平台**已分配的单节点设备**内执行；脚本不申请资源，不接管其他作业。4 卡布局从 micro-batch=12 开始；如需 8 卡，可用 micro-batch=6 保持 accumulation=4、global batch=192。低显存时降低 micro-batch 并增加累积，记录在对比报告。不要默认所有可见 NPU 都可占用。

注意：包内 `launch_stage.py` 的 smoke、benchmark、train 三种模式都要求 R1/R2 完整性回执和完整输入检查通过；缺 R2 或父模型时不能用这个入口冒烟，只能先做环境和独立算子诊断。固定 batch/noise/time 的 GPU 数值参考尚未作为现成 fixture 随包交付，需按 [HANDOFF_STATUS.md](HANDOFF_STATUS.md) 与 GPU 评测执行方生成并记录。先完成数值比较，再启动 20 步。没有 GPU 数值参考时，先做可执行诊断并记录缺项，不虚构 numerical parity：

```bash
python scripts/launch_stage.py smoke --root "$ECR_ROOT" \
  --run "$ECR_ROOT/runs/npu-smoke-a1" --nproc 4 --micro-batch 12
```

查看 `console.log`、`train_log.jsonl`、20 步导出和来源计数。修复后用新 attempt，保留失败日志。验证 HF 导出能在原 GPU evaluator 加载、产生合法动作。然后用同一父模型重新开始 600 步吞吐试跑：

```bash
python scripts/launch_stage.py benchmark --root "$ECR_ROOT" \
  --run "$ECR_ROOT/runs/npu-benchmark-a1" --nproc 4 --micro-batch 12
```

两类受限运行仍保持 **10k 的 LR horizon**，不会把 cosine 曲线压缩到 20/600 步。前 100 步预热，后 500 步单独计时；NPU 各 rank 同步与最慢 rank 窗口由现场适配代码记录。`process-timing.json` 是包含初始化/导出的完整进程耗时，不能冒充稳定计算窗口。

通过后生成 `$ECR_ROOT/checks/port-acceptance.json`，字段为 `numerics_passed`、`smoke_passed`、`hf_export_load_passed`（均为真实验收结论）、`trainer_sha256`、`data_config_sha256`、`nproc`、`micro_batch`、`attention_backend`，并附各项证据路径。不得仅为解除启动检查而填写 true。任何实质代码、布局或 attention 更改后重验相应项目。

```bash
python scripts/launch_stage.py train --root "$ECR_ROOT" \
  --run "$ECR_ROOT/runs/r2-continue-s1-ascend-a1" --nproc 4 --micro-batch 12
```

正式运行重新从 20k 父模型初始化 optimizer 和 RNG，不继承试跑权重。保存本阶段 4k / 7k / 10k；启动参数固定 seed=1、global batch=192、AdamW、LR=2e−5、warmup=500、cosine 到 5e−6、weight decay=0.1。原输出策略保留导出但不提供精确断点续跑；中断后记录 attempt，不将 HF 权重加载冒充恢复 optimizer/RNG。

## 5 评测和对比

按 [EVALUATION.md](EVALUATION.md) 将三个 HF exports 放到现有 GPU RoboCasa 测评环境。Ascend 训练集群无需搭建另一套仿真器。没有可用 GPU evaluator 时保留完整导出，记录“训练完成、评测等待”，不要承诺已经可以在 NPU 上直接运行原 CUDA/EGL 仿真评测。

先分别跑 dev200，最大成功数选点、并列选较晚步；再对选中模型跑一次 test500。然后执行：

```bash
python scripts/compare_test500.py /path/to/new/test/episodes.jsonl \
  --output /path/to/new/test/gpu-vs-ascend.json
python scripts/analyze_training_time.py \
  "$ECR_ROOT/runs/r2-continue-s1-ascend-a1/train_log.jsonl"
```

`compare_test500.py` 检查 500 个唯一 task/seed、任务分母和 horizon，再给出总成绩、NAV、非 NAV 及逐任务差值。它不会将名义 seed 配对自动当作物理初始状态相同。

GPU 历史日志的 10k 区间和为 **42,296.26 s（11.75 h）**，稳定 step 平均 **4.2217 s**；这个口径不含初始化/最终导出。与新 NPU `process-timing` 比较时必须先统一边界；设备型号、数量和设备小时一起报告。8 NPU 对 4 GPU 不能直接称为同资源加速。

最终交回运行目录、适配 diff、下载与完整性回执、选点证据、逐局评测、timing 和 `comparison.md`。不要只交一个百分比。
