# ECR · Xiaomi XR‑1 最佳模型在 Ascend 上的训练复现与对比指南

> **公开指南与私有完整包。** 完整 v3 ZIP 已上传 ModelScope；固定版本、下载命令和权限说明见 [DOWNLOADS.md](DOWNLOADS.md)。当前接手阻塞项及解除条件见 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)。GitHub 可公开阅读，完整包与训练数据仍需各自仓库的读取权限。

版本：v3（原包不变）· 云端访问与接手审计更新：2026-11-12 · 交接对象：Ascend 集群 Codex / 训练工程师 / 实验分析人员

**执行入口：[START_HERE.md](START_HERE.md)**。按 [DOWNLOADS.md](DOWNLOADS.md) 下载并校验完整私有云端交接包，按入口依次下载、校验、适配、冒烟、训练和评测。最新上传范围见 [UPLOAD_REPORT.md](UPLOAD_REPORT.md)。

**本次复现目标是第二轮 Continue s1：test500 为 455/500（91.0%），导航 46/50，非导航 409/450。最直接的复现方式是加载其固定的第一轮续训父模型，再用 Human:R1:R2 = 1:0.5:0.5 的配方训练 10,000 步。** 父模型已经完成 20,000 步 SFT，因此该模型累计 SFT 为 30,000 步。

配套私有交接包已固定原训练源码、数据目录、模型文件校验值、开发集选点结果、test500 逐局结果和 GPU 训练计时证据。**Ascend 适配尚未实现或实机验证**；v3 提供可生成的 NPU 适配候选、差异文件和分阶段启动器；候选尚需在目标设备上完成数值/算子/导出验收。R2 recovery 和固定 20k 父模型的最新交付状态、下载清单及验收结果见 [DOWNLOADS.md](DOWNLOADS.md) 和 [MISSING_INPUTS.md](MISSING_INPUTS.md)。包内 `cloud-inputs.pending.json` 保留 v3 冻结时的状态；补传入口以最新公开说明为准。

## 1. 建议执行顺序与待补信息

1. 确认 Ascend 型号、卡数、每卡 HBM、CPU 架构及平台提供的 CANN 镜像；记录容器 digest、驱动和 Python 依赖。
2. 下载 Human 10 个数据集、第一轮 recovery 18 个目录、第二轮 recovery 13 个目录，以及 **A3 s1 续训 20k 父模型**，逐文件验 SHA256。
3. 从本包固定源码做 NPU 适配，完成相同 batch 的数值检查、多卡 20 步训练、HF 导出和原评测端加载检查。
4. 运行独立吞吐试验；然后重新加载固定父模型、重置优化器和随机数，正式训练 10k，保存 4k / 7k / 10k。
5. 三个候选各做 dev200，按预定规则选点，只对选中 checkpoint 做 test500。使用原 GPU RoboCasa 评测环境比较效果，并单列训练、导出、评测耗时。

| 尚待补齐的项目 | 对执行的影响 |
|---|---|
| Ascend 型号 / 卡数 / 每卡显存 / CANN 镜像 | 决定适配栈、micro-batch 和并行布局；目前不能给可信 NPU ETA |
| 训练输入和读取权限 | 按 [DOWNLOADS.md](DOWNLOADS.md) 的固定清单恢复 Human、R1、R2 和 20k 父模型，保存本机完整验收回执 |
| NPU 适配 commit 和冒烟报告 | 原脚本含 CUDA、NCCL、FlashAttention 2 路径，需要修改并验收 |

这些是执行输入，并不改变已经确定的模型、数据和训练配方。配置清单见 [cloud-inputs.pending.json（私有配套文件）](PRIVATE_INPUTS.md)。

## 2. 固定模型链：加载哪一个 checkpoint

```text
Xiaomi-Robotics-1-RoboCasa365 原版
  └─ A3 s1：Human:R1 = 75:25，10k，训练 seed 1
       └─ A3 s1 续训：同配方再 10k，训练 seed 101，累计 20k
            └─ Round 2 Continue s1：Human:R1:R2 = 50:25:25
                 再 10k，训练 seed 1，累计 30k → 455/500

原版 → A3 s2：10k，446/500，是平行分支，不是上述模型的父模型。
```

| 角色 | 已固定的原始路径 | 用途 |
|---|---|---|
| Xiaomi 原版 | `$LEGACY_WORK/models/rc365_xr1_20261006/Xiaomi-Robotics-1-RoboCasa365` | 从原版重走全部 SFT 阶段，或 Fresh 对照 |
| 主链第一版 A3 s1 | `$LEGACY_WORK/runs/rc365_xr1_20261006/sft/ecr_a3_yixin_s1/hf/step_10000` | 累计 10k；438/500 |
| **本次推荐训练起点：A3 s1 续训** | `$LEGACY_WORK/runs/rc365_xr1_20261006/sft/ecr_a3_yixin_s1_cont_s101/hf/step_10000` | **累计 20k；438/500** |
| **待复现的最佳结果** | `$LEGACY_WORK/runs/rc365_xr1_20261006/sft/ecr_round2_continue_20261101_s1/hf/step_10000` | 累计 30k；455/500 |

路径是历史节点上的源地址，不保证 Ascend 机器存在相同挂载。下载后建议将父模型恢复到 `$ECR_ROOT/models/xr1-a3-s1-cont-20k/`，以 22 个文件的 SHA256 和字节数确认身份，不能只比目录名或 `config.json`。

- 父模型和原版逐文件 pins：[model-pins-base-parent.json（私有配套文件）](PRIVATE_INPUTS.md)。
- 最佳结果及其他冻结模型 pins：[model-file-pins.json（私有配套文件）](PRIVATE_INPUTS.md)、[round2-default.json（私有配套文件）](PRIVATE_INPUTS.md)。
- 实际继承证据：[lineage.json（私有配套文件）](PRIVATE_INPUTS.md)。

**两种复现范围要分开报告。**

| 范围 | 训练方法 | 应如何表述 |
|---|---|---|
| 推荐的硬件迁移对照 | 同一份已冻结 20k 父模型 → Ascend 上重训最后 10k | “第二轮 SFT 在 Ascend 上复现”；前 20k 已在 GPU 上完成 |
| 全部 SFT 都在 Ascend 上完成 | 原版 → A3 10k → A3 续训 10k → R2 10k；每阶段重置 optimizer / scheduler | “三阶段共 30k SFT 在 Ascend 上完成”；仍使用固定的历史 recovery 数据 |

全链复现时，前两阶段使用 `data_ecr_a3_yixin.json` 的任务内 Human:R1=75:25，NAV 只用 Human。不能把三个阶段都替换成 R2 的 50:25:25。第一阶段源码 `3cf90b429d704a3d2e6609775f6dd1fc799e54e8`，续训源码 `e1d0ad63faeae74f156b20eb6b2292546b623a5b`；本包在 `reference/a3-first/` 和 `reference/a3-continue/` 保存了关键原文件。

从原版直接按 R2 配方训练 10k 是 **Fresh** 对照：既有 seed 1 为 436/500（87.2%），不是最佳 Continue 模型的同起点复现。直接加载 455/500 的最终权重继续训练，则是新一轮实验。

## 3. 数据下载、目录恢复和校验

### 3.1 必须取得的输入

| 数据组 | 目录数 | 内容与抽样用途 |
|---|---:|---|
| Human | 10 | 3,176,983 帧；覆盖全部 10 个任务，含 NAV |
| 第一轮 recovery | 18 | 9 个任务 × 两个来源模型，**π0.5 与 XR‑1 均须保留** |
| 第二轮 recovery | 13 | 9 个任务，3,445 个 recovery segments、673,550 帧；同一任务可分为多个子技能目录 |
| 起始模型 | 1 套 / 22 文件 | A3 s1 续训 20k 的完整 HF 导出，含 processor、tokenizer、代码和权重分片 |

**训练输入是已整理的 LeRobot recovery 数据，不是直接读取 3,000 局 rollout 或只下载失败视频。** 第二轮数据在 g16 上的源目录为：

```text
$LEGACY_WORK/embodiedclaw-robocasa/datasets/ecr-recovery-unified-9task/v3-round2-20261101/
```

Ascend 上建议保持以下逻辑结构；41 个目录的完整名称见 [DATASET_PATHS.md](DATASET_PATHS.md)：

```text
ECR_ROOT/
├── models/
│   ├── xr1-a3-s1-cont-20k/          # 推荐起点，完整 22 文件
│   └── xiaomi-original/             # 全链复现或 Fresh 对照才需要
├── datasets/
│   ├── rc365_xr1_20261006/
│   │   └── <human-name>/<version>/payload/lerobot/
│   └── ecr-recovery-unified-9task/
│       ├── v1-20261020/<18 exports>/lerobot/
│       └── v3-round2-20261101/<13 exports>/lerobot/
├── configs/
└── runs/                            # 每次实验使用新目录

每个 lerobot/：
├── meta/                            # info、episodes、tasks 等原始元数据
├── data/chunk-*/episode_*.parquet
└── videos/chunk-*/<camera-key>/episode_*.mp4
```

原格式为 LeRobot v2.1、20 fps、三视角 256×256。相机名为 `robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand`。保留 `info.json` 中的路径模板、episode index 和所有分片，不能将同名 `episode_000000.*` 扁平合并。

### 3.2 准备下载环境

以下命令从本交接包根目录执行。下载工具放在独立虚拟环境，避免改变训练依赖。

```bash
set -euo pipefail
export ECR_ROOT=/data/ecr-ascend
python3.12 -m venv .venv-download
.venv-download/bin/python -m pip install 'modelscope-hub==0.6.1'

# 私有仓库使用有读取权限的 ModelScope token；终端输入不回显。
read -rsp 'ModelScope read token: ' MODELSCOPE_API_TOKEN
export MODELSCOPE_API_TOKEN
```

ModelScope 官方 SDK 支持按 revision 下载单文件或仓库，并支持 `MODELSCOPE_API_TOKEN`。本包下载器只按显式文件清单取文件、再次核验 SHA256；不会遍历整个 ECR 封存库。v3 支持 gzip 清单和有界并发；旧数据仓库可显式使用分支读取加冻结内容哈希。命令接口依据 [ModelScope Hub v0.6.1](https://github.com/modelscope/modelscope_hub/blob/v0.6.1/README.md) 和 [固定版本 API](https://github.com/modelscope/modelscope_hub/blob/v0.6.1/src/modelscope_hub/api.py)。

### 3.3 Human：已有确切下载 pins

10 个私有仓库的 repo、revision、archive SHA、payload manifest SHA 已保存于 [human10.json](downloads/human10.json)。例如 TurnOffStove：

```text
repo:     dvdnh2026/ds-robocasa365-target-turnoffstove-human-51f148772b84
revision: 35aec5baa9032b2febfa884714980d7f82543074
```

```bash
.venv-download/bin/python scripts/download_files.py \
  manifests/human10-download.jsonl --root "$ECR_ROOT" --check-only
.venv-download/bin/python scripts/download_files.py \
  manifests/human10-download.jsonl --root "$ECR_ROOT"
.venv-download/bin/python scripts/unpack_human.py --root "$ECR_ROOT"
```

上述流程下载每个仓库的 `lerobot.tar` 和 `SHA256SUMS`，检查固定哈希，解压到 `payload/`，逐文件核验内容和文件集合，最后核对 episode / frame 数。已有同路径文件必须校验一致，不自动覆盖不同内容。下载过程中还会保留 SDK 缓存和压缩包，磁盘预算需计入这部分空间。

### 3.4 R1、R2 和 20k 父模型：按固定清单下载

最新可执行下载命令统一放在 [DOWNLOADS.md](DOWNLOADS.md)，避免把旧归档完成报告当成本次输入交付证明。

| 输入 | 下载入口 | 内容验收 |
|---|---|---|
| R1 recovery | DOWNLOADS 第 3 节；也可用完整 v3 的 `manifests/round1-download.jsonl.gz` | 205,003 文件、49,789,441,549 bytes；原包 `scripts/verify_recovery.py --round round1` |
| R2 recovery 与固定 20k 父模型 | [两项补传交付单](MISSING_INPUTS.md)，使用其链接的冻结下载清单 | R2 原始 manifest 的全部 24,245 数据文件和顶层控制文件；父模型全部 22 文件 |

R1 原始 manifest SHA256 为 `ad34dd82983b3f22036218b7129ab9d2d3ab5d3243018a86ef4d513f0683a8b5`；R2 为 `aeb4d7271114522d7dc0e2c85ac3f1b26a93ff8b3d40baa014f380bbae112676`。它们是清单本身的哈希，每个数据文件仍须匹配清单中的大小和 SHA256。

补传映射放在 `$ECR_ROOT/input-maps/`，不修改原 ZIP 内文件。父模型恢复到 `$ECR_ROOT/models/xr1-a3-s1-cont-20k/`，与原包 `reference/model-pins-base-parent.json` 的 `continue` 条目一致。公开 `downloads/verify_round2_parent.py` 可独立验收这两项，不依赖 Human/R1；正式训练仍要求全部输入通过检查。

下载器成功仅证明清单所列文件已通过校验。还须完成冻结文件集合对账，不能使用少了视频、parquet 或权重分片的清单。恢复时保留完整逻辑路径，不按 basename 合并；使用新的验收输出路径，保留已有回执。

如需查看历史 370,075 条归档映射，按 DOWNLOADS 第 3 节下载五份 metadata。历史 metadata revision `d530e9b77c290307643c517aaf09dab27e1b2480` 只属于归档元数据，不能填到 R1 数据仓库。R1 仍使用显式 `sha256_locked_branch` 从 `master` 读取并逐文件锁定内容。R2/父模型补传单独记录实际 revision，不从旧归档地址推测。

原版模型仅用于原版起点的路线或算子诊断，下载入口见 DOWNLOADS 第 4 节；正式 Continue 仍从固定 20k 父模型开始。

### 3.5 生成机器本地配置

数据全部验收后，生成本地绝对路径配置并校验父模型：

```bash
.venv-download/bin/python scripts/prepare_config.py --root "$ECR_ROOT" \
  --out "$ECR_ROOT/configs/round2-502525.json"
```

该工具只替换原配置的 `$WORK` 前缀，检查 41 个数据根目录的必要组件及 22 个父模型文件；不会改变采样比例。随后在训练环境运行固定源码中的 `round2_input_check.py`，检查实际 parquet / 三路视频 / action / state。文件校验和内容检查都通过后才开始训练。

## 4. 必须保持一致的训练配方

原第二轮源码固定于项目训练仓库的 commit **`45e7476b4d0ead4fa545da1d3fa390a0bfbdbaf7`**，子目录 `training/rc365_xr1_20261006/`；本包提供完整子目录快照 [source-round2（私有配套文件）](PRIVATE_INPUTS.md)。XR‑1 上游辅助代码固定为 **`0dd7aef8dc87296246aae812a1f59ccb708e5546`**，原路径 `$LEGACY_WORK/code/rc365_xr1_20261006/Xiaomi-Robotics-1`，需按 START_HERE 第 3 节获取这一公开上游。私有项目训练子目录已经包含在 ZIP 中，无需另行等待源码交付。不要用随时间变化的主分支代替。

| 参数 | 第二轮 Continue s1 实际设置 |
|---|---|
| 起点 | A3 s1 续训的 `step_10000`，累计 20k；**新建 optimizer / scheduler** |
| 更新次数 | 10,000 optimizer steps；不是 10,000 microsteps，也不是固定若干 epochs |
| 全局 batch | 192 observation samples / update |
| 原 GPU 布局 | 4 卡 × 每卡 micro-batch 12 × 梯度累积 4 |
| 训练随机种子 | 1；重复实验另用 seed 2，保持同一个父模型 |
| Optimizer | AdamW；β₁=0.9，β₂=0.95，ε=1e−8；原实现 `foreach=True` |
| Weight decay | 0.1；仅作用于 requires-grad、ndim≥2 且名称不含 `norm` / `adaln` 的参数，其他参数 decay=0 |
| LR | 500 步 warmup，起始 5e−7、峰值 2e−5；随后 cosine 降至 5e−6；保持原 `lr_at` 的步号语义 |
| Gradient clipping | 全局 L2 norm，阈值 1.0 |
| 精度 | FP32 master weights，BF16 compute，FP32 reduction；HF 导出 BF16 |
| 训练参数 | 冻结语言 embedding / tied LM head，其余按原脚本全量 SFT；无 LoRA |
| 并行与显存 | FSDP2；按原 vision / text / DiT block 分片；activation checkpoint 范围保持原实现 |
| DataLoader | 每 rank 8 workers，原 shuffle buffer 256 |
| Flow-matching | 每 observation 重复 4 次噪声 / 时间抽样；不把这计成 768 个不同 observation |
| Loss | 0.5 × masked MSE + 1.0 × temporal FFT loss；二值 action 维度 6 / 11 不进入 FFT loss |
| HF 导出点 | 本阶段 4,000 / 7,000 / 10,000 步 |

`weight_decay=0.1` 是优化器系数；实际衰减为 `1 − learning_rate × 0.1`，不是每步把参数减去 10%。原源文件的说明文字写过 “fused AdamW”，执行代码实际为 `foreach=True`，复现以执行代码为准。

### 4.1 采样比例的准确含义

对于有 recovery 的 **9 个任务**，按任务内 Human:R1:R2=50:25:25 抽样。任务概率由 Human 帧数分布决定，NAV 始终只用 Human。每个来源池内按 episode 长度加权选 episode，每次访问抽 16 个有放回的时间锚点，再进入 shuffle buffer。原实现见 `rc365_data.py` 与 `round2_sampler.py`。

这不是“全部任务总体严格 50:25:25”，也不是“先均匀选 10 个任务”。原 Continue s1 的 1,920,000 个实际 observation 中，Human/R1/R2 约为 **51.12% / 24.47% / 24.41%**，NAV 的 Human-only 规则会使总体 Human 比例略高。检查每个任务和每个来源的实际计数，缺源、跳过损坏样本或 NAV 混入 recovery 均须报错。

两轮 recovery 都用 `[recovery]`，Human 用 `[human]`。`recovery_full_chunk=false`：沿用原 A3 的尾部 action padding，不能改成 tailfix；也不能启用 `global_uniform_frames`。同任务多个 R2 子目录是该轮来源池的组成部分，不应按目录数平均分配权重。

### 4.2 观测和动作不能换成通用处理流程

- 三相机历史帧索引按时间排列为 `[max(0,t−6), max(0,t−4), max(0,t−2), t]`，history=4、间隔=2；中心裁剪 ratio=0.95，沿用固定 processor / chat template。
- 原始 `observation.state` 为 16D，经 XR‑1 的 `observation_to_state` 转为 14D，再 padding 至模型 60D。不要重新加 q01/q99 标准化。
- 原始 LeRobot action 为 12D，重排索引为 `[5,6,7,8,9,10,11,0,1,2,3,4]`，未来 chunk=16；padding 至 60D 后只对有效维度计算相应 loss。
- `robot_type=robocasa365`；使用既有 HF `MiBoTForActionGeneration` 路径。模型原始代码、处理器、action mask 和 flow objective 一起固定，不能只保留同样的学习率后换成另一个通用 SFT trainer。

## 5. Ascend 适配范围与开训验收

### 5.1 环境建议

先使用云平台为具体 Ascend 型号提供的匹配镜像，再固定包版本。原 GPU 环境为 PyTorch 2.8.0、Transformers 4.57.1；为减少框架差异，可优先评估 **PyTorch 2.8.0 + torch-npu 2.8.0.post4 + CANN 9.0.0 + Python 3.12**。这是官方版本表中的一组组合，不是已验证可运行 XR‑1 的承诺，驱动、固件、算子和芯片仍需匹配。若平台只提供其他组合，应记录变更并重新做数值验收。[Ascend 官方版本兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)

保留训练侧数据依赖：`transformers==4.57.1`、`av==15.1.0`、`pandas==2.3.3`、`pyarrow==21.0.0`、`imageio[ffmpeg]==2.37.0`。其余依赖从原环境 freeze 和固定 XR‑1 源码核对，最终交付实际 `pip freeze`，不能把这里的短列表当作完整 lockfile。aarch64 上视频解码等 wheel 是否可用也需要验证。

### 5.2 必做代码适配

| 原路径 / 行为 | Ascend 需要完成的工作 |
|---|---|
| `cuda` device、`torch.cuda.*`、NCCL | 显式 NPU device / `torch_npu` 初始化、HCCL；同步与显存统计改为对应 NPU API |
| `attn_implementation="flash_attention_2"` | 选用该环境支持的 attention 路径并验证 causal mask、padding mask、历史 token、KV cache；同时检查自定义模型内部调用，不能只改一个字符串 |
| FSDP2 / DTensor | 验证 NPU device mesh、block sharding、FP32 reduction、梯度累积、全局 norm clipping、state dict 聚合和 HF 导出 |
| AdamW `foreach=True` | 核验目标栈对 DTensor 的支持；若需更换算子实现，验证参数组和更新公式、记录差异 |
| `torch.fft.rfft` 频域 loss | 验证复数算子及反向梯度；不得为跑通而删除该 loss；CPU fallback 必须显式记录并计入耗时 |
| Activation checkpoint / FSDP hooks | 保留原 text-MLP / vision-block 边界，以及 `0.0 * logits.sum()` 依赖；不要随意包裹会改变 KV cache 行为的整层 |
| `env.sh` / `round2_job.sh` | 原有 g10 主机门槛、Slurm 路径和 CUDA 参数；Ascend 应新建调度入口 |

官方 TorchNPU 文档列有 FSDP2 / DTensor 支持，但不等于本模型的每个组合已通过验证。[Ascend PyTorch 官方仓库](https://github.com/Ascend/pytorch)

先完成可比版本，再单列性能优化版本。若直接改变 action chunk、历史长度、loss、数据混合、全局 batch 或优化器，该结果应作为新配方报告。

### 5.3 验收顺序

1. **输入**：全部数据文件和父权重哈希通过；41 个根目录可读、三视角可解码；state、action、tag、padding 和来源标记正确。
2. **数值**：同一父权重、固定同一份预处理 batch、noise 和 time，比较 GPU / NPU 的 forward loss、MSE / FFT 分项、梯度及一次参数更新。BF16 跨后端不要求逐位相同；容差先由同后端重复与 FP32 参考测定并记录。
3. **多卡**：20 个完整 optimizer steps，无 NaN/Inf、无静默 CPU fallback、无数据跳过；输出来源计数。20 步仅用于验收，不用来估算正式收敛或宣称性能优势。
4. **导出**：导出标准 HF checkpoint，在原 GPU 评测端完整加载 processor / 权重，产生有限且形状正确的动作。
5. **计时试跑**：固定布局，先预热 100 步，再测至少 300–500 步；检查数据供给和 HCCL 是否成为瓶颈。正式训练重新从父模型开始，不继承试跑的更新。

### 5.4 正式启动模板

以下模板要求训练工程师已将原 `train_xr1.py` 及依赖适配到 NPU，并保持原 CLI 语义。v3 的 `prepare_npu_candidate.py` 会生成未验收候选，`launch_stage.py` 提供 smoke/benchmark/train 入口；现场验收后使用。 `ECR_TRAINER` 应指向已验收的适配入口，`XR1_CODE` 应指向固定上游源码。

```bash
set -euo pipefail
export ECR_ROOT=/data/ecr-ascend
export XR1_CODE=/data/code/Xiaomi-Robotics-1
export ECR_TRAINER=/data/code/ecr-ascend/scripts/train_xr1.py
export ECR_RUN="$ECR_ROOT/runs/r2-continue-s1-ascend-v3"
export ECR_NPROC=4
export ECR_MICRO_BATCH=12

# 在平台已经配置好 CANN / torch_npu 的训练环境中执行。
# 设备绑定与资源分配由 Ascend 平台的作业系统设置。
test ! -e "$ECR_RUN"
mkdir -p "$ECR_RUN"
cp "$ECR_ROOT/configs/round2-502525.json" "$ECR_RUN/data_config.json"

python -m torch.distributed.run --standalone --nproc_per_node="$ECR_NPROC" \
  "$ECR_TRAINER" \
  --xr1-code "$XR1_CODE" \
  --base-model "$ECR_ROOT/models/xr1-a3-s1-cont-20k" \
  --data-config "$ECR_RUN/data_config.json" --out "$ECR_RUN" \
  --steps 10000 --global-batch 192 --micro-batch "$ECR_MICRO_BATCH" \
  --lr 2e-5 --min-lr 5e-6 --warmup 500 \
  --weight-decay 0.1 --clip 1.0 --repeats 4 --workers 8 --seed 1 \
  --save-steps 4000,7000,10000 --retain-outputs --resume-every 100000
```

| 设备数 | micro-batch / device | 累积步数 | global batch |
|---:|---:|---:|---:|
| 4 | 12 | 4 | 192 |
| 8 | 6 | 4 | 192 |
| 8 | 12 | 2 | 192 |
| 4 | 6 | 8 | 192 |

原 trainer 由 `global_batch / (world_size × micro_batch)` 推导累积步数，要求整除。先匹配原 4×12×4；显存不够时调低 micro-batch 并增加累积，记录变化。改变 rank / worker 数会改变数据与随机数流，即使全局 batch 相同也不能宣称逐样本一致。原 GPU 记录的峰值 allocated memory 约 52.4 GiB 仅供参考，不能直接当作 NPU 容量保证。

原正式任务使用 `--retain-outputs`，保留 HF exports，但关闭了周期性完整 resume state；发生中断不能仅凭 HF 导出声称精确续跑。若 Ascend 需要断点续训，另行实现并测试 optimizer、scheduler、RNG、sampler / DataLoader 状态的完整恢复，保留每次 attempt 和计时；保存策略变化也需计入耗时比较。

## 6. 如何评测并对比效果

### 6.1 固定开发集选点，再测 test500

| 项目 | 固定协议 |
|---|---|
| 开发集 | 10 tasks × 20 episodes = 200；base seed `100007` |
| 候选 checkpoint | 本阶段 4k / 7k / 10k，各自完整跑 dev200 |
| 选点规则 | 最大 dev successes；并列选较晚步数；不根据 test500 再换 checkpoint |
| 测试集 | 10 tasks × 50 episodes = 500；base seed `7` |
| Prompt / history / actions | `[human]`；history=4、interval=2、crop=0.95；action chunk / replan=16；5 个 Euler flow steps |
| 仿真环境 | 原 GPU RoboCasa 评测器；RoboCasa `456174f62b89b8fca99eaaf33949c29fec9cfc2a`，robosuite `5ce6643` |
| 场景生成 | 原生 seeded reset，保留原任务索引、horizon 和错误处理协议 |

**test seed 不是训练 seed。** 改训练 seed 1→2 时，dev / test 清单仍保持相同。单局 seed 按 `base_seed + target50_index × N + episode_index` 生成；`target50_index` 是冻结的 50-task 原始索引，不是当前 10-task 列表的位置。完整索引和 horizon 见 [tasks_final.json（私有配套文件）](PRIVATE_INPUTS.md)，原 test500 每局 task / seed 见 [gpu-test500-episodes.jsonl（私有配套文件）](PRIVATE_INPUTS.md)。

不要为了跨机器复现而改用历史保存 XML 的 `reset_to` 路径：项目此前发现过该路径改变物理体质量的问题，它不属于当前 benchmark。相同 task / seed 只能先视为名义配对；严格配对分析还需核验实际初始状态。

建议将 Ascend 导出的候选 HF 模型交给现有 GPU 评测器。这样变化主要来自训练后端。若同时将模型推理、RoboCasa 渲染和物理环境迁往另一个平台，要另做评测端一致性实验，不能将其差异全部归因于训练。

原入口为 `scripts/sft_eval.sh` / `scripts/run_eval.sh`，内部调用和队列逻辑保存在源码快照中。现有执行方需将新模型路径与独立输出目录接入相同流程，保留 strict one-pass 检查；本指南不将包含 g10 环境限制的脚本当作 Ascend 通用启动器。

### 6.2 既有结果与新实验报告表

原 Continue s1 的 dev 为 **173/200、175/200、175/200**，依照并列选晚规则选 10k，而不是依据 test 成绩挑选。证据见 [selection.json（私有配套文件）](PRIVATE_INPUTS.md)。

| 实验 | 训练 seed | Dev 4k / 7k / 10k | 选点 | Test 全部 | NAV | 非 NAV |
|---|---:|---|---:|---:|---:|---:|
| GPU 原始 Continue s1 | 1 | 173 / 175 / 175 | 10k | 455/500（91.0%） | 46/50（92.0%） | 409/450（90.9%） |
| Ascend Continue s1 | 1 | 待测 | 待选 | 待测 | 待测 | 待测 |
| Ascend Continue s2（建议重复） | 2 | 待测 | 待选 | 待测 | 待测 | 待测 |

原 GPU Continue s2 同父模型、训练 10k，dev 选 7k，test 为 442/500（88.4%）。因此 91.0% 是当前固定 best 的单次结果，并非“任意 seed 必须达到”的验收保证。若比较平台平均效果，应固定相同训练 seeds，在两平台报告所有重复结果，不能只选每个平台最高分。

至少报告：10 任务逐项成功数、总成功率、NAV、非 NAV、技术错误数、训练 seed、选点、模型 SHA。技术错误与策略失败分开记录，不能静默减少分母。少量百分点差异需结合逐局结果与置信区间解释；有实际初始状态配对证据时可做 paired bootstrap / McNemar，否则明确是名义 seed 配对。事先约定等效性界限，不能“差距不显著”就宣称完全等价。

解析汇总时不要使用历史 `main10` 字段名推断任务范围；以 `all_selected`、明确的 `NavigateKitchen` 和原始逐局列表计算非 NAV。

### 6.3 需要保存的评测产物

每个 dev / test 输出保留 `episodes.jsonl`、summary、selection、任务队列 / seed / horizon、推理配置、模型 pins、环境版本、运行日志和视频。与 GPU 基线做耗时比较时，录制设置必须相同；历史第二轮的常规评测视频为三视角横拼、stride 4、10 fps。该评测产物并不等同于完整 FD / FR 状态封存。

如果此次评测数据还要用于 failure detection / recovery，需另外保存逐步 actions、完整可恢复状态、实际初始场景、成功判定、终态和三路原始视频，并独立验证恢复 / 回放能力。将增加的记录开销单独计时，不能以普通评测视频宣称已经具备完整 recovery 数据。

## 7. 如何对比训练时间

### 7.1 已核实的 GPU 基线

基线作业为 g10 / `72788` / 4 GPU / 10,000 updates，原始日志 SHA256：

```text
8050dad30620002e10d4effeb8799bd5ac893a0ed87b48149a682708286b46f8
```

| 指标 | GPU 原始记录 | Ascend |
|---|---:|---:|
| `s_per_step` 日志区间累计 | 42,296.26 s ≈ **11.75 h** | 待测 |
| 稳定阶段平均 step 时间 | **4.2217 s** | 待测 |
| 稳定阶段中位数 / P95 | 4.205 / 4.418 s | 待测 |
| 稳定阶段 observation 吞吐 | **45.48 samples/s** | 待测 |
| 精确端到端训练 wall time | 此处没有同口径精确值 | 待测 |
| 训练设备型号 / 数量 | 型号待硬件回执补核；4 卡 g10 | 待填 |

稳定阶段口径为 step>100，排除 4001 / 7001，因为它们包含前一个 checkpoint 导出的耗时；共 9,898 个 step。11.75 h 是原 rank0 记录的区间总和，**不含模型初始化和最终 10k 导出**，但包含前面两个导出落入下一 step 的时间，也不是纯 kernel time。Monitor 从观察到运行至观察到离队约 11h55m，采样间隔 180 s，因此仅是粗略运维参照，不能替代精确计时。

原日志压缩包和可复算指标分别见 [gpu-train-log.jsonl.gz（私有配套文件）](PRIVATE_INPUTS.md)、[gpu-timing.json（私有配套文件）](PRIVATE_INPUTS.md)。复算命令：

```bash
python scripts/analyze_training_time.py reference/gpu-train-log.jsonl.gz
```

### 7.2 新作业计时方法

必须同时记录两套时间：

1. **性能窗口**：同 global batch、precision、loss、数据、保存策略；预热后测稳定窗口。所有 rank 在窗口开始和结束做设备同步，使用最慢 rank 的窗口时间；记录至少三个窗口的均值、中位数 / P95、吞吐、显存、CPU / IO / HCCL 信息。首轮编译单独记录，不把异步 launch 时间当作计算完成时间。逐 step 的同步若有额外开销，也应保持两后端一致并注明。
2. **端到端实验**：记录进程开始、模型 / 数据 ready、第一步、10k 完成、最后 export 完成、dev 完成和 test 完成的 UTC 时间。下载、排队、环境构建、编译、训练、导出和评测分开报告，同时提供包含导出的训练总 wall time。

定义：

```text
global observations/s = 192 × measured optimizer updates / synchronized elapsed seconds
device-hours           = device count × training wall hours
speedup (matched run)  = GPU wall time / Ascend wall time
training ETA           ≈ remaining updates × measured steady seconds/update
                         + remaining checkpoint exports + final export
experiment ETA         = training ETA + remaining dev + test runtime
```

8 NPU 对 4 GPU 可报告整体时间与 device-hours，但不能只称为同资源 speedup。只有拿到 GPU 型号和相同边界的计时数据后，才给平台性能结论；当前 11.75 h 可用于规划和定位，不能作为精确端到端比值的分母。

最终性能对比表还应包括：设备型号 / HBM / 卡数、CPU / 内存 / 存储、软件和容器版本、world size / micro / accumulation、workers、attention backend、是否编译、是否 fallback、checkpoint 策略。若做了优化，先给“相同配方的移植版”，再给“优化版”，分别说明变更和效果。

## 8. 完成后应交回的文件

```text
run/
├── provenance.json                 # 父模型、源码/适配commit、数据manifest、硬件、容器
├── pip-freeze.txt
├── data_config.json
├── input_check.json
├── numerical_parity.json
├── smoke_acceptance.json
├── train_log.jsonl
├── sampling_summary.json
├── timing.json                     # 明确每段起止时间及计时边界
├── hf/step_4000/, step_7000/, step_10000/
├── model-file-pins.json
├── selection.json
├── eval/dev_step4000/, dev_step7000/, dev_step10000/, test/
└── comparison.md                   # 效果、训练时间、资源、差异与结论
```

本包 SHA256 总清单为 `SHA256SUMS`，本地来源和哈希见 [evidence-index.json（私有配套文件）](PRIVATE_INPUTS.md)。其中历史 `dataset-catalog.json` 的 `preparation_state` 是提交时的 pending 状态；后来成功的 g10 暂存以 `round2-readiness.json` 为准。它证明历史训练输入曾通过验收，不证明本轮 ModelScope 下载或 Ascend 环境已经就绪。

原 v3 封包时的离线工具检查保留于 `verification.json`。2026-11-12 已补齐 R2/20k 父模型的真实云端映射并独立对账，公开下载工具的映射和样本访问测试通过；全量 payload 回下载尚待验收。最新结果见 [补传核验摘要](downloads/round2-parent-verification.json)。尚未在 Ascend 上训练或完成 NPU 算子验收；目标平台和 GPU evaluator 条件仍见 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)。v3 补入完整 R1 下载清单、集群执行入口、环境探测、NPU 候选生成、分阶段启动器、数据验收和 test500 对比工具；v1/v2 原包保留。
