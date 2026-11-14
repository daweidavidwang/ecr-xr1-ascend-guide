# WAM → Ascend 复测交接：LoopWAM 与 Awomo-0.5I

交接日期：2026-11-13。研究状态冻结于 **2026-11-13 21:22:11 UTC**。本次交接由 WAM owner 根据用户要求编写，交给另一个 agent 在 Ascend 上独立适配和复测。以下是已验收证据快照，不代表接手时的实时调度状态。

**结论：两模型已有 NVIDIA 环境、固定输入和部分评测/训练工程证据；没有 Ascend 实测结果，也没有完成正式 Human SFT 或 recovery SFT。不要将“文件齐备”“CPU 检查通过”“短训练通过”写成完整训练或性能复现成功。**

本仓库原来的 XR1 指南继续有效。本页是独立 WAM 交接，不能套用 XR1 的 20k 父模型、50:25:25 配方、10k 步数或 455/500 成绩。WAM 主线的 Target50 是 **50 个任务类别**，不是 50 个 episode，也不是旧 XR1 的十任务 test500。

**交接文档维护约定：** 本页及 `wam/` 的后续更新只引用可匿名访问的公开 GitHub 上游，不写入私有 GitHub 仓库的名称、链接、分支或内部文件路径。接手所需的配置、算法、代码片段、实验结论和验收条件直接写入本交接文档及其附页；不得让接手者依靠私有仓库补齐说明。公开上游引用须固定版本；未交付的大文件或尚未验证的步骤按实际状态说明。

## 1. 接手阅读顺序

1. 本页：目标、当前事实、代码身份和接手责任。
2. [数据与训练合同](wam/DATA_AND_TRAINING.md)：下载、读取器、动作/历史语义、训练诊断与未完成部分。
3. [评测合同与 GPU 对照](wam/EVALUATION.md)：官方 pretrain 与共同 ECR target 的区别、逐任务结果、渲染和统计规则。
4. [Ascend 执行手册](wam/ASCEND_RUNBOOK.md)：适配点、验收阶段、启动方式和回交材料。
5. [实现细节与可复制代码](wam/IMPLEMENTATION_DETAILS.md)：数据划分、共享视频读取、Awomo 样本/采样、场景请求生成与验收。
6. [公开源码 pins](wam/pins/source-pins.json)、[GPU 完整包版本](wam/GPU_ENVIRONMENTS.md)：固定上游版本与对照环境。所有必要合同写在本目录，不依赖外部工程仓库。

## 2. 目标与先后关系

研究问题：LoopWAM / Awomo-0.5I 在与 ECR 对齐的 RoboCasa Target50 上表现如何；Human SFT 和 failure recovery SFT 能否改善表现。

| 工作线 | 定义 | 计划分母 | 当前已验收 |
|---|---|---:|---:|
| Awomo 官方复现 | 原发布策略与原生评测，`split=pretrain` | 50 任务 × 30 局 = 1,500 | 17 完整任务，510 局 |
| LoopWAM 官方复现 | 原发布策略与原生评测，`split=pretrain` | 50 任务 × 50 局 = 2,500 | 7 完整任务，350 局 |
| Awomo ECR Target50 | 共同 ECR 环境和固定 test 请求，`split=target` | 50 × 50 = 2,500 | 0 正式局 |
| LoopWAM ECR Target50 | 与上一行相同初态合同，各自保留原生策略语义 | 50 × 50 = 2,500 | 0 正式局 |
| Human SFT | 固定发布权重 → Human 微调 | 两模型分别定义配方 | 正式更新 0 |
| Recovery SFT | 经验证的 Human 模型 → 与 ECR 对齐的 recovery | 配方、数据验收尚未完成 | 正式更新 0 |

860/9000 是已完成完整任务的局数进度，**不是成功率**。另有 Awomo 53、LoopWAM 17 个有效 partial episode，单独保留，未纳入上述完整任务计数。部分任务结果不能证明与官方总分一致。官方报告参考值为 Awomo 64.4%、LoopWAM 49.5%；其 split、任务权重和采样方式必须按各自协议比较。

Ascend 第一阶段是工程和硬件对照：固定同一模型、数据、输入、随机噪声、损失定义，再比较运行正确性、峰值内存、吞吐和策略效果。Ascend 适配诊断可先推进；完整研究仍按基线 → Human → recovery 建立可解释的模型链。

## 3. 当前已经做完什么

| 层次 | 已有证据 | 不应扩大的结论 |
|---|---|---|
| NVIDIA 环境 | g11、B300 两节点已有模型/仿真准备及各自验收记录；B300 上两模型真实推理和闭环通过 | 不等于 Ascend、任意驱动或新节点通过 |
| 公共输入 | 模型、基础组件、资产 33 文件，54,932,913,325 bytes；Human 原始输入 307 文件，43,253,612,263 bytes 已有源摘要校验 | 下载成功不等于训练读取正确 |
| Human 数据视图 | 50 任务、25,307 演示、14,957,899 帧；Loop 原生数据读取与训练/验证划分已验 | 不代表 Awomo 全数据运行时读取器已完成 |
| Loop 训练 | GPU2 单卡与两卡原生短训练、参数更新、保存、新进程恢复已验 | 两卡续训存在小量数值差异，**不是逐位一致**；不是正式 SFT |
| Awomo 数据适配 | 150 演示/450 窗口数值检查；3 个完整历史窗口、4 个早期历史窗口像素检查；3 个完整历史原生 processor 样本通过 | 不是全体 25,307 演示的逐像素/前向检查 |
| Awomo 全窗口索引 | 910,723 train、10,720 validation 窗口；双节点产物逐字节一致；CPU 模拟 1/2/8/16 rank 采样恢复通过 | 不是实际分布式训练或模型/优化器恢复 |
| Awomo 训练诊断 | 真实模型、样本、FP32 主参数及 CPU 前置检查通过；GPU 诊断实现已固定 | 真正前向、反向、AdamW 更新与恢复仍未验收 |
| ECR 共同场景 | 冻结 dev/test 请求；两个 dev 场景的共同环境重建/初态与观测检查通过；策略/完整入口 CPU 前置通过 | 全 50 类物理覆盖、真实策略完整 dev、正式 test 尚未完成 |

Awomo 早期历史原生 processor 的新 CPU 请求在冻结快照时已提交、尚未 owner 验收。后续若有新结果，必须读取相应返回和验收记录后追加，不根据队列提交状态推断成功。

## 4. 代码来源与可恢复版本

公开上游：[LoopWAM](https://github.com/loop-wam/LoopWAM)、[Awomo-0.5](https://github.com/Awomo-WestlakeDI/Awomo-0.5)。Awomo 官方成绩说明：[RoboCasa leaderboard](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions_md/Awomo-0.5I_2026_10_09.md)。

本次接手只需要下列公开源码。原版评测和后续训练基底分别固定，不能拿后续训练代码替代原报告策略。

| 用途 | 公开仓库 | 固定 commit |
|---|---|---|
| Loop 官方策略和 bundled simulator | `loop-wam/LoopWAM` | `3400f1c1c1f7243cb9fbc3456ae05092964bc6cd` |
| Awomo 官方策略 | `Awomo-WestlakeDI/Awomo-0.5` | `1145c18afd13a3876c6740f098bd734f5e35617a` |
| Awomo 后续公开 SFT 基底 | 同上，独立 checkout | `32353be48397614206d947b1994b39049e7f4fbe` |
| Awomo 官方 RoboCasa | `robocasa/robocasa` | `4f8a2980def75a55dff96b990745b83540425f09` |
| ECR 共同 RoboCasa | `robocasa/robocasa` | `456174f62b89b8fca99eaaf33949c29fec9cfc2a` |
| Awomo / ECR robosuite | `ARISE-Initiative/robosuite` | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |

只在新目录获取；已有目录先查 HEAD 和未提交修改：

```bash
export WAM_ROOT=/data/wam-ascend
mkdir -p "$WAM_ROOT/code"
git clone https://github.com/loop-wam/LoopWAM.git "$WAM_ROOT/code/loopwam"
git -C "$WAM_ROOT/code/loopwam" checkout --detach 3400f1c1c1f7243cb9fbc3456ae05092964bc6cd
git clone https://github.com/Awomo-WestlakeDI/Awomo-0.5.git "$WAM_ROOT/code/awomo"
git -C "$WAM_ROOT/code/awomo" checkout --detach 1145c18afd13a3876c6740f098bd734f5e35617a
git -C "$WAM_ROOT/code/awomo" worktree add --detach "$WAM_ROOT/code/awomo-sft" 32353be48397614206d947b1994b39049e7f4fbe
```

模拟器按 [环境说明](wam/GPU_ENVIRONMENTS.md) 的公开 pins 单独安装。从固定版本创建自己的适配分支，保存 base commit、适配 diff 和新 commit；不要改写现有 B300 运行源码或不可变请求。

**工程改动的交付方式：** 本交接把 Human v3→v2.1 读取、Awomo RoboCasa 动作/历史/processor/窗口采样、诊断目标、恢复规则和 ECR 评测入口语义直接写在 [实现细节](wam/IMPLEMENTATION_DETAILS.md) 与各合同页。公开上游本身不包含这些新增工程模块；接手者按本文代码/规格实现并验收自己的 Ascend 分支，不能把原 GPU 验收结论自动赋给新实现。这里没有已完成的 NPU 训练启动器。

**文件边界：** 本目录包含文档、可复制的核心辅助代码、公共输入清单、任务协议和结果摘要；大权重、原始 rollout 和 GPU 数值 fixture 未附带。模型/数据可按固定公开源下载，CPU 样本可按本文从原始演示重建。GPU/NPU 逐层数值对照仍需生成并校验真实参考张量；不能把历史样本摘要当作已经交付的 payload。

## 5. 现有落点与迁移方式

| 现有角色 | 路径/标识 | 用法 |
|---|---|---|
| B300 节点 1 | `b300-1` / `gpu1`，根 `/data/ubuntu/wam-robocasa` | 输入与环境已有独立记录；不可默认空闲 |
| B300 节点 2 | `b300-2` / `gpu2`，同名根 | Loop 两卡训练证据来自此节点 |
| g11 | 根 `/mnt/fasterwam-data/david/wam-robocasa` | 已准备环境/输入；不要假设与 B300 同一存储 |
| 资源 owner | 原集群资源调度负责人（Monitor） | 旧节点执行、传输、GPU 分配由其统一协调 |

接手者可以从公共固定源重新下载；已有源节点文件需由当前资源 owner 按精确清单转移并校验。旧的 g10 内网路线、堡垒机和代理配置不代表新 Ascend 节点已可用，不把凭据或节点访问权限写进 GitHub。

本交接不启动 Ascend 作业、不预约卡，也不接管原 WAM 队列。原 WAM owner 继续推进 B300 线；接手者负责 Ascend 分支的适配、独立记录和回交。正式开始前记录实际 Ascend 型号、设备数、每卡 HBM、CPU 架构、驱动、固件、CANN、torch/torch_npu 和平台镜像 digest。

## 6. 下一位 agent 的第一批可执行工作

1. 校验本仓库 `PUBLIC_SHA256SUMS`，获取公开的固定源码；核对公共输入清单，按实现细节重建划分和场景请求，再准备真实数值对照 fixture。
2. 建立两个隔离的 NPU 策略环境，先做最小 NPU 算子、各组件加载和单次动作推理；保持原模型、动作维度、图像处理和采样步骤。
3. 在开发输入上获取匹配的 GPU 数值参考。现有 CPU/像素证据可复用，但尚没有本次 Ascend 跨硬件前反向参考包。
4. 先验证短闭环，再做原生长历史、完整 horizon 与训练/恢复测试。适配顺序和判定标准见 [执行手册](wam/ASCEND_RUNBOOK.md)。
5. 能运行后，分别报告工程正确性、NPU 性能、原版官方复现、共同 Target50 结果。Human/recovery 训练不得用诊断权重接续，也不得以 test 选择配置。

回交到本仓库的独立 WAM 结果目录，至少包含：适配 commit、环境锁、输入 SHA、实际命令、测试日志索引、数值误差、训练/恢复结论、峰值内存/吞吐、逐局身份和成功数/计划分母。大文件保持私有并提供固定版本与 SHA。没有完成的项目保留明确待办和失败原因。
