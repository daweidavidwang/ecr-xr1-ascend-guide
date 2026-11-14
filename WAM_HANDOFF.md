# WAM → Ascend 复测交接：LoopWAM 与 Awomo-0.5I

交接日期：2026-11-13。研究状态冻结于 **2026-11-13 21:22:11 UTC**。本次交接由 WAM owner 根据用户要求编写，交给另一个 agent 在 Ascend 上独立适配和复测。以下是已验收证据快照，不代表接手时的实时调度状态。

**增补阅读提示：** 第 1–6 节保留上述冻结快照；第 7 节补充截至 **2026-11-14 07:13 UTC** 已完成并随后验收的 B300 工程结果。涉及“尚未完成”的旧状态以第 7 节明确列出的更新为准，其余不变。

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

**新增实现的直接入口（2026-11-14 核查）：** 接手 agent 按下面的文档内容重建新增模块，无需取得其他 GitHub 仓库权限。

| 需要实现或复核的内容 | 本交接中的具体位置 | 交付形式 |
|---|---|---|
| Human 整 episode 划分、Loop 共享视频读取 | [实现细节](wam/IMPLEMENTATION_DETAILS.md) 第 1 节 | 划分函数、目录与 metadata 规格、逐相机时间戳函数及接入位置 |
| Awomo 动作/状态、图像请求与历史、processor | [实现细节](wam/IMPLEMENTATION_DETAILS.md) 第 2–3 节 | 核心辅助代码、固定样本身份、公开原生函数及逐张量比较要求 |
| 全量窗口、rank 采样与数据游标恢复 | [实现细节](wam/IMPLEMENTATION_DETAILS.md) 第 4 节 | 索引和恢复代码、预期计数、数据与 batch 布局绑定规则 |
| 两模型短训练与模型/优化器恢复 | [数据与训练合同](wam/DATA_AND_TRAINING.md)及实现细节第 5 节 | 配方、损失定义、更新顺序、保存内容与验收标准；NPU 启动器待实现 |
| ECR 场景请求与共同评测入口 | [实现细节](wam/IMPLEMENTATION_DETAILS.md) 第 6–7 节 | 可重建 dev/test 的生成代码、固定 SHA256、初态及策略包装规格 |

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

## 7. 2026-11-14 工程结果增补与 Ascend 接手注意

### 7.1 Awomo 已完成真实短训练与单卡保存恢复

在 `gpu2` 的一张 B300 上，三个 Human train 窗口完成 **3 次逻辑更新**。连续运行更新 1→2→3，再从更新 1 的保存点重放 2→3，因此实际执行 **5 次 AdamW 更新**。三个窗口来自 OpenCabinet、StoreLeftoversInBowl、CategorizeCondiments；具体源 episode、窗口和图像语义见本目录的数据与实现附页。

训练范围为全部 DiT，包括 proprio encoder；Qwen 基座、已发布 LoRA 与 VAE 冻结。171 个可训练张量、3,876,166,656 个元素，第一次更新后 171 个张量全部发生变化。FP32 主参数、BF16 前向 autocast；AdamW `lr=1e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0`，梯度裁剪 1.0，batch size 1。使用 PyTorch math SDPA 和确定性算法，seed 20261113。图像与全部 80 维动作各采独立标准高斯噪声，固定 `sigma=0.5`、timestep scale 1000；损失为图像 velocity MSE 加 **11 个有效动作坐标**的 velocity MSE 均值。有效坐标为 `[7,8,9,10,11,12,16,58,59,63,76]`。这是明确限定的训练诊断，不代表公开原版训练配方已复现。

重放更新 2、3 的输入噪声、loss、梯度范数、171 个参数张量摘要、AdamW 状态与 CPU/CUDA RNG 摘要，与连续运行对应项完全一致。冻结编码器未变化。峰值 allocated memory 为 **102,755,370,496 bytes**，reserved 为 **108,240,306,176 bytes**；这是该单卡诊断的观测值，不能推断 Ascend 内存需求或训练吞吐。

两份保存点仍在源节点私有保留：更新 1 为 46,514,214,051 bytes，SHA256 `8b5ebd73425cfedd52e8fb76fd8dcf188e6194108b42d0baa3b8b3d161742796`；更新 3 为 46,514,214,115 bytes，SHA256 `0bbcab4de25cd740dede383867c7455b5851105f97aa689bb0d4cf97404b34fd`。源节点已复核大小与摘要，**本仓库未包含 checkpoint payload**。它们仅用于诊断，禁止作为正式 SFT 或 benchmark 候选的初始化权重。尚未验收 Awomo 多卡模型/优化器恢复、完整 sampler 训练、早期历史训练或全数据训练。

### 7.2 两模型在 GPU2 的原生官方入口检查已通过

保持已发布策略、原生动作与历史语义，分别完成 NavigateKitchen 单局检查：Awomo 用官方 seed 207，在 227/450 步成功；LoopWAM 用官方 seed 42，执行 450/450 步未成功。两者均完成所属渲染上下文的关闭，实际渲染 GPU 身份匹配分配设备，没有修改 scene/material。已知原生 shininess 警告按既有协议记录，未知 GL 错误仍失败退出。

这两局属于工程检查，不加入正式成绩，不足以比较模型强弱。正式完整任务进度仍为 Awomo 510/1500、LoopWAM 350/2500；共同 Target50 为 0/5000，正式 Human/recovery SFT 为 0。

### 7.3 ECR 初态不稳定的原因与单独版本化候选

在公开 RoboCasa commit `456174f62b89b8fca99eaaf33949c29fec9cfc2a` 的 `robocasa/models/fixtures/counter.py`，`Counter.get_reset_regions` 包含：

```python
valid_geoms = list(set(valid_geoms))
```

这里对 XML Element 去重后不保留原顺序，独立进程中可在相同 RNG 状态下得到不同候选区域次序。实测影响了 OpenStandMixerHead 的放置，也影响 PickPlaceSinkToCounter 的放置失败/重采样路径。因此固定环境 seed 本身不足以保证同一物理场景。

目前测试的唯一改动是稳定去重：

```python
valid_geoms = list(dict.fromkeys(valid_geoms))
```

在独立进程内应用上述方法替换；原安装文件、其他方法、原生 reset、材质和动作语义不变，无 trace 钩子。该改动使用单独的 **`ecr-target50-stable-counter-v2-candidate`** 标识，场景身份为 `(protocol_id, 原 scene_id)`。保留原请求的 seed/task/split/horizon 作为来源，但候选改变种子到物理场景的映射，不能与原 v1 或官方 pretrain 混合计分。

已验收的覆盖范围：

- 两台 B300 节点各运行 50 个任务类别的 dev slot 0，每个输入两个独立 CPU 进程，共 200 次初始化。100/100 个站内重复对和 50/50 个跨节点场景的 XML、metadata、layout/style、integration state、state16/state14 一致。
- GPU2 恢复原生三相机渲染后，验证任务索引 4/6/7/11（NavigateKitchen、OpenDrawer、OpenStandMixerHead、PickPlaceSinkToCounter），各两个独立进程，各执行 12 个固定控制步，共 8 次运行、96 步。每次初态与 CPU 对照匹配；每对的 13 个采样时刻 × 4 类数组（integration/state16/state14/三相机图像）共 52 对数组完全一致，四场景合计 **208/208 对**。

上述通过尚不覆盖全部 50 类真实渲染、完整 dev1000、learned policy 或正式 test。GPU1 的候选渲染验证也仍单独待验。Ascend 接手者应在独立候选分支复现这一差异，保留原版本对照，并分别报告物理初态一致性与跨硬件模型数值误差。原始轨迹数组和完整 CPU 初态 payload 未附在本仓库；应重新生成或按精确文件清单取得并校验，不能用上述统计代替逐数组检查。

所有新增结果仍来自 NVIDIA B300；没有 Ascend 运行结果。当前新增的 GPU2 两模型策略诊断只针对原 v1 的两个已验场景、每场景最多 64 步，尚未作为完成结果，也未自动应用上述候选改动。

**后继失败更新（2026-11-14）：** 两模型在原 v1 的 NavigateKitchen 均完成 64 步，OpenDrawer 则在首次推理前因 metadata 比较失败停止。两者的 XML、状态、初始图像及保存后的 metadata 与参考完全一致；问题定位到原生 tuple 与 JSON 读取后的 list 直接比较。具体原因、可复制修复和严格验收范围已直接写入 [实现细节第 8 节](wam/IMPLEMENTATION_DETAILS.md#8-原生-metadata-与-json-参考的严格比较)。修复后本地及两模型各自的现场 CPU 前置均通过 17 项测试；GPU 复测结果仍待验收。这两次失败保留，不计正式成绩。

**候选全 50 类渲染的下一批工作：** 已准备两节点独立的验证合同，仍使用第 7.3 节同一候选与 dev slot 0。每节点按任务索引 0–9、10–19、20–29、30–39、40–49 分五组，每组 10 类 × 两个独立进程 × 12 步，即 20 次运行、240 步；每组最多一张 GPU、45 分钟。50 个 CPU 初态参考逐项取自两节点各两次已验结果，原四个参考保持不变。新的入口通过 38 项本地测试；每节点须先通过新的 CPU 前置，再运行各组。只有五组无缺失、无重复且完整数组验收通过，才报告该节点 50 类渲染通过；提交任务不算验证完成，仍不覆盖完整 dev/test 或策略成绩。
