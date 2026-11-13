# ECR XR‑1 Target50：Ascend 接手与复现实验

版本 **v1，2026-11-13**。这是当前 **50-task** 工作的交接入口。仓库原有 README / START_HERE 描述的是旧 **10-task、R2 Continue 10k、test500** 实验；本文件与 `target50/` 定义新实验，不能混用数据、步数或成绩。

**接手目标：先在 Ascend 从原始 Xiaomi checkpoint 重新训练完整 Target50 Human SFT，训练 seeds 1、2，各 48,000 updates；在 19,200 / 33,600 / 48,000 做 dev1000，按 dev 选点后做 test2500；随后按三轮 recovery 计划推进。** 对比同配方 GPU 与 NPU 的效果和训练时间。Ascend 实机适配尚未完成，不能把本文件当作已经验证的 NPU 一键启动器。

## 1. 当前已经做到哪里

状态证据截至 **2026-11-13 21:04 UTC / 16:04 Toronto**；队列观测截至 **20:38 UTC**。这是冻结交接快照，接手时重新读取实际作业状态。

| 项目 | 已证实状态 | 接手含义 |
|---|---|---|
| Human50 | 50 tasks、25,307 episodes、14,957,899 frames；307 个源文件的完整 SHA 已核验；B300 两份独立本地拷贝 | 直接下载下文固定公开来源；不要找旧 Human10 |
| XR‑1 Human s1 / s2 | 各完成 **10,000 / 48,000**；完整 checkpoint 与后继预检通过 | 是中间训练状态，尚未完成本阶段 |
| 下一段 | B300 `359`（s1）、`383`（s2），各 4 GPU，等待 `358` 诊断终态及容量 | 尚未到首个 19,200 dev 点；没有可信开始时间 |
| 正式 dev / test | **未开始，无 Target50 成绩** | 表格留空；旧 91.0% 不能移到这里 |
| Target50 D1 / D2 / D3 | 尚未生产 | 旧 9-task recovery 不能充当 49-task recovery |
| 轨迹保存与恢复 | 输入/模型 query 重放、部分数值状态、文件分块和 fixture 检查已通过；真实完整 RGB / 隐状态恢复未通过 | 完整采集门槛尚未解除；工程检查不等于评测结果 |
| Ascend | 无本实验实机训练或数值验收 | 接手 agent 负责目标硬件适配、验证与计时 |

当前 GPU 运行根为两台 B300 上的 `/data/ubuntu/embodiedclaw-robocasa`，不是 Ascend 的挂载路径。已训练 10k 的 DCP 路径与 SHA 见 [当前模型与源码交付](target50/SOURCE_AND_STATUS.md)。**Ascend 主对照从原始 base 的 step 0 开始，不需要先下载 B300 中间 DCP。**

## 2. 接手顺序

1. **下载并核验 Human50 和原始 base。** 具体仓库、固定 revision、307 文件清单、22 模型文件 pins、下载命令及目录见 [DATA.md](target50/DATA.md)。数据入口已经确定；不依赖历史私有 Human10。
2. **取得新的私有 Target50 源码补充包。** 身份与交付状态见 [SOURCE_AND_STATUS.md](target50/SOURCE_AND_STATUS.md)。旧 v3 云包包含旧10任务代码，**不包含当前 Target50 reader / sampler / restart 的完整实现**。公共 GitHub 只发布指南与下载元信息。
3. **在 Ascend 做适配和验收。** 按 [TRAINING_AND_ASCEND.md](target50/TRAINING_AND_ASCEND.md) 保持科学配方，记录设备、CANN / torch-npu 版本、attention 实现、并行布局和实际 diff。不要在 Ascend 原样运行 B300 hostname / CUDA 专用脚本。
4. **先完成短跑、恢复、导出和基准计时，再重置到 base 开正式 H50。** 两个训练 seed 独立初始化；吞吐试验输出与正式运行分开。
5. **19.2k / 33.6k / 48k 的 HF 导出提交同一冻结 dev1000。** 三点齐全后选点，执行 test2500；必须沿用 [EVALUATION_AND_ROLLOUT.md](target50/EVALUATION_AND_ROLLOUT.md) 的 seed、horizon、推理配置、完整录制及技术失败处理规则。
6. **后续 recovery 不是现成下载任务。** 按第 4 节先用本轮 dev-selected teacher 采集新 rollout、生产/验收 recovery，再开始下一阶段训练；数据未齐不能把阶段标为 ready。

可立即开始的是下载、输入校验、环境盘点和已取得源码后的适配。正式 NPU 开训的 gate 是输入完整、适配数值检查、多卡完整状态恢复、HF 导出重载都通过。完整录制问题不阻塞 Human SFT 的独立准备，但必须在批量 dev/test/rollout 前解决。

## 3. “50 全场景”的准确范围

这里是 RoboCasa365 **Target50 的 50 个任务**：18 atomic-seen、16 composite-seen、16 composite-unseen；环境 `split=target` 使用 10 个 target 厨房配置。不是 50 个厨房，也不是 pretrain 的 2,500 个厨房。

- 固定 [tasks50.json](target50/tasks50.json) 的任务名、顺序、官方分组、旧10标记及逐任务 horizon；[完整50任务表](target50/TASKS.md) 可直接检查。
- 三个公开 Human 源是 **LeRobot v3.0**。旧10私有封存多为 v2.1，不能把 v3 文件按旧 `episode_*.parquet` 模板读取。
- 25,307 是本版本实际示范数，不写成精确 25,000；14,957,899 是冻结实际帧数。
- `composite_unseen` 是官方组名。我们在这组的 target Human 数据上也做了 SFT，不能把它解释为“本次训练没见过的任务”。

## 4. Human + 三轮 recovery 的模型链

`H50 → rollout1 → D1 → R1-50 / R1-cont → rollout2 → D2 → R2-50 → rollout3 → D3 → R3-50`

上述箭头表示数据依赖；权重父节点按下表，不自动沿箭头继承。

| 阶段（每阶段 seeds 1、2） | 权重初始化 | 任务内 Human:D1:D2:D3 | 本段 updates / warmup | 后续用途 |
|---|---|---|---|---|
| H50 | 原始 Xiaomi base | 1:0:0:0 | 48,000 / 2,400 | Human 对照；dev-selected teacher 生成 D1 |
| R1-50 | **原始 Xiaomi base** | 0.75:0.25:0:0 | 48,000 / 2,400 | 第一轮联合 SFT |
| R1-cont | 对应 seed 的 R1 checkpoint；具体父点尚待冻结 | 同 R1 | 再 48,000 / 2,400 | 同配方续训候选；与 R1 分别报告 |
| R2-50 | R1 与 R1-cont 候选中按 dev 选中的共同 teacher | 0.50:0.25:0.25:0 | 48,000 / 2,400 | 第二轮 SFT；D3 teacher |
| R3-50 | dev-selected R2 共同 teacher | 0.50:1/6:1/6:1/6 | 48,000 / 2,400 | 第三轮 SFT |

原计划尚未固定R1-cont的具体父点；建议使用对应seed的48k段末，以明确连续同配方第二段，执行前在父模型清单中冻结该选择。此处是待落实约定，不是已有checkpoint事实，也不等同R2的dev-selected父模型。所有实际父checkpoint必须列出完整路径与文件SHA。

**R3 配比是新计划，尚未证明最优。** 本轮只把旧10中较好的配方扩展，不承诺“50-task 最佳”。旧 R1 recovery 含 π0.5 与 XR 两种失败来源；新三轮计划使用本轮 XR teacher，须记录这一变化。

采样必须保持：任务权重由冻结 Human 帧数决定 → 任务内选 Human / 各轮 recovery → 来源池内按帧采样。不能替换成全局50:50、任务均匀，或按目录数平均分配来源。NAV 始终 Human-only；其他49任务逐任务记录真实 recovery 覆盖。没有可恢复失败与生产程序缺失是两种状态，不能统一静默回退。

每轮 **50×300=15,000** 个训练侧 rollout 槽，三轮共45,000；每任务前2槽是冒烟，计入15,000。三轮 seeds 不同。recovery 是从失败/接管流程生产的合格片段，不是把全部 rollout 原样作为监督目标；数量由实际失败和验收决定。

主计划为5阶段×2seed=10训练段，按当前S=48k共480k updates；每段批准上限50k。每个阶段3次dev，共30,000 dev局；每阶段/seed选一个模型test，共25,000 test局。额外配比搜索、第三轮因果对照或扩训练预算要单列；不能借硬件迁移改变基准。

## 5. 训练种子、选点与对比

每个阶段两个训练 seed 使用同一数据版本。后续两个 seed 加载同一个已固定 teacher，改变训练随机数；不是两条完全独立的数据生产链。H50两seed均从原始base开始。

每个 seed 的候选本段 step 是19,200、33,600、48,000；只按完整 dev1000 成功数选最高，并列选较晚训练步。R1 teacher 从 R1 / R1-cont 两段候选选取；跨seed再并列时选较晚累计步，再选较小seed。保持每个seed完整报告；**test不能用于挑teacher、挑checkpoint或决定续训**。

GPU / NPU 对比优先使用同一已验证 GPU RoboCasa evaluator，读取各自 HF 导出，固定 dev/test scene 请求与推理参数。不要同时更换训练硬件和模拟器后直接归因于训练硬件。Ascend 本机推理可作为独立系统移植实验，先验证动作与推理数值。

交付50任务总表、18/16/16三组、旧10/新40、NAV/非NAV、每训练seed及两seed均值；无结果留空，不能用0代替未测。时间分列数据准备、环境/适配、预热编译、稳定训练、checkpoint/export、dev、test、排队；同时报设备型号/卡数、step秒数、samples/s和device-hours。

## 6. 给 Ascend agent 的执行指令

> 阅读本仓库 TARGET50_HANDOFF.md 及其配套文档。先核验公开 Human50 的307个文件和三个 episode map，以及原始 XR‑1 的22个模型文件。取得并核验新的私有 Target50 源码包；不要用旧10-task v3训练器替代。建立私有工作目录，固定平台配套CANN/PyTorch/torch-npu组合，实现并记录NPU适配diff，保留48k/2400warmup/global192/原loss及reader语义。做数据、固定batch数值、多卡save/resume、HF导出与评测端加载验收，再从原始base正式训练H50 seed1/2。按19.2k/33.6k/48k做dev1000选点及test2500，保存全部轨迹/三路视频/状态与版本，报告50任务分项、NAV及训练耗时。Target50 recovery尚未交付，后续按本轮teacher生成D1/D2/D3并验收后再训练；不可拿旧9任务数据假装覆盖50任务。记录所有阻塞、变更、实际文件SHA和未测项，不覆盖旧模型或旧结果，不把适配短跑当正式实验。

公开指南不授予私有源代码/数据仓库权限。凭据通过集群 secret 或私有渠道提供；不会写进此仓库。新私有补充包的云端状态如未就绪，按交付文档取得本地包即可继续，无需等待旧10模型或B300中间权重。

机器可读合同：[handoff-contract.json](target50/handoff-contract.json)；空白结果总表：[results50-template.csv](target50/results50-template.csv)；本次入口与索引验证：[VALIDATION.md](target50/VALIDATION.md)。
