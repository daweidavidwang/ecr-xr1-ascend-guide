# WAM 评测协议与当前 GPU 对照

返回 [WAM 总入口](../WAM_HANDOFF.md)。所有当前结果均为 NVIDIA 线冻结快照，尚无 Ascend 实测。

## 1. 两条协议严格分开

**官方报告复现**用各自发布代码与原生 simulator，`split=pretrain`；**ECR Target50**用共同 ECR simulator、`split=target`、相同冻结场景请求。两条线的50个任务名相同，不意味着场景、环境实现、初态或成绩可直接混合。

[任务清单](pins/target50-tasks.json) 包含全部50任务及 horizon，18 atomic_seen + 16 composite_seen + 16 composite_unseen。[官方协议](pins/official-pretrain-protocol.json) 包含100个模型/任务条目的完整 horizon、trial数与seed起点；不要自己从任务字母顺序重造seed。

### LoopWAM 官方 pretrain

- 原策略 commit `3400f1c1c1f7243cb9fbc3456ae05092964bc6cd`，发布 checkpoint、dataset_stats 不变。
- 每任务50局，预测64动作、每次重规划执行48动作，quality_score=5。
- 原 `policy_seed=null`，并发调用顺序会影响随机采样；不能为了复现固定策略seed后仍称“原发布协议”。固定噪声用于单独的数值诊断。
- README 的每个分号子组独立重置任务局部index：constructor=`42+1000*subgroup_local_index`，reset=`constructor+trial`；直接使用清单中的明确值。
- 每任务创建一个原生环境，连续reset50局；success、terminated、truncated或horizon停止。
- 原脚本对 bundled registry 的 horizon 乘1.5；本交接清单已经是最终值，**不要再乘一次**。

### Awomo 官方 pretrain

- 原策略 commit `1145c18afd13a3876c6740f098bd734f5e35617a`；RoboCasa `4f8a2980def75a55dff96b990745b83540425f09`。
- 每任务30局；constructor seed7；reset=`7+50*index_within_native_registry_group+trial`。
- 策略seed=`episode_reset_seed*10000+replan_index`，每次执行16动作；压缩输入开启。
- 客户端每4控制步送观测；9个history槽，每40控制步留一个历史。
- 每局由原生 `run_episode` 创建和关闭环境；只按success/horizon停止，原生逻辑忽略terminated/truncated。不能改成另一模型的终止规则。

官方参考为 Awomo64.4%、LoopWAM49.5%。目前分母未完成，不能宣布复现一致或不一致。Loop 官方逐任务原始成功数尚无已核验对照，当前只可报告自己的任务结果。

## 2. ECR Target50 共同协议

共同环境固定 RoboCasa `456174f62b89b8fca99eaaf33949c29fec9cfc2a`、robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`；完整包版本和资产验收边界见 [GPU环境](GPU_ENVIRONMENTS.md)。源码、软件和资产均需验收，不能单凭一个commit判断场景等价。

在 [实现细节](IMPLEMENTATION_DETAILS.md) 中运行内嵌生成器，可从本目录任务清单逐字节重建下列文件：

| 文件 | 数量 | SHA256 |
|---|---:|---|
| `dev.jsonl` | 50×20=1,000 | `4f4af1f0a1baa628b4f10ef9d906459d2bd955c4cbb573778be90fc6c4237cee` |
| `test.jsonl` | 50×50=2,500 | `c312526053e6042b0d9963a91ebd5dea7c9cec12d21b20f16daba7c86ab53d3e` |

两个模型分别执行同一test清单，合计5,000局。场景ID、请求seed和顺序一致；各自的输入预处理、动作映射、预测长度、重规划和策略采样保留原语义。**不把 XR1 推理配方拷进 WAM。**

请求seed不同只说明请求不同，不证明物理初态不同或与Human数据无重叠。实际要记录/比较：原始model XML、layout/style、episode metadata、instruction、MuJoCo integration state、state14/state16、三相机图像、实际reset seed。禁止用会改变mass/inertia的XML重放代替原生reset后声称物理等价。

已有共同两个dev场景：`target50-v1/dev/04/000` NavigateKitchen seed510004000，`target50-v1/dev/06/000` OpenDrawer seed510006000。完整horizon分别450和750。共同环境四进程重建48步、104个成对数组等证据已验；它们是开发诊断，不能计作正式test。

Awomo在共同target的策略RNG与大的environment seed分离：`(7+50*target_index+slot)*10000+replan`，避免超uint32；环境仍用冻结的大seed。此规则在共同target合同中单独声明，不能与官方pretrain seed公式混写。

共同原生策略短诊断每模型两场景、每场景最多64步；完整入口必须使用完整horizon，其进程、身份、观测与错误验收规格直接列于 [实现细节](IMPLEMENTATION_DETAILS.md)。快照时两者尚未完成真实GPU策略验收，剩余48类实际重建/覆盖也未完成。清单和CPU测试通过不放行全部5,000正式局；先完成同场景/动作包装、长历史和完整dev前置。

## 3. 已知故障与处理边界

1. **B300 CUDA12.8/NVRTC 与架构兼容问题。** 原Awomo实际模型失败后，通过诊断确认compute目标相关，换到隔离的cu129环境后完整推理通过。不是模型权重错误，也不能照搬cu129到Ascend。
2. **EGL用户态库与实际分配设备不一致。** 当前GPU运行使用隔离的库路径，核对实际GPU UUID/PCI与EGL映射，不修改共享系统栈。NPU不承接这个OpenGL后端；推荐保留GPU evaluator。
3. **原生材质shininess越界/GL错误。** 曾导致正式任务中断；直接clamp材质会改变像素，因此没有用“修改材质后成绩”冒充原版。已做精确错误归类、保存动作重放和环境关闭/清理诊断。新Ascend复测仍需记录未知GL错误，不能全局吞掉GL异常。
4. **Loop原CLI的Hydra导入方式。** 旧启动方式失败发生在server启动/模型加载之前；后来通过直接以原脚本 `__main__` 启动解决。接手端从固定公开 Loop checkout 的 `eval_robocasa.sh` 查明 `serve` / `client` 对应入口，在各自环境中以独立进程执行原脚本，保留其工作目录、Hydra 配置搜索路径和完整参数。不要把原 CLI 当普通模块 import 后调用，以免改变配置发现行为。NPU 设备适配仍须另行实现并验收。
5. **训练checkpoint“变化见证”曾有误判。** 抽样/strided参数观察不能代替完整张量比较；保存/恢复需要全参数、optimizer与数据游标证据。
6. **长任务的外层超时。** PackDessert、StoreLeftovers 旧CPU计划超出worker允许值，失败不是策略失败。接手端分别设定模型加载、单次请求、单局和整批运行的时间预算；整批预算必须覆盖全部固定局数、加载与关闭时间，外层进程上限还须包含清理余量。具体秒数由目标硬件开发阶段实测后冻结，不能因超时缩短协议 horizon 或删去未完成局。超时保留 attempt、已完成记录与错误，再用新 run 继续处理。

### 3.1 独立启动与验收合同

接手端按下面的合同实现自己的启动入口；全部依据来自本指南和固定公开上游，不要求取得原运行端的包装器、请求文件或调度回执。

| 阶段 | 必须读取或保存的内容 | 通过条件 |
|---|---|---|
| 固定运行身份 | [公开源码 pins](pins/source-pins.json)、本机适配 diff/commit、模型与输入清单、实际环境版本、独立 run ID | 原版与适配版本均可定位；逐文件输入校验通过；记录实际生效配置 |
| 固定评测计划 | 官方线读取本页第 1 节及 [100 条模型/任务协议](pins/official-pretrain-protocol.json)；共同 ECR 线按[实现细节](IMPLEMENTATION_DETAILS.md)第 6 节生成请求 | task、split、trial/slot、constructor/reset/policy seed、horizon 和计划分母全部在运行前固定；两条协议分别记录 |
| 启动策略与模拟器 | policy 与 evaluator 各自的解释器、工作目录、完整 argv、网络地址、分配设备、加载/请求/单局/整批预算 | 模型实际加载后完成开发输入的真实请求；端口可连接或进程存活本身不算推理通过；保留原生传输与预处理 |
| 核对运行输入 | 初态 XML、完整 metadata、integration state、state14/state16、三相机、实际 seed；逐次请求、返回动作块和执行步 | 共同 ECR 必须先通过第 2 节及实现细节第 7–8 节的匹配要求；官方线保留各自原生环境生命周期与 seed 规则 |
| 结束与回交 | 逐局 success/停止原因/error、所有自有子进程退出码、环境关闭记录、文件大小与 SHA256 索引 | 每个固定身份至多一份有效完成结果；异常、超时和缺文件单列；调度终态未知时标明未知，不补造退出码 |

历史验收摘要用于说明已有结论，不能当作本机启动凭据。新分支的 CPU、推理、闭环、训练恢复和完整评测分别按[Ascend 执行手册](ASCEND_RUNBOOK.md)生成自己的证据；某一级缺项就记录该级未完成。正式运行前后保存源码 diff 与环境锁，防止运行期间改动被漏记。

## 4. 结果统计与重复运行

- 每条记录至少含 model/run/port commit、split、task、scene_id、trial、请求/实际seed、模型/数据/config SHA、实际steps、success、termination reason、error、action/observation证据索引。
- 错误不是策略失败，也不是成功。报告 success / valid-completed / planned 三个数量和 error 数；未完成时保留原计划分母，不能只对成功返回的局数报“全量成功率”。
- 同一固定身份的多次attempt全部保留，按事先规定选择至多一个有效完成记录；不能根据失败/成功选择性重跑。
- GPU与Ascend使用不同run命名空间。硬件对照的场景必须配对；策略随机性、有限样本误差和任务完成顺序偏差要报告。
- 基础统计按任务成功率及分组汇总报告，说明macro/micro权重，附置信区间或配对差异估计。未完成50任务时不对外宣称最终总体结果。
- dev用于代码/配置/模型选择；test只在冻结后执行。不得用test调attention、噪声、normalizer、训练步数或checkpoint。

## 5. 已完成任务的冻结结果

完整机器可读结果见 [accepted-results-20261113.json](pins/accepted-results-20261113.json)，包含原回执SHA256。以下任务完成顺序不是随机抽样；表中均为原GPU运行。

### Awomo

| 任务 | 本次成功/完成 | 官方成功/完成 |
|---|---:|---:|
| ArrangeBreadBasket | 22/30 | 21/30 |
| ArrangeTea | 14/30 | 18/30 |
| CloseBlenderLid | 12/30 | 12/30 |
| CloseFridge | 29/30 | 28/30 |
| CloseToasterOvenDoor | 30/30 | 29/30 |
| CoffeeSetupMug | 25/30 | 26/30 |
| DeliverStraw | 17/30 | 12/30 |
| NavigateKitchen | 28/30 | 23/30 |
| OpenDrawer | 25/30 | 26/30 |
| OpenStandMixerHead | 28/30 | 28/30 |
| OpenCabinet | 29/30 | 26/30 |
| PickPlaceCounterToCabinet | 26/30 | 25/30 |
| PickPlaceCounterToStove | 27/30 | 26/30 |
| PickPlaceDrawerToCounter | 30/30 | 28/30 |
| PickPlaceSinkToCounter | 28/30 | 30/30 |
| PickPlaceToasterToCounter | 25/30 | 25/30 |
| SlideDishwasherRack | 29/30 | 29/30 |

### LoopWAM

| 任务 | 本次成功/完成 | 官方成功/完成 |
|---|---:|---:|
| BreadSelection | 10/50 | 未取得逐任务原始对照 |
| CategorizeCondiments | 4/50 | 未取得逐任务原始对照 |
| CloseBlenderLid | 24/50 | 未取得逐任务原始对照 |
| CloseFridge | 49/50 | 未取得逐任务原始对照 |
| CloseToasterOvenDoor | 41/50 | 未取得逐任务原始对照 |
| CoffeeSetupMug | 23/50 | 未取得逐任务原始对照 |
| CuttingToolSelection | 12/50 | 未取得逐任务原始对照 |


不完整任务单列：Awomo CategorizeCondiments 1/11（计划30）、BreadSelection 10/13（计划30）、CuttingToolSelection 13/29（计划30）；Loop ArrangeBreadBasket 0/4（计划50）、ArrangeTea 5/13（计划50）。这些是有效partial的成功/已完成局数，剩余局因错误/中断未形成完整分片，不能加入完整任务平均后声称全量。
