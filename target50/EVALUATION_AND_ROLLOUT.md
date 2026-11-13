# Target50 评测、三轮采集与完整数据保存

返回 [交接入口](../TARGET50_HANDOFF.md)。当前没有已完成的Target50正式评测；以下是冻结协议和执行门槛。

## 1. 固定环境与推理

RoboCasa源码固定 `456174f62b89b8fca99eaaf33949c29fec9cfc2a`；XR上游固定 `0dd7aef8dc87296246aae812a1f59ccb708e5546`。逐任务horizon和旧10子组按 [tasks50.json](tasks50.json)，不要用统一horizon覆盖长任务。环境 `split=target`，通过原生reset建立场景。保留MuJoCo / robosuite / RoboCasa、资产版本和EGL/渲染环境的实际lock。

固定推理：`[human]` prompt、history4、interval2、crop0.95、action/replan16、5个Euler flow steps；三相机与训练一致。把训练seed与评测scene seed区分。每个训练seed/模型使用相同dev/test清单，但policy采样RNG、场景生成器RNG必须另存；同scene seed不保证模型输出相同。

最佳跨训练硬件对比是：Ascend训练→标准HF导出→同一已验收GPU policy server和RoboCasa evaluator。Ascend不需要为了训练强行承担原GPU EGL渲染；若无可用GPU评测服务，先交付HF导出与评测工单，明确评测依赖，不能报成功率。若要在Ascend上做policy推理，先单独验收该端动作数值。

## 2. 场景请求和分母

`requested_seed = phase_base + 1000 * target50_index + slot`。

| phase | base | 每任务slots（从0开始） | 总局数 |
|---|---:|---:|---:|
| dev | 510000000 | 20 | 1000 |
| test | 520000000 | 50 | 2500 |
| rollout_r1 | 530000000 | 300 | 15000 |
| rollout_r2 | 540000000 | 300 | 15000 |
| rollout_r3 | 550000000 | 300 | 15000 |

索引/文件SHA见 [scene-requests-manifest.json](scene-requests-manifest.json)。取得新私有源码包后可原样重建：

```bash
python3 "$ECR_TARGET50_SOURCE/training/ecr_target50_20261112/scene_manifest.py" \
  --catalog "$ECR_GUIDE/target50/tasks50.json" \
  --out "$ECR_ROOT/protocol/scene-requests-v1"
```

`ECR_TARGET50_SOURCE` 指私有包的 `source/`。生成后逐文件核验本公开manifest中的bytes/SHA。输出目录必须新建，不覆盖已有协议。所有phase共48,500个唯一requested seeds；worker、分片数、可用卡数不改变清单。4/8/16分片只重新分配同一slot，不重新生成seed。

这些是**场景请求**，不是已经保存的真实初态。运行时记录actual_seed、layout/style、native XML、模型/资产、初态/观测指纹。Human发布者metadata未给全量scene seed/layout/style/初态，不能宣称已证明与训练物理场景零交集；跨模型严格配对必须用实际初态证明，不只比较seed。

## 3. 开发集选点与test

H50同一个训练seed三个候选19,200/33,600/48,000，各跑完整dev1000。按成功数最大选点，平分选较晚step；三点不能混不同training_contract_sha256。缺少候选或未完成dev时，不伪造完整选点结论。

选中checkpoint再做test2500。每个seed只有本轮选中的test结果进入主表。test的NAV为50局，非NAV2450局。旧10子组500局，新40子组2000局；atomic_seen900、composite_seen800、composite_unseen800。最终报告每任务50局成功数及率、分组task-macro、全50、每训练seed与两seed均值。完整均衡分母时overall与task-macro一致；未完整时官方整体成绩留空。

逐attempt记录 `model_id/training_seed/training_step/checkpoint_sha256/training_contract_sha256/scene_id/phase/task/requested_seed/actual_seed/attempt/outcome/artifact_manifest_sha256`。outcome分success、policy_failure、technical_error。策略失败正常计入，不能重试直到成功；技术错误保留原attempt和原因，修复后同slot/seed新attempt，不换容易场景、不增分母。不同attempt不当作额外独立测试局。

私有包 `evaluation_results.py` 提供完整性/汇总与Human选点代码，`evaluation_workplan.py` 提供固定分片计划。它们只校验记录/身份，**不证明磁盘视频、轨迹或恢复能力已经验收**。接入实际runner仍需做制品验收。

## 4. 三轮recovery生产

每轮teacher按前阶段dev决定；不要按test最高挑选。轮内前2局/任务是100局冒烟，属于该轮15,000槽。成功、策略失败和技术错误全部保存。D1来自H50 teacher；D2来自R1/R1-cont teacher；D3来自R2 teacher。

流程为：从失败边界建立case → 开发/复用救援程序及版本 → 留出失败case验证 → 生产 → 语义审核 → LeRobot导出。保留原始HDF5、接管步、完整源rollout关联和未经裁剪原件。旧流程参考门槛是程序试验SR≥70%，每版本抽10条语义审核且≥7/10才导出；新任务需要真实验证，不能假设旧程序覆盖49任务。

逐任务交付采集数、策略失败数、技术错误数、可恢复case数、程序覆盖、合格segment/帧数及剔除原因。NAV只Human；无可恢复失败与未完成程序开发分别记录。新增FD/FR生产不能借用已从本项目排除的旧“1,272局待yixin交付”任务来填完成度。

## 5. 必须直接保存的制品

每个唯一 `phase/model/task/slot/attempt` 新建目录，不覆盖旧attempt：

```text
attempt_000/
  episode.json               # seeds、结果、终态、异常、时长、身份
  scene/                     # native XML、layout/style、资产和初态SHA
  chunks/                    # 分块原始RGB、数值观测/状态、动作
  query_trace/               # 实际模型输入、输出动作计划、query关联/RNG
  events.jsonl               # step前实际Gym动作，step后reward/flags/info
  videos/                    # left/right/wrist三路直接录制
  state/                     # controller/fixture/task/Observable/RNG等
  manifest.json              # 全部文件bytes/SHA及schema版本
  PAYLOAD_COMPLETE.json      # 只在完整写入和验收后创建
  # 异常attempt保存INCOMPLETE及已写内容，不能伪装完整
```

若完成T个动作，保存 **T+1个边界状态/观测**，包括终态。保存原始RGB及三路视频；MP4是有损展示副本，不能作为像素精确重放的唯一依据。实际Gym action字典与XR12维动作都要保存；原始LeRobot动作顺序与Gym顺序不同。

完整FD/FR接管需要：MuJoCo integration state、controller与parts内部状态、fixture/task/Observable缓存、Python/NumPy/设备/环境RNG、policy历史state/image、未消费action plan、成功判定上下文。数组必须存值而非仅hash；不支持的opaque对象显式记gap，不能用None替代后宣布可恢复。保存格式不执行任意pickle作为通用状态协议。

验收包含唯一slot/attempt覆盖、所有文件哈希、数组schema/时序/长度、三路视频解码及终态、跨进程原生恢复/重放、成功/失败视频语义抽查。只证明物理数值相同，不足以证明RGB、controller或任意接管点完整。

当前工程事实：两场景32动作/33状态录制、分块保真和fixture错误边界保存已验；完整真实RGB恢复仍有差异和近空白读回诊断，typed-state代码仅本地测试通过。**不能据此放行全50长程bulk**。接手者应沿这些证据定位问题，不能删相机、放松阈值或只录末帧来宣称问题已解决。

## 6. 封存与分析交付

既定g16新独立封存根规划为 `/work/david/embodiedclaw-robocasa/archive/target50_xr1_v1/`，**本交接不声称该目录已完成封存**。Ascend产物先保留本地原件，按实际获准的私有传输路线登记新批次，逐文件校验后才记独立副本。

按模型/训练seed/phase/任务/slot/attempt组织；README明确parent checkpoint、source/data/recipe/inference SHA、采集轮次、验收范围、未通过项和实际落点。完整manifest、selection.json、逐局attempts.jsonl、50任务总表、训练计时/导出/评测日志一起交付。只含视频或部分状态的批次必须按实际范围标记，不再让后续FD/FR分析者误以为可完整恢复。

可填写的主结果总表见 [results50-template.csv](results50-template.csv)，50任务×5阶段×2seed，所有未测值留空。机器可读范围与状态见 [handoff-contract.json](handoff-contract.json)。本次公开文档工具验证见 [VALIDATION.md](VALIDATION.md)。
