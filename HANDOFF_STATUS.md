# 接手状态与阻塞项

核对日期：**2026-11-12**。目标仍是固定 A3 s1 累计 20k 父模型，再进行第二轮 Continue 10k；不是从原版开始的 Fresh，也不是从最终 30k best 再训练。

**完整 ZIP 已可下载，但正式 Continue 训练尚未就绪。** 首先按 [DOWNLOADS.md](DOWNLOADS.md) 完成访问检查、ZIP 下载和包内校验，再按下表推进。不要将“指南上传完成”写成“全部训练输入已交付”。

## 已解除的交接问题

| 项目 | 现状 |
|---|---|
| 找不到 v3 ZIP | 已上传至现有私有 ModelScope 仓库；有固定 revision、准确路径、下载清单、大小与 SHA256 |
| GitHub 与本地文件链接混淆 | 文档已提供云端文件链接；无需访问 owner 本地路径或等待手工传包 |
| 只有公开文档，缺训练源码和工具 | 完整包内有冻结训练源码及九个交接工具；明确要求解压整个包，公开 `downloads/` 与包内 `scripts/` 分目录执行 |
| 不知道需要哪些仓库权限 | 已提供 12 仓库小文件访问清单，以及 401/403/404 排查说明；权限需在接手机器实测 |

## 仍需关闭的条件

| 编号 / 状态 | 影响阶段与现有证据 | 由谁补齐 / 解除条件 |
|---|---|---|
| A1 接手账号读取权限：待接手方验证 | GitHub 公开不等于 ModelScope 可读；ZIP、R1、十个 Human 仓库权限独立。本次使用 owner 现有凭据，不能据此保证另一账号可读 | 仓库 owner 授权接手账号；接手 agent 在目标机器跑 `downloads/access-check.jsonl` 和 ZIP 下载，保存成功回执。凭据只走 secret 或环境变量 |
| D1 R2 recovery 下载映射：未交付 | 包内 `cloud-inputs.pending.json` 的 repo/revision 仍为 null。需要 13 个 LeRobot roots、3,445 segments / 673,550 frames；冻结 manifest 覆盖 24,245 文件、5,557,378,683 bytes | 数据上传执行方提供实际云端 repo、revision、逐文件路径/大小/SHA256，包含原始 `MANIFEST.sha256`、顶层 meta/README；按 [MISSING_INPUTS.md](MISSING_INPUTS.md) 下载并通过 `verify_recovery.py --round round2` |
| D2 固定 20k 父模型下载映射：未交付 | ZIP 有完整 22 文件身份 pins，无权重本体或下载地址。三个冻结权重分片的哈希在本次检索的本机上传/归档回执中未找到可交付云位置 | 模型上传执行方交付匹配 `reference/model-pins-base-parent.json` 中 `continue` 的全部 22 文件映射，落入 `$ECR_ROOT/models/xr1-a3-s1-cont-20k/`，通过 `prepare_config.py` 的身份校验 |
| E1 Ascend 平台配置：未提供 | 缺目标芯片/卡数/HBM、CPU 架构、已分配资源、驱动/固件、CANN/TorchNPU 镜像及实际依赖锁定；短依赖列表不是已验证环境 | 集群执行方保存 `cluster-preflight.json`、容器 digest、依赖版本、资源分配和磁盘预算。先核实 Human 体积、缓存和 checkpoint 临时副本空间，再做全量下载 |
| E2 NPU 数值与运行验收：待完成 | 提供的是候选生成器，无实机通过报告；固定 batch/noise/time 的 GPU 数值 fixture 尚未作为现成输入随包交付 | GPU/NPU 执行方固定同一父模型和预处理 batch/noise/time，保存 loss 分项、梯度、单步更新及容差依据；再做 20 步多卡、HF 导出和 GPU 加载，生成绑定代码/配置/布局的真实 `port-acceptance.json` |
| V1 GPU evaluator 环境与执行方：待确认 | ZIP 有评测源码、任务索引与基线逐局记录，但无完整 GPU 环境镜像和 RoboCasa 仿真资产，也未指定当前可用评测节点或导出接收路径 | GPU 执行方确认固定 RoboCasa/robosuite commit、资产和环境、调度方式、导出交付位置、加载验证及回传位置；按 [EVALUATION.md](EVALUATION.md) 执行三个 dev200、选点和一次 test500 |

D1/D2 的“未交付”指本交接资料没有**经核验且可执行的下载映射**；本次没有穷举所有账号或其他云仓库，不据此断言世界上没有文件。若上传方已有新回执，先按完整路径和冻结 SHA256 对账，再添加映射，无需重复上传一致内容。

## 接手时可以推进到哪里

1. **立即可做**：读取最新 GitHub 文档；在当前机器验证权限；下载、解压和校验 v3；下载并验收 Human/R1；获取固定公开 XR-1 上游；环境探测、静态审阅和独立算子诊断。
2. **D1/D2 齐全后**：生成 `manifests/round2-parent-download.jsonl`，完整恢复输入，执行 R1/R2 integrity、父模型身份检查和 `round2_input_check.py`。这个 JSONL 目前未随包交付，不能用空表或猜测的地址替代。
3. **完整输入和资源齐全后**：进行数值验收、多卡 smoke 和吞吐试跑。包内 `launch_stage.py` 的 smoke/benchmark/train 均检查 R1、R2 和完整输入回执；缺数据时不能靠启动 smoke 绕过。
4. **移植验收后**：从固定 20k 父模型重新开始正式 10k，不继承 smoke/benchmark 的权重或优化器。
5. **GPU evaluator 可用后**：检查导出可加载，再完成 dev/test；只有训练完成而尚无评测资源时，状态写为“训练完成，评测等待”。

每个外部缺口提交“编号、repo/路径或所需环境、错误类别、缺少谁的交付、成功条件”，同时推进不依赖它的工作。保留失败记录；不修改原训练配方、用其他父模型补位或把验收布尔值手填为通过。

## 本次检查范围

- 已将原始 v3 ZIP 上传私有仓库，并实际重新下载全部 11,563,326 bytes，核验 SHA256 与原文件一致。
- 解压后的 244 项 `SHA256SUMS` 校验全部通过，包内 6 项离线检查全部通过；这些不是 Ascend 实机验收。
- 已检查 ZIP 中的源码/工具/清单目录和冻结待补项；训练数据与父模型权重不在 11 MB 的包中。
- 下载流程使用本仓库的 `download_files.py` 和固定 `modelscope-hub==0.6.1` 在新的独立下载目录验证；最终结果记录见 [公开访问验收摘要](downloads/access-verification.json)。
- 检查本机相关上传/归档回执，以及父模型三个权重分片的固定 SHA256；没有新回执能关闭 D1/D2。只含 rollout、config、模型 pins 或训练日志的记录不能作为数据/权重已交付证据。
- 本次没有重新下载几十 GB 的全量训练数据，没有在 Ascend 上运行训练，也没有执行 GPU 仿真评测。小文件读取成功不代替训练数据完整性或硬件验收。

公开 GitHub 以本页说明最新访问与接手状态；ZIP 内文档和参考源码保持 v3 冻结状态。文档更新不会修改包内 SHA256SUMS。
