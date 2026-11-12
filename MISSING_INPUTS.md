# Continue 复现尚缺的上传输入

> **公开指南与私有完整包。** 完整 v3 ZIP 已上传 ModelScope；固定版本、下载命令和权限说明见 [DOWNLOADS.md](DOWNLOADS.md)。当前接手阻塞项及解除条件见 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)。GitHub 可公开阅读，完整包与训练数据仍需各自仓库的读取权限。

**2026-11-12 复核：完整 ZIP 已在云端，但以下两项仍没有可交给接手 agent 的已核验下载映射。** 本次检查了冻结包、最新公开交接资料和本机相关上传/归档记录；未发现能关闭这两项缺口的新回执，不表示已穷举其他仓库。

这份数据覆盖交付单以 2026-11-10 13:41 多伦多上传完成报告为基线。它不否认其他仓库可能存在这些文件；如果另有新报告，先按完整路径与 SHA 对账，避免重传。

| 输入 | 原始源目录 | Ascend 恢复目录 | 验收要求 |
|---|---|---|---|
| R2 recovery 全目录 | g16 `$LEGACY_WORK/embodiedclaw-robocasa/datasets/ecr-recovery-unified-9task/v3-round2-20261101/` | `$ECR_ROOT/datasets/ecr-recovery-unified-9task/v3-round2-20261101/` | 固定 manifest 内 24,245 文件、5,557,378,683 B；13 个 LeRobot roots，3,445 episodes / 673,550 frames |
| 20k 父模型完整 HF 导出 | g10 `$LEGACY_WORK/runs/rc365_xr1_20261006/sft/ecr_a3_yixin_s1_cont_s101/hf/step_10000/`（评测侧可能映射为 `$LEGACY_WORK/…`） | `$ECR_ROOT/models/xr1-a3-s1-cont-20k/` | 全部 22 文件逐项匹配 `reference/model-pins-base-parent.json` 的 `continue` 条目 |

R2 manifest 的 SHA256：`aeb4d7271114522d7dc0e2c85ac3f1b26a93ff8b3d40baa014f380bbae112676`。这是 manifest 本身的哈希；实际数据以它列出的逐文件大小和哈希验收。R2 顶层 `meta.json` SHA：`0c036a91ed278408325576d9f2ab8c96d0d89f248c9e3672b2246411b1579cb0`；README SHA：`56997b2a4570a67c24e9d9e616fb4d67bb43629b58dd0c81d5ddd4eac1b1fa62`。

上传方应交回只读用户可访问的仓库链接和逐文件 JSONL，记录：`repo_id`、`repo_type`、`revision`、`remote_path`、`local_path`、`sha256`、`bytes`。`local_path` 必须是上表对应的 ECR_ROOT 下相对路径。实际权重即使存在 dataset repo，`repo_type` 也填 `dataset`。目录名、进度条、只有 config/log 的目录均不能证明权重就绪。

优先使用确切 revision；若服务只能提供分支读取，可使用 `revision="master"`、`revision_mode="sha256_locked_branch"`，前提是每个文件已由冻结源清单锁定 SHA256 和字节数，下载时全部复核。分支读取与不可变 revision 在报告中分开标明。哈希不一致必须报错，不能接受替代文件。

**不要从最终 455/500 模型再训练 10k。** 它累计已经 30k，会变成新一轮实验。不要替换成 A3 s2 446/500；那是不同父分支。也不要在输入不齐时把 Fresh 10k 的结果命名为 Continue。

当前 agent 可以自行完成环境检查、已知输入下载、原版算子诊断和适配；在这两项输入核验通过前不能将正式 Continue 训练标为 ready。本文件仅说明所需交付范围，不自动对旧服务器执行复制或云上传。

权限、数值参考、NPU 环境和 GPU evaluator 等其他接手条件，统一见 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)。ZIP 下载成功不会自动补齐本页两项输入。
