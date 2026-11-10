# Continue 复现尚缺的上传输入

> **Public guide with download entry points / 公开指南与下载入口。** Known ModelScope addresses, fixed download pins and generic download tools are available in [DOWNLOADS.md](DOWNLOADS.md). Dataset repositories remain private. Frozen training source, NPU execution tools and GPU comparison evidence remain in the separately transferred v3 handoff ZIP; see [PRIVATE_INPUTS.md](PRIVATE_INPUTS.md).

这份交付单针对已核对的 2026-11-10 13:41 多伦多上传完成报告。它不否认其他仓库可能存在这些文件；如果另有新报告，先按完整路径与 SHA 对账，避免重传。

| 输入 | 原始源目录 | Ascend 恢复目录 | 验收要求 |
|---|---|---|---|
| R2 recovery 全目录 | g16 `$LEGACY_WORK/embodiedclaw-robocasa/datasets/ecr-recovery-unified-9task/v3-round2-20261101/` | `$ECR_ROOT/datasets/ecr-recovery-unified-9task/v3-round2-20261101/` | 固定 manifest 内 24,245 文件、5,557,378,683 B；13 个 LeRobot roots，3,445 episodes / 673,550 frames |
| 20k 父模型完整 HF 导出 | g10 `$LEGACY_WORK/runs/rc365_xr1_20261006/sft/ecr_a3_yixin_s1_cont_s101/hf/step_10000/`（评测侧可能映射为 `$LEGACY_WORK/…`） | `$ECR_ROOT/models/xr1-a3-s1-cont-20k/` | 全部 22 文件逐项匹配 `reference/model-pins-base-parent.json` 的 `continue` 条目 |

R2 manifest 的 SHA256：`aeb4d7271114522d7dc0e2c85ac3f1b26a93ff8b3d40baa014f380bbae112676`。这是 manifest 本身的哈希；实际数据以它列出的逐文件大小和哈希验收。R2 顶层 `meta.json` SHA：`0c036a91ed278408325576d9f2ab8c96d0d89f248c9e3672b2246411b1579cb0`；README SHA：`56997b2a4570a67c24e9d9e616fb4d67bb43629b58dd0c81d5ddd4eac1b1fa62`。

上传方应交回只读用户可访问的仓库链接和逐文件 JSONL，记录：`repo_id`、`repo_type`、`revision`、`remote_path`、`local_path`、`sha256`、`bytes`。`local_path` 必须是上表对应的 ECR_ROOT 下相对路径。实际权重即使存在 dataset repo，`repo_type` 也填 `dataset`。目录名、进度条、只有 config/log 的目录均不能证明权重就绪。

优先使用确切 revision；若服务只能提供分支读取，可使用 `revision="master"`、`revision_mode="sha256_locked_branch"`，前提是每个文件已由冻结源清单锁定 SHA256 和字节数，下载时全部复核。分支读取与不可变 revision 在报告中分开标明。哈希不一致必须报错，不能接受替代文件。

**不要从最终 455/500 模型再训练 10k。** 它累计已经 30k，会变成新一轮实验。不要替换成 A3 s2 446/500；那是不同父分支。也不要在输入不齐时把 Fresh 10k 的结果命名为 Continue。

当前 agent 可以自行完成环境检查、已知输入下载、原版算子诊断和适配；在这两项输入核验通过前不能将正式 Continue 训练标为 ready。本文件仅说明所需交付范围，不自动对旧服务器执行复制或云上传。
