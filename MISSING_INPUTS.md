# R2 recovery 与固定 20k 父模型：已完成补传和验收

**2026-11-12：两项云端交付缺口已关闭。** 本次现场核验了源文件，完成私有补传，再从云端向空目录下载全部 **24,270 文件、15,683,968,143 bytes**，与冻结清单和父模型 pins 全部一致。源文件保留，并在上传后重新核验。

| 输入 | 已交付内容 | 恢复目录（相对 ECR_ROOT） |
|---|---|---|
| R2 recovery | 24,245 数据文件 / 5,557,378,683 bytes，另含 3 个顶层控制文件；13 个 LeRobot roots、3,445 segments / 673,550 frames | `datasets/ecr-recovery-unified-9task/v3-round2-20261101/` |
| 固定 A3 s1 累计 20k 父模型 | 全部 22 文件 / 10,122,587,755 bytes，包括 3 个权重分片、tokenizer、processor 与自定义代码 | `models/xr1-a3-s1-cont-20k/` |

## 实际下载入口

- [私有 ModelScope 仓库](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-xr1-ascend-r2-parent-20261112)
- [完整下载映射（固定版本）](https://modelscope.cn/api/v1/datasets/dvdnh2026/ds-ecr-xr1-ascend-r2-parent-20261112/repo?Revision=203223f41d3de33efdc1c0587de28075ddac963a&FilePath=manifests%2Fround2-parent-download.jsonl.gz)：浏览器需要登录有权限的账号；agent 推荐使用下面的公开下载入口。
- 映射仓库内路径：`manifests/round2-parent-download.jsonl.gz`
- 映射 revision：`203223f41d3de33efdc1c0587de28075ddac963a`
- 映射大小：`1193185` bytes；SHA256：`fcce42bbf84728ca0b3545360e33be50353a24368e60ed8c8903d7bcb12d6b91`
- [机器可读入口](downloads/round2-parent-metadata-download.jsonl)：先取得并核验私有 gzip 映射，内含全部文件的实际 repo、固定 revision、远端路径、本地恢复路径、大小和 SHA256。
- **可复制命令：[DOWNLOADS.md 第 6 节](DOWNLOADS.md#6-r2-和固定-20k-父模型补传下载与独立验收)。** 无需访问旧服务器或自行猜测下载地址。

映射单独上传并固定版本，其中的数据记录引用已完成的 payload 提交，避免自引用。每个 payload revision 都是 40 位固定提交；这次补传不使用浮动 `master`。

## 冻结身份与验收

R2 原始 `MANIFEST.sha256`：`aeb4d7271114522d7dc0e2c85ac3f1b26a93ff8b3d40baa014f380bbae112676`；顶层 `meta.json`：`0c036a91ed278408325576d9f2ab8c96d0d89f248c9e3672b2246411b1579cb0`；顶层 README：`56997b2a4570a67c24e9d9e616fb4d67bb43629b58dd0c81d5ddd4eac1b1fa62`。父模型逐项匹配完整 v3 包 `reference/model-pins-base-parent.json` 的 `continue` 条目。

[验收摘要](downloads/round2-parent-verification.json) 区分现场源校验、云端固定版本核对、全量真实回下载，以及公开下载工具的独立实测。全量回下载使用 ModelScope Hub API 0.4.5 的等价调用，再执行本仓库相同哈希的 `verify_round2_parent.py`；公开 0.6.1 下载器另行实测完整映射、7 个数据/模型文件及 13 个仓库入口。不能把这些结果写成已在 Ascend 集群完成验收。

原 v3 ZIP 的字节与包内 SHA256SUMS 保持不变；其中 `cloud-inputs.pending.json` 和旧文档保留封包时的历史状态。最新补传映射放在 `$ECR_ROOT/input-maps/`，以本页和 DOWNLOADS 为当前交付依据。

**父模型仍是累计 20k 的 A3 s1 续训模型。** 最终 30k best、A3 s2、原版 Fresh 都不能替代它。仓库是私有的，接手 agent 需要有效读取凭据，并在自己的机器执行完整下载和校验。其他条件见 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)：输入文件已交付，NPU 环境、数值与运行验收、GPU evaluator 仍需就绪。
