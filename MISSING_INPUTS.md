# R2 与固定 20k 父模型：下载入口已补齐，全量回下载待验收

**2026-11-12：全部 24,270 文件、15,683,968,143 bytes 已上传，固定云版本的大小/SHA 已逐批核对。** 完整下载映射已从云端独立下载，并与冻结源文件集合、大小和哈希全部对账。全量 payload 回下载尚未验收通过；此处不宣称两项交付已全部完成。

## 立即可用的下载入口

- [私有 ModelScope 仓库](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-xr1-ascend-r2-parent-20261112)
- [完整 gzip 下载映射：固定版本](https://modelscope.cn/api/v1/datasets/dvdnh2026/ds-ecr-xr1-ascend-r2-parent-20261112/repo?Revision=203223f41d3de33efdc1c0587de28075ddac963a&FilePath=manifests%2Fround2-parent-download.jsonl.gz)，需要有权限的读取凭据。
- 映射 revision：`203223f41d3de33efdc1c0587de28075ddac963a`；仓库内路径：`manifests/round2-parent-download.jsonl.gz`。
- 映射大小：`1193185` bytes；SHA256：`fcce42bbf84728ca0b3545360e33be50353a24368e60ed8c8903d7bcb12d6b91`。
- [机器可读入口](downloads/round2-parent-metadata-download.jsonl)与[当前核验状态](downloads/round2-parent-verification.json)。
- **按 [DOWNLOADS 第 6 节](DOWNLOADS.md#6-r2-和固定-20k-父模型补传下载与独立验收) 下载全部文件并在自己的机器校验。** 无需访问原服务器或自行生成地址。

R2 包含 24,245 个数据文件 / 5,557,378,683 bytes，另有原始 MANIFEST、meta、README 三个控制文件；13 roots、3,445 segments / 673,550 frames 的冻结身份不变。父模型包含全部 22 文件 / 10,122,587,755 bytes，逐项匹配 v3 包 `reference/model-pins-base-parent.json` 的 `continue` pins。

R2 恢复到 `$ECR_ROOT/datasets/ecr-recovery-unified-9task/v3-round2-20261101/`；父模型恢复到 `$ECR_ROOT/models/xr1-a3-s1-cont-20k/`。完整映射放在 `$ECR_ROOT/input-maps/`；原 v3 ZIP 不改动，其中 pending/null 是封包时的历史状态。

现场源文件 SHA 已全量验证。公开 0.6.1 下载工具已实测完整映射、数据/模型样本和 13 个仓库访问入口；全量回下载仍在执行任务中验收，结果会补入本页。目标机器仍需自己的读取凭据和完整性回执。

固定 20k 父模型不能替换成最终 30k best、A3 s2 或原版 Fresh。其他平台与评测条件见 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)。
