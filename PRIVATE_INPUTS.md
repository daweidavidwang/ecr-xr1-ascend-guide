# 完整私有交接包与访问边界

完整 `XR1-Ascend训练复现指南-v3.zip` 已于 **2026-11-12** 上传至现有私有 ModelScope 归档仓库。按 [DOWNLOADS.md 第 5 节](DOWNLOADS.md#5-完整训练交接包已上传按固定版本下载) 获取固定 revision 的文件、读取权限和可复制下载命令。无需等待 owner 手工传送 ZIP，也无需访问旧训练节点。文件不在 GitHub Releases。

ZIP 为 11,563,326 bytes，SHA256 `9e52a2342ad542af952376326bafdecd2fc43704301efbe461922e71fabe1466`。已实际重新下载，大小、ZIP SHA256 与原文件一致；原 v3 未重新打包。

包内提供：

- `scripts/`：下载、解包、完整性检查、配置生成、环境探测、NPU 候选生成、分阶段启动和比较工具。
- `manifests/`：Human 和 R1 的冻结下载映射、公开原版模型 pins，以及明确标为待补的 R2/父模型信息。
- `reference/source-round2/`、`reference/a3-first/`、`reference/a3-continue/`：冻结训练源码和配置；从包内快照开始，无需另取私有训练 Git 仓库。
- `reference/model-pins-base-parent.json`：父模型完整 22 文件的预期身份；pins 是校验信息，不是权重文件。
- GPU 逐局结果、选点、训练计时与相关冻结证据。
- `SHA256SUMS`：包内文件校验表；与本公开仓库的 `PUBLIC_SHA256SUMS` 区别使用。

包内**不提供**训练数据本体、20k 父模型权重、Ascend 驱动/CANN 镜像、已验收的 NPU 移植、GPU 仿真资产或可直接搬用的完整环境镜像。固定公开 XR-1 上游仍按 START_HERE 中的 commit 另行取得。

公开 GitHub 文档不附带私有仓库权限。接手者需获得 ZIP 仓库、R1 仓库及十个 Human 仓库的读取权限；首次访问先运行 [12 仓库小文件探测](DOWNLOADS.md#0-先取得完整包并验证读取权限)。不要把 owner 的 token 贴进文档或共享链接。

完整阻塞项、核验范围与阶段完成条件见 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)。
