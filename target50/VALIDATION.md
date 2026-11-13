# 本次交接发布验证

2026-11-13验证范围：

- 新公开下载器实际下载并核验三个固定HF revision的12个metadata文件；共1,717,383 bytes。不宣称本次在owner工作站重新下载43.25GB全部载荷。
- 使用Python3.12 / PyArrow21.0.0从这些metadata重建三个episode map，全部精确匹配B300既有32,904,734-byte索引集合的SHA，逐任务计数一致。
- 新私有包中11个实际Human训练核心文件逐文件SHA，与已验收B300 s1 step10000的run contract完全一致。
- 原始模型22-file download map的bytes/SHA与Target50 base pins完全一致。
- 50-task目录、18/16/16分组、旧10子组、帧数和scene manifest的各phase分母交叉核验。
- 新私有包的场景生成器重建全部48,500个请求，5个phase文件和manifest与冻结版本精确相同。
- 下载工具的正常下载、已有正确文件、错误哈希保留、路径逃逸和symlink拒绝5项本地检查通过。
- 本次没有运行NPU、模型训练或科学评测；这些验收不能替代目标机器的正式gate。

完整源数据曾在B300按307文件pins核验；本机校验确认文档所给入口和索引可重现。接手者仍须在Ascend本机完成全部文件和实际reader验收。

## 私有源码云交付验收（2026-11-13 22:15 UTC）

- 新ZIP为329,314 bytes，SHA256 `be53dddf20f67763a873d9a9998c6653b32ba6ba17c26db1fde8ba037658e9b7`；104个成员、103项内部SHA全部通过。
- 现有账号 `dvdnh2026` 及目标仓库私有性已实际核验；只向新独立路径上传单一ZIP，保留旧包。
- 固定revision `26401275c1f3cfbaeb621e73f682c93c3a70e9d8`，从空缓存通过网络完整回下载，bytes/外层SHA/103内部SHA全部通过；owner独立复核下载副本及ZIP CRC。
- Human50三个固定HF来源的12个metadata文件再次在新目录实下载，全部1,717,383 bytes及SHA通过；未在本次重新下载43.25GB全部数据。
- 云交付真实字段见[source-delivery.json](source-delivery.json)，下载清单见[source-bundle-download.jsonl](source-bundle-download.jsonl)。此前缺上传凭据状态已解除。
- 已验证现有账号读取，不宣称匿名可下载，也不代表Ascend目标机器已配置凭据或通过NPU验收。

原始XR模型补充核验：22个固定文件的远端存在性、大小和SHA与公开清单一致；匿名完整下载5,580-byte `config.json`并验SHA。未重新下载10.12GB权重。详细范围见[input-access-verification.json](input-access-verification.json)。Monitor云端上传/下载使用`modelscope-hub==0.4.6`，公开下载器固定`0.6.1`；下载器清单及已有文件验收另行核验，不将不同SDK版本混写。
