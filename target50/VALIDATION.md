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
