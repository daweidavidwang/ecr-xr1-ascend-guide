# WAM 数据、checkpoint 与训练交接

返回 [WAM 总入口](../WAM_HANDOFF.md)。本页描述截至 2026-11-13 21:22 UTC 的已验收输入与候选训练合同；不把候选配方当成官方配方或已完成实验。

## 1. 公共模型和仿真输入

[public-inputs-v1.json](pins/public-inputs-v1.json) 包含精确的 33 个文件、不可变 revision、目标相对路径、字节数和原始内容摘要，总计 **54,932,913,325 bytes**。仅下载主权重不足以推理，必须有对应统计、tokenizer、视觉/语言编码器和 VAE。

| 组件 | 仓库与固定 revision | 用途 |
|---|---|---|
| Loop 主模型、统计、压缩资产 | HF dataset `loop-wam/loopwam` @ `6056976759363e95972bb71cbc5fd06924b160fb` | `robocasa365/step_112500.pt`、`dataset_stats.json`、两份 assets archive |
| Awomo 主模型 | HF model `Auwomo/Awomo-0.5I-RoboCasa365` @ `d02e3560a9e5e649bb0718a97b9e4b923df59594` | 主权重、config、许可证等 |
| Loop 基础组件 | `Wan-AI/Wan2.2-TI2V-5B` @ `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | Wan VAE、UMT5 编码器与 tokenizer；不是重新下载完整 Wan DiT |
| Awomo 视觉语言编码器 | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` | 两个 safetensors 分片及完整 processor/tokenizer |
| Awomo VAE | `black-forest-labs/FLUX.2-klein-base-4B` @ `a3b4f4849157f664bdbc776fd7453c2783562f4d` | `vae/` 配置与权重、许可证 |

两份主权重必须完全匹配：

```text
Loop: 12,041,801,337 bytes
SHA256 ce12fcafeb45177c9821147bd9b848bed20c8abb2507a9b537a941d575f87752
Awomo: 8,281,033,675 bytes
SHA256 e745faf5f75ab0428f00755739374e67d7600724445e1c25973474ef0ca951be
```

### 下载和验收

清单的 `url` 是固定公开源，`destination` 相对于 `$WAM_ROOT`。在获分配的下载节点使用平台已有下载工具取这些文件，不使用可变 `main`。若 Hugging Face 原站失败，可把域名换成 `hf-mirror.com`，**保持 `/resolve/<40位revision>/<path>` 和预期内容摘要不变**。镜像不保证包含全部仓库或 revision；缺失时停止并记录，不退回最新版本。

例如 Loop 主模型可按清单用带断点续传的下载器写入 `.part`，完整校验后再发布到目标路径。大文件先核对磁盘预算，已有文件先验哈希，遇到不同内容保留原文件，使用新的 attempt 目录。不要把 `.part`、错误页或只有 HTTP 200 的响应当作文件通过。

下面是只取该固定主权重的示例；由接手 agent 在获准的下载节点执行，其他文件仍需按完整清单下载：

```bash
mkdir -p "$WAM_ROOT/models/loopwam/robocasa365"
curl --fail --location --continue-at - --max-time 6900 \
  'https://hf-mirror.com/datasets/loop-wam/loopwam/resolve/6056976759363e95972bb71cbc5fd06924b160fb/robocasa365/step_112500.pt' \
  --output "$WAM_ROOT/models/loopwam/robocasa365/step_112500.pt.part"
# 先检查 .part 的精确 bytes 与上方 SHA256；二者通过且最终路径不存在时再发布。
# 不要用这一条成功代替其余 32 个模型/资产文件的验收。
```

本目录提供不联网、不改动输入的验证器：

```bash
python3 wam/verify_inputs.py wam/pins/public-inputs-v1.json --root "$WAM_ROOT"
python3 wam/verify_inputs.py wam/pins/human-hf-full-v1.json --root "$WAM_ROOT"
```

在指南仓库根执行。也可先加 `--component loopwam` 只验一个分组；全量验收仍要运行不带筛选的检查。`git_blob_sha1` 不是文件裸 SHA1：必须算 `SHA1(b"blob " + decimal_size + b"\0" + file_bytes)`；验证器已区分它与 SHA256，并为每个通过的文件输出 SHA256。输出可以重定向到新的验收文件，退出码非零就不能视作通过。

下载器只需逐条读取本目录清单：保持 revision/path，写入独立 `.part`，按 bytes 与指定摘要验证，再原子发布；已存在但不一致的文件保留并报错。不要依赖另一个工程的节点专用入口。下载后统一运行上方公开验证器。

### 容量规划

公共模型/压缩资产 + Human 源文件共 **98,186,525,588 bytes**，约 98.2 GB 十进制；解压资产还约 24.97 GB。实际还需数据视图、下载缓存、两个环境、日志、训练 optimizer/checkpoint，不能按这 98.2 GB 申请总磁盘。原 Awomo 单卡诊断要求 200 GB 空闲、checkpoint 上限 160 GB；Loop 初始每 seed 的 checkpoint 上限 400 GB，均是原执行上限，不是已测 Ascend 占用。

## 2. Human Target50 原始数据

[human-hf-full-v1.json](pins/human-hf-full-v1.json) 固定全部 **307 文件、43,253,612,263 bytes**：

| 数据组 | 公开 HF dataset | revision |
|---|---|---|
| 18 atomic seen | `ember-lab-berkeley/robocasa365-target-atomic` | `ee95cbe19134b1762b2d50fba87f5406d4d59143` |
| 16 composite seen | `ember-lab-berkeley/robocasa365-target-composite-seen` | `d431cfc244ecfa55865ea3bf8681efa0fd2e54b5` |
| 16 composite unseen | `ember-lab-berkeley/robocasa365-target-composite-unseen` | `33f25a67c1a3116477a6c06f20ef22db3c7797ae` |

清单目标名 `datasets/human-hf-sample-v1/` 是历史命名，**当前清单是全量输入**；不要因为目录名 sample 而只取第一批。全量由 307 文件集合、大小、摘要和元数据确认。

验收计数：25,307 episodes / 14,957,899 frames。按冻结的整 episode 划分：

| 分区 | episodes | frames |
|---|---:|---:|
| train | 25,012 | 14,783,844 |
| validation | 295 | 174,055 |

每个任务独立重置 NumPy `default_rng(42)`，按原 `source_episode_index` 排序后做99/1整 episode划分。精确排序、代码、源身份字段和校验计数见 [实现细节](IMPLEMENTATION_DETAILS.md)。不要按帧随机拆分，不将 validation 混入训练；相同 seed 不能代替相同源身份与排序。

源为 LeRobot v3 共享 parquet/视频分片，不能直接当作“一段视频对应一个 episode”的 v2.1 数据。现有适配生成每任务/episode 的数值视图并复用源视频；**每个相机有自己的时间 offset**，必须按 `offset + control_step / 20` 读帧。不能只取首个相机的 offset，不把全分片起点当作 episode 起点，不打乱 parquet 行顺序。

三视角为 `robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand`；元数据 key 加 `observation.images.` 前缀。原视图256×256。[实现细节](IMPLEMENTATION_DETAILS.md) 直接给出转换规格、共享视频时间函数及其在公开 Loop 读取器中的插入位置；参数化本机路径后重新生成数值/解码证据。

## 3. Awomo RoboCasa 数据语义：不能套 RoboDojo 训练

公开的 Awomo SFT 原生例程针对 RoboDojo：14 个有效动作槽、25 Hz、20 history、不同图像尺寸/颜色处理。这不是 RoboCasa 配方。当前 RoboCasa 适配固定如下：

- 控制频率 20 Hz；每次预测并执行 16 个动作；图像按原生拼接/压缩过程形成 RGB，**高320、宽224**（数组末三维为 `[320,224,3]`）。
- 9 个 history 槽，间隔为 **40 个环境控制步**；客户端每 4 步送一次观测，不能把 40 个控制步误作 40 个采样图像。
- Human state16 通过原版 `state_vector` 得 model80；保留原四元数处理和坐标定义。
- Human action12 的顺序为 `base xyz, zero torso, mode, eef xyz, eef rotation3, gripper`。模型输出 80 槽，只监督 11 个有效槽：`[7,8,9,10,11,12,16,58,59,63,76]`。
- 映射：raw[5:11] → model[7:13]；gripper closedness `(raw[11]+1)/2` → model[16]；raw[0:2] → model[58:60]；raw[2] → model[63]；raw[4] → model[76]。其余槽置 0。raw torso[3] 必须为 0；mode/gripper 必须恰为 ±1；不默默 clipping 数据。
- 使用原版 `action12` 解码，要求 Human→model→Human 的 float32 roundtrip 完全一致。模型80、Human12、模拟器执行动作不是可以随意交换的向量。
- 窗口起点在 16 步网格；目标图像在 `start+16`；动作是 `[start,start+16)`。未来图像只能作 target，不能进入条件 prompt/历史。
- 早期缺失 history 由 valid mask 表示。原生 processor 实际条件图像数随有效历史变化：0 history 是 1 图；4 history 是 5 图；9 history 是 10 图。不能为了 batch shape 把未来图像填成历史。

本交接 [实现细节](IMPLEMENTATION_DETAILS.md) 内直接包含数值映射、像素请求重组、样本检查和采样辅助代码。公开原生参照是 `eval_robocasa365/common.py`、`eval_robocasa365/awomo_05i/policy.py` 和 `awomo/awomo_05i/prompt.py`；新增辅助代码需接入自己的分支。

### 已验证的样本范围

- 数值适配：50 任务中选 150 episodes、450 窗口（300 train / 150 validation）；跨两节点 2,250 数组一致，不是全 25,307 演示。
- 完整历史：OpenCabinet episode499/start368、StoreLeftoversInBowl episode502/start384、CategorizeCondiments episode512/start368。真实解码/原生处理及 3 个完整 processor NPZ 样本通过，两节点 51 数组一致；这些是 train 样本。
- 早期历史：OpenCabinet/start0、OpenCabinet/start176、StoreLeftoversInBowl/start0、CategorizeCondiments/start0。0/4 有效 history 的像素与原生序列处理通过，跨两站 36 数组一致。早期 processor 完整样本新请求在快照时尚未验收。
- Qwen/VAE 完整前向、模型反向和训练恢复不属于以上 CPU/processor 验收。

GPU 诊断的三个完整样本 NPZ SHA256：

```text
OpenCabinet: 959e3a5bd3cc84e2d50c55af0633a96904b73c7e48d54d7ea3c841fca03f724f
StoreLeftoversInBowl: c40f869f9ce2b949c57234fd17d92db526b4991eb8234a79f1ab47b4d949e487
CategorizeCondiments: e07ad2288bdb58e6e197dfe149438713cd8ca200cb6bcdf2d3e4c1524738ab99
```

这三份历史 NPZ payload 未附带；其 SHA 是历史文件身份，不是下载入口。可按 [实现细节](IMPLEMENTATION_DETAILS.md) 中明确的 episode/start、固定公开输入和原生 processor 重新生成真实样本，保存数组的 shape/dtype/content hash。NPZ 容器或依赖改变可能影响文件字节；需要历史逐数组比较时，向原运行端取这三份按上述 SHA 标识的文件即可，无需工程仓库访问。不能拿随机数据冒充原样本。

### 全窗口与可恢复采样

已验索引：**910,723 train + 10,720 validation**，dataset SHA256：

```text
a514224e5a149effa98279627b91fa25d3b19f33bff1b3fc9832dd76b2294b49
```

候选规则：episode 内 start=0,16,… 且 `start+16 < episode_length`，丢弃不完整动作段/无未来图像窗口。由此 212,276 train 与 2,535 validation 的尾部 action rows 未监督；没有零完整窗口的 episode。必须报告这个数据覆盖变化，不能称作官方训练完全等价。

训练按完整窗口均匀、不放回 epoch shuffle，NumPy PCG64 / SeedSequence([seed, epoch])；顺序先 global batch、再 rank、再 local batch；不足完整 global batch 的尾部显式丢弃。validation 使用全部完整窗口的确定性顺序。CPU 模拟 world=1/2/8/16、local_batch=1、seed42/epoch0，序列与 cursor 恢复已验。该规则仍为候选，**完整数据运行时读取器、实际 DDP、性能与正式配方选择未完成**。

## 4. LoopWAM Human SFT 当前合同

冻结初始配方完整列在下表；该合同准备完成而正式训练未启动：

| 项目 | 值 |
|---|---|
| 起点 | 发布 `step_112500.pt`，保留原 normalizer/stats |
| seed / 更新 | 42、43，各 1,000 optimizer updates |
| 并行 | 单节点 2 卡；每 rank microbatch1；accumulation1；global batch2 |
| 优化器 | AdamW，lr1e-5，betas(0.9,0.95)，weight_decay0.01，clip1.0 |
| schedule | 原生 cosine，warmup50，最低 lr1e-7，总 horizon1,000 |
| 精度/切分 | BF16、DeepSpeed ZeRO2 |
| 保存/决策 | 每250步保存；固定用最终1,000步作主要决策点；不按 test 选点 |
| 原资源上限 | 每 seed 24h、48 GPUh、400 GB checkpoints；不是 ETA |

这是初始固定训练量，不是收敛承诺，也不是与 XR1 的算力等量对照。增卡时若要保持同配方，应保留 global batch、更新次数和样本序列；修改任何一项都需作为新的实验条件报告。

GPU2 两卡短训练验收：原生 update1/2/3 loss 日志为 0.2796/0.2536/0.548；新进程恢复后 update3 日志也是0.548，但全张量审计发现 MoT 中 682,917 个元素不同、最大绝对差约3.05e-5。因此只可声明“保存恢复和真实更新通过”，不能声称 bitwise 等价。此处 loss 只作工程回执，不是模型优劣指标。GPU1 两卡尚未同等验收。

## 5. Awomo 真实训练诊断：实现已有，GPU 尚未通过

接手者需要基于公开 Awomo 源码建立本机训练诊断入口；原 GPU 实现的必要目标、输入和恢复语义完整写在下方及 [实现细节](IMPLEMENTATION_DETAILS.md)。这些是待移植合同，不能把公开上游误当成已包含 RoboCasa Human 训练入口。

诊断语义如下，Ascend 适配先保持它：

- 从原 `model.pt` 精确加载 FP32 master，3,876,166,656 个可训练 DiT 参数，包含 proprio encoder；Qwen base、发布 LoRA、VAE 冻结。
- FP32 master / BF16 forward autocast；不在 optimizer 建立前先把可训练权重永久降为 BF16。
- AdamW lr1e-5、betas(0.9,0.999)、eps1e-8、weight_decay0、clip1；batch1；seed20261113。
- 固定 sigma0.5、timestep scale1000；图像和80维动作独立标准正态噪声；目标为图像 velocity MSE + 11有效动作槽 velocity MSE。
- 3 个逻辑更新：update1 后保存完整模型/Adam/RNG/数据 cursor，连续跑2/3；新加载 update1 再跑2/3，一共实际执行5次更新。
- 原候选使用 deterministic algorithms 与 Math SDPA，同设备要求完整状态/损失恢复一致。若 NPU 算子不支持，记录具体差异及更改后的判定，不能直接关掉检查并称原合同通过。
- `Awomo05i.velocity` 本来带 `no_grad`；训练诊断直接调用同一 DiT forward 获得梯度。不能调用推理函数后把“运行成功”当作反向成功。
- 检查所有训练参数的有限梯度、完整参数变化、完整 optimizer/RNG/cursor，检查 frozen encoder 不变。只记录 loss、抽样参数或一个平均 hash 不够。

FP32 master CPU 加载已有 171 个 DiT tensors 的完整证据；两站真实训练 CPU 前置各 35 项测试通过、无跳过。**GPU 前后向、显存可容纳性、5次更新和恢复未执行/未验收；正式 Awomo SFT 的更新数、scheduler、sigma分布、采样策略等仍未冻结。** 固定 sigma0.5 诊断不能直接扩成正式 SFT。

## 6. Recovery SFT：现有 XR1 资料可参考，不能直接继承

本仓库原 XR1 指南已有 R1/R2 数据交付，但 WAM 尚未完成它们的接入和配方验收。WAM Target50 的 Human 覆盖50类；历史 recovery 只覆盖部分任务，不得写成“50类 recovery 全覆盖”。

接手后需逐项完成：来源/父模型/失败类型/成功终点标记审计；原 action/state/camera/fps 与 WAM 映射；episode/segment 边界；片段开头是否具有真实可用历史；未到任务终点的 recovery 如何标注、监督和统计；Human/recovery 混合权重按任务还是全局；训练/验证/测试泄漏与来源模型偏差检查。

不能把独立 recovery segments 拼成连续历史，不能把 segment 结束当作任务成功，不能借用 test rollout 生成训练样本后继续称同一 held-out test。先锁定父模型与混合合同，再做短训练/恢复/闭环验证，最后做正式对照。XR1 的 50:25:25、20k父模型与最终30k结果均不属于 WAM。
