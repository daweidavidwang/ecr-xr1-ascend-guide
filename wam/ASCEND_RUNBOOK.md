# Ascend 接手执行手册

返回 [WAM 总入口](../WAM_HANDOFF.md)。**本页是尚待实施的迁移与验收步骤，不是已验证的 NPU 安装命令或性能保证。** 尚未获知接收端 Ascend 型号/容量/镜像，不凭空指定 CANN、驱动或 torch_npu 版本。

## 0. 接手先建立运行记录

在平台分配的资源内，建立独立 `$WAM_ROOT`、代码分支、环境和 run 目录。记录：

```text
hardware: Ascend 型号、卡数、每卡 HBM、节点数、CPU 架构/内存、磁盘空闲
software: OS/kernel、驱动/固件、CANN、容器 image digest
python: Python、torch、torch_npu、torchvision、transformers、diffusers
distributed: HCCL、DeepSpeed/Accelerate/FSDP 版本和实际拓扑
provenance: 原始 commit、NPU commit、diff、模型/数据/配置 SHA256
execution: 精确 argv、环境变量名称及非敏感值、开始/结束、退出码、资源预算
```

不要把另一机器 GPU 验收当作 NPU 通过，不把所有可见 NPU 当作已分配设备。`npu-smi info`、最小 BF16 matmul、有限数值检查、NPU synchronize 与 HCCL 两 rank all-reduce 的成功分别记录；没有实际多卡时先做单卡，不虚构多卡结论。

原集群由 Monitor 唯一执行远程任务和转移，接收集群则遵守自己的调度 owner。用户要求交接并复测，不等于接管旧 WAM 作业、修改他人驱动或抢占其他任务。

## 1. 环境：两个策略分别准备

下面是 **GPU 对照环境** 的主要版本，不是直接照装的 NPU lock：

| 层 | GPU 对照 | Ascend 处理 |
|---|---|---|
| Loop policy | Python3.10、torch2.7.1+cu128、torchvision0.22.1+cu128 | 选平台兼容 torch/torch_npu 配对，替换 CUDA wheel |
| Loop 上层 | transformers4.49.0、accelerate1.12.0、deepspeed0.18.5、numpy1.26.4、av16.0.1、pyarrow23.0.0 | 先尽量保留；逐项记录因 NPU/CPU 架构造成的变更 |
| Awomo policy | Python3.11、torch2.8.0+cu129、torchvision0.23.0+cu129 | 选受支持 NPU 配对，不能装 cu129 包 |
| Awomo 上层 | transformers5.5.0、diffusers0.39.0、accelerate1.15.0、numpy2.2.5 | 特别验证 Qwen3-VL、Flux2 API 和内部 helper 兼容性 |
| ECR 共同模拟器 | Python3.11.9、torch CPU2.7.1、numpy2.2.5、mujoco3.3.1，固定 RoboCasa/robosuite | 保持独立 evaluator；不在其环境安装 NPU/CUDA policy 依赖 |

完整 GPU 包版本直接列于 [GPU_ENVIRONMENTS.md](GPU_ENVIRONMENTS.md)，包括各 simulator 的公开源码 pins。Awomo `requirements-sft.txt` 原生 RoboDojo 环境与已验 RoboCasa inference 环境不同，不混装后声称依赖未变。

Ascend 的 PyTorch 路线是 CANN + torch_npu，不要求先重写为 MindSpore。官方 [TorchNPU](https://github.com/Ascend/pytorch)、[DeepSpeed NPU setup](https://www.deepspeed.ai/tutorials/accelerator-setup-guide/#huawei-ascend-npu) 可作为起点；DeepSpeed 支持 NPU 不代表本项目固定配置的所有 op/ZeRO/保存路径自动通过。

`transfer_to_npu` 可以辅助迁移，但不应代替逐项检查。既有代码的显式 CUDA 类型判断、特定设备断言、数学后端选择、checkpoint RNG、device mesh 等仍需审查。改完运行依赖检查、保存完整 freeze，模型从固定本地路径加载，关闭运行时自动拉取 latest。

## 2. 必须修改/验证的源码位置

### LoopWAM

| 位置 | 当前行为 | NPU 工作 |
|---|---|---|
| `pyproject.toml` | torch/torchvision 固定 `+cu128` | 分离 NPU 依赖，保留原 GPU 对照文件 |
| `src/fastwam/models/wan22/wan_video_dit.py` | 名为 `flash_attention` 的函数实际调用 PyTorch SDPA；RoPE 用 float64、complex/view_as_complex；已有局部 npu 分支 | 验证带 mask SDPA 和 RoPE 前后向；必要时用实数 sin/cos 等价实现并比较输出/梯度 |
| `wan_video_vae.py` | VAE 注意力和视频编码/解码 | 单独检查实际 shape、dtype、卷积/注意力算子和峰值内存 |
| `fastwam.py` | 双设备推理要求 device.type==cuda，调用 CUDA synchronize/empty_cache | 改为明确设备抽象；跨卡分支与 KV cache 移动单独验收 |
| `src/fastwam/trainer.py` | Accelerate + DeepSpeed + AdamW，原数据/normalizer/恢复逻辑 | HCCL/ZeRO、数据顺序、优化器状态、保存恢复、真实梯度都要验 |
| `scripts/accelerate_configs/accelerate_zero2_ds.yaml` | NVIDIA 训练布局 | 生成本机 NPU launch 配置，不把 CUDA 扩展留在依赖中 |

RoPE 里存在 `freqs.device.type == 'npu'` 只说明有局部兼容分支；其输入还用 float64/view_as_complex，不能据此宣布完整昇腾支持。也不要因为函数名叫 flash_attention 就盲目安装 CUDA flash-attn。

### Awomo-0.5I

推理位置对应公开原版 commit；`awomo/training/` 对应另列的公开 SFT 基底，原版评测 checkout 不含此目录。

| 位置 | 当前行为 | NPU 工作 |
|---|---|---|
| `awomo/awomo_05i/model.py` | Qwen3-VL、Flux2Transformer2DModel、Flux2 VAE，默认 CUDA、TF32 设置 | 显式 NPU device；单独测试 VLM hidden layers、VAE 和自定义图像/动作 tokens 的 DiT |
| `awomo/awomo_05i/prompt.py` / `eval_robocasa365/common.py` | 原生 prompt、图像序列、动作和状态映射 | 保持语义；验证0/部分/9个有效历史槽的不同 shape |
| `awomo/training/trainer.py` | 明确断言 CUDA、访问 CUDA 内存信息 | 改为目标设备 API，不能只替换字符串 |
| `awomo/training/distributed/fsdp.py` | FSDP2/fully_shard，NCCL | 换 HCCL 并验证实际分片/通信/optimizer state |
| `awomo/training/distributed/mesh_utils.py` | nccl→cuda，否则cpu | 必须新增真正 NPU mesh 分支，否则改 backend 后仍可能落到 CPU |
| 本交接的训练诊断规格 | FP32 master、BF16 autocast、固定真实样本、完整恢复比较 | 按 [实现细节](IMPLEMENTATION_DETAILS.md) 建本机入口，保存输入/实现/运行证据 |

Diffusers 文档已有 [`_native_npu` attention](https://huggingface.co/docs/diffusers/optimization/attention_backends)，昇腾有 [Qwen3-VL/FSDP 训练案例](https://www.hiascend.com/developer/techArticles/20260127-2)。这只支持“存在适配路线”的判断。Awomo 完整组合、特殊 bool mask、历史 tokens、前反向与固定版本仍需实测；其他型号/版本的 Qwen3-VL 成功不能替代当前4B模型验收。

## 3. 阶段 A：输入与 CPU 合同

先校验 [模型/资产](pins/public-inputs-v1.json)、[Human](pins/human-hf-full-v1.json) 全部文件与 licenses，按 [实现细节](IMPLEMENTATION_DETAILS.md) 重建整 episode 划分、共享视频 offset 视图、Awomo 样本和采样游标。该页给出可复制辅助代码和公开原生参照；上游没有本项目新增测试集合，不能照抄不存在的测试命令。

为本机实现建立并运行这些 CPU 验收：有效/非法动作与 state；每相机不同非零offset；episode边界；0/4/9历史槽；原生请求压缩与画布逐像素一致；原生与标准processor输出一致；无未来条件泄漏；train/validation身份不交叉；world1/2/8/16采样顺序和cursor恢复。记录实际检查数、失败和skip。历史完整集合两站各43项无skip只描述原GPU线前置，不能当作当前分支的结果。

CPU 输出必须满足：动作 roundtrip、state 编码、相机 offset、history valid mask、future target 不入 prompt、训练/验证不交叉、采样 cursor 恢复。NPZ/元数据严格校验；这些不能替代下一阶段的神经网络前向。

## 4. 阶段 B：单卡加载与数值对照

逐组件执行：Loop UMT5 → VAE → video/action expert；Awomo Qwen+发布LoRA → VAE → DiT/proprio。然后执行一次完整原生策略，检查动作维度、有限性、原生反归一化/映射及实际设备落点，记录 CPU fallback。模型成功 load 不代表推理完成。

使用固定开发输入分别覆盖：静态单步、早期0 history、部分history、完整9 history、不同文字长度和图像 token 数。不要使用正式 test 选 attention backend 或阈值。

跨 GPU/NPU 对照需另外生成明确 fixture：

1. 同一 CPU 源 batch、token IDs、image grids、state/action、mask；相同模型文件。
2. 相同预生成噪声张量和 timestep/sigma。**相同 seed 不保证 CUDA/NPU 产生相同随机张量**。
3. 对照 VAE latent、VLM context、DiT image/action velocity、解码动作；训练再对照 loss 分项、梯度范数和一次更新。
4. 记录 max/mean/p99 absolute error、relative error（注明零分母规则）、有限性、动作越界/离散 gripper/mode 翻转；报告 dtype/backend/shape。
5. 先测 GPU 同后端重复误差，结合任务量纲预先写明阈值，再判断 NPU；不能看完 NPU 后放宽阈值。跨硬件不承诺 bitwise equality。

当前交接中的 CPU/pixel NPZ 不包含完整已验 GPU/NPU 前反向参考。需由有容量的 GPU 执行端生成固定参考并回交 SHA/环境/源码，不能把“已有 GPU 曾推理成功”替代数值对照。

## 5. 阶段 C：策略服务与 RoboCasa 闭环

推荐拆分：**Ascend policy server + 原 NVIDIA/OpenGL evaluator**。Loop 保留其 HTTP 协议，Awomo 保留 msgpack 协议；按源码参数配置绑定地址、端口与客户端目标，确认消息大小/压缩和超时。现有包装器默认同机/localhost，需要明确适配连接地址，不能原样在两机之间运行。

通过已有受控内网连接，不将未认证策略服务公开到互联网。保留会话 reset、历史缓存清理、请求顺序与每 episode 的策略 RNG 语义。分别记录网络等待、模型推理和仿真耗时；网络引入的延迟不能算成 NPU 算子速度。

Ascend NPU 不替代 MuJoCo OpenGL 渲染设备。若目标机器仅有 NPU，可做 CPU OSMesa 工程验证；其图像/性能与当前 NVIDIA evaluator 的一致性必须另验，未验时不能归入正式硬件效果对照。参照 [MuJoCo 文档](https://mujoco.readthedocs.io/en/3.2.5/programming/)。

先在开发场景完成短闭环和长历史，再跑完整 horizon。共同 ECR 初始两个 dev 请求是 NavigateKitchen（450步）、OpenDrawer（750步）；不能误写为450/450，也不能把64步诊断算作完整任务。逐项检查实际 reset 初态、图像、动作包装和终止规则，详见 [评测合同](EVALUATION.md)。

## 6. 阶段 D：训练与恢复

先移植短训练入口，完成真 backward、optimizer update、参数变化和 frozen 参数不变；保存包括 model/optimizer/scheduler/scaler（如使用）/各 rank RNG/数据 cursor 的完整状态，新进程恢复。

- Loop 按已有3步两rank原生诊断语义验证，再做固定1,000步 Human 配方。原GPU恢复不是逐位一致，NPU报告应明确采用怎样的数值标准。
- Awomo 按 [数据与训练合同](DATA_AND_TRAINING.md) 的3逻辑/5实际更新诊断验收。需要新 NPU 入口；没有一个已验证的 `train_ascend.sh` 可直接调用。
- 同设备恢复比较与跨设备数值比较分开。不能直接用 CUDA RNG state 去恢复 NPU，不能假设 CUDA sharded optimizer checkpoint 能被 HCCL 新布局直接加载。
- 参数/梯度/Adam状态至少核对全部张量的形状、dtype、有限性、step与变化；保存加载成功后还要继续更新并证明读取了预期下一个样本。
- 多卡增加 HCCL、FSDP/ZeRO 分片、rank均衡、global batch和恢复布局检查；CPU 模拟16 rank采样不算16卡训练通过。

若单卡放不下，记录实际峰值与失败位置，再选择 checkpointing、切分或 offload。新机制需要重复对应数值/恢复验证，不在未测前保证某个卡数足够。吞吐跑与正式训练分开，正式训练重新从固定发布权重和明确的 optimizer 初态开始；诊断权重不得作为正式候选。

## 7. 阶段 E：正式复测与性能报告

工程通过后，先固定 port commit、模型/数据/环境、attention实现、dtype、推理步数、训练配方和预算，再运行本次 Ascend 的完整对照。保留原 NVIDIA 结果，不将 NPU 重跑覆盖成原 run。

至少分别报告：

| 指标 | 定义 |
|---|---|
| 模型推理 | 同 batch/history/image/token shape、同采样步数；预热后 NPU synchronize 的延迟 p50/p95、动作 chunk/s |
| 训练吞吐 | 相同 global batch、累积和更新数；样本/s、optimizer update/s；预热与稳定窗口分开 |
| 内存 | 每卡实际峰值 allocated/reserved 与主机内存；注明是否含全部编码器/optimizer/offload |
| 端到端耗时 | 下载/初始化、训练、保存导出、网络、仿真/评测各自计时 |
| 策略效果 | 每任务成功/有效完成/计划局数、错误、partial、split和场景身份；统一 macro/micro 定义 |
| 可靠性 | 运行/恢复次数、故障类型、重试是否相同身份、是否有 fallback 或算子替换 |

不把“可以训练”写成“比 B300 快”，不把 CPU fallback 隐藏在 NPU 吞吐里，不拿不同 precision/步数/图像尺寸的速度直接作硬件比值。正式 Target50 需要其全场景/完整dev验收，不因单次NPU前向成功跳过。

## 8. 回交目录和完成条件

建议在自己的工作区保存以下结构；公开 GitHub 只提交可公开的文本、代码差异与摘要，大文件/私有输入继续留私有存储：

```text
wam-ascend/<run-id>/
  environment.json          # hardware、镜像、软件和依赖锁摘要
  provenance.json           # source/base/port/model/data/config pins
  commands.json             # 实际 argv、有效配置、资源和时间范围
  input-verification.json   # 全文件集合、bytes/digests、数据划分
  numerics.json             # GPU/NPU逐层和完整策略对照、阈值与结论
  train-smoke.json          # 更新、冻结参数、保存/恢复、rank状态
  performance.json          # 计时方法、样本数、预热/稳态、内存
  episodes.jsonl            # 独立run命名空间，每身份至多一个有效结果
  artifact-index.json       # 私有大文件位置、不可变revision、SHA256
  REPORT.md                 # 结论、完成/未完成、问题和下一步
```

完成等级按证据逐层标注：环境通过 → 推理通过 → 数值对照通过 → 短闭环通过 → 训练恢复通过 → 多卡通过 → 完整协议复测。每一级允许单独交付，但不能用前一级替代后一级。若阻塞在算子/容量/权限，提供精确失败位置和已完成独立工作；不要填写假的“passed”解除启动检查。
