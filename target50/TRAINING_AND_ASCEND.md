# 固定训练配方、Ascend 适配与计时

返回 [Target50交接](../TARGET50_HANDOFF.md)。本页区分已经在B300运行的实现与待Ascend验收的移植。

## 1. H50 科学参数

| 参数 | 固定值 |
|---|---|
| 初始化 | 完整22文件原始 `Xiaomi-Robotics-1-RoboCasa365`；每seed从step0独立开始 |
| 阶段步数 | **48,000 optimizer updates**，不是microsteps |
| 规模公式 | `S=1000*ceil(10000*14957899/3176983/1000)=48000`；曝光量预算，不是已收敛结论 |
| seeds | 1、2 |
| global batch | **192 observations/update** |
| GPU参考布局 | 4 devices × micro12 × accumulation4 |
| 8-device候选布局 | micro6 × accumulation4，或micro12 × accumulation2，均为192；Ascend实测后固定一种 |
| worker | 原实现每rank 8，prefetch_factor4，FIFO；变更拓扑须重新验证采样和恢复 |
| optimizer | AdamW，β=(0.9,0.95)，eps=1e-8；GPU `foreach=True` |
| weight decay | 0.1：仅requires_grad、ndim≥2、name不含`norm`/`adaln`；其余0 |
| LR | 起始5e-7，warmup2400到peak2e-5，其后cosine降至min5e-6 |
| clipping | 全局L2梯度norm，阈值1.0，在完整accumulation后执行 |
| 精度/并行 | FP32 master，BF16 compute，FP32 reduction；FSDP2，HF导出BF16 |
| 训练参数 | 冻结语言embedding及tied权重；其余按原实现全量SFT，无LoRA |
| activation checkpoint | text MLP与vision block，non-reentrant；保留原粒度 |
| 输入 | 三相机，history4/interval2/crop0.95，模型state/action padding维度60 |
| 预测目标 | **16步action horizon**；原12维action重排，尾部重复最后真实动作 |
| 标签 | Human `[human]`；后续D1/D2/D3统一 `[recovery]` |
| 保存 | 每5000及每作业段末完整restart；19,200 / 33,600 / 48,000额外HF导出 |
| 选点 | 三个HF候选分别dev1000，按dev选择后test2500 |

LR使用更新前零基索引 `u`：warmup时 `5e-7+(2e-5-5e-7)*u/2400`；其后 `5e-6+0.5*(2e-5-5e-6)*(1+cos(pi*(u-2400)/(48000-2400)))`。日志step=`u+1`。最后一次update用u=47999；不要因为更换scheduler产生一步偏移。

模型每个观测的VLM context复用4次独立flow `(t,noise)`，**不把global batch标成768**。`t=0.999*(1-Beta(1.5,1.0))`，Gaussian noise与原模型一致；`x_t=(1-t)*noise+t*action`，速度目标`action-noise`。损失为 **0.5×masked MSE + 1.0×FFT项**。FFT沿时间维rFFT，比较预测/目标频谱差的复数绝对值；排除XR action维6、11（gripper/control_mode）及无效padding，保留原mask语义。不要省略FFT、改loss权重或用不同归一化后称同配方。

历史FSDP实现还加零权重logits项以触发末层output backward hook；数学贡献为0，但删除可能使通过KV缓存传回的梯度无法正确unshard。移植需要验证这一依赖，不只看loss有限。

## 2. 采样和完整恢复

Human sampler按episode长度加权选择episode，每次访问抽 `min(16,T)` 个**不重复**anchors，进入256-anchor shuffle buffer；worker/rank随机数独立。别把它换成PyTorch随机打乱一遍dataset的epoch式训练。48k×192=9,216,000观测抽样，小于全部Human帧数；**不意味着每一帧被遍历或做了三遍训练**。

后续R2短recovery使用每次visit16个**可重复**anchors，区别于H50。每阶段保留任务和来源实际抽样计数，不能仅在配置里写比例。

两种“续训”必须区分：

- **同一48k阶段因时限分段**：完整恢复model、optimizer moments、globalstep、学习率进度、每rank设备/CPU/Python/NumPy RNG、每worker最后已消费位置、shuffle buffer及processor RNG；丢弃尚未消费的prefetch并精确重放。不能以HF权重替代全状态restart。
- **新语义阶段，如R1→R1-cont或R1teacher→R2**：加载父模型权重，重建optimizer/schedule，记录本段与累计step。父模型和数据版本重新冻结。

B300分段已实现0→1000→10000→19200，后续到33600/48000；作业段末不改变总48000的日程。单作业控制在24h以内，按实测吞吐缩短资源分段。新Ascend端不直接反序列化B300的分片DCP来冒充无缝恢复；本实验的跨硬件对照从base重新训练。

## 3. 固定软件身份

- 当前私有源码补充包：`4bf26d9831a355651b820eb116d8b02ea44f300d`；实际Human训练核心11文件版本 `d0fcba205f5dcfdd502159ae8d6cdd8a80b17b59`。后续改动主要是调度/评测准备；以run contract的逐文件SHA为最终身份。
- [XR上游仓库](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)，commit `0dd7aef8dc87296246aae812a1f59ccb708e5546`。不要跟随main。
- B300环境参考：Torch2.9.1/CUDA13.0；Transformers4.57.1、NumPy2.1.3、Pillow11.3.0、safetensors0.6.2、accelerate1.11.0、einops0.8.1、av15.1.0、pandas2.3.3、pyarrow21.0.0、imageio2.37.0。这组是GPU来源，不是Ascend已验证组合。
- Ascend按目标平台镜像与[官方TorchNPU兼容说明](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)选定CANN、driver、PyTorch、torch-npu。记录Python/CPU架构、镜像digest、wheel版本、设备型号/HBM/互连。不要盲装GPU Torch或CUDA flash-attn到NPU环境。

```bash
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1 \
  "$ECR_ROOT/code/Xiaomi-Robotics-1"
git -C "$ECR_ROOT/code/Xiaomi-Robotics-1" checkout --detach \
  0dd7aef8dc87296246aae812a1f59ccb708e5546
```

这里公共上游不含ECR自己的完整训练器。新的私有补充包取得方式见 [SOURCE_AND_STATUS.md](SOURCE_AND_STATUS.md)。

## 4. 需要实现和审计的 Ascend 改动

| 原GPU实现 | NPU迁移要求 |
|---|---|
| `torch.cuda`、NCCL、CUDA RNG/同步 | 使用经当前平台验证的NPU设备、HCCL、NPU RNG、sync及计时；不能只全局替换字符串 |
| VLM / DiT FlashAttention2 | 验证NPU attention的causal/padding mask、KV cache、RoPE、dtype与梯度；记录真实kernel，拒绝静默CPU fallback |
| cuDNN Conv3d layout patch | B300补丁要求CUDA且检查cuDNN，不可直接调用；在NPU验证同形状卷积、layout与梯度，不改变参数身份 |
| `torch.fft.rfft`与复数abs | 验证支持性、FP32语义、autograd和mask；不为开跑删掉FFT项 |
| FSDP2 / DTensor / optimizer foreach | 检验平台支持、全局grad norm及FP32归约；若换并行或optimizer kernel，记录差异并重做数值/恢复验收 |
| DCP与rank state | 实现NPU全状态存取、完整性标记、RNG/worker consumed cursor恢复；强制同一训练合同 |
| B300 path / readiness / hostname gates | 路径变更集中在机器适配层；形成Ascend新contract，保留原内容pins，不能删除所有检查来绕过硬编码 |

核心文件路径均相对私有包的 `source/`：

```text
training/ecr_target50_20261112/
  train_human.py             # 正式训练、DCP、HF重载检查
  human_v3.py                # 固定v3 episode与视频读取
  human_loader.py            # 实际processor、worker、已消费位置恢复
  human_stream.py            # Human采样buffer/RNG
  human_recipe.py, human_v3_recipe.py
  config/tasks50.json, xr1_base_pins.json
  b300/scripts/strict_runtime.py, patch_layout.py   # GPU参考，不是NPU补丁
  scene_manifest.py, evaluation_results.py, evaluation_workplan.py
training/rc365_xr1_20261006/scripts/
  train_xr1.py               # FlowObjective、LR、FSDP、export
  rc365_data.py              # action/state/图像、Human和历史混合语义
  round2_sampler.py          # 后续recovery参考
```

`human_recipe` / `human_v3_recipe` 当前严格绑定B300根、receipt和recipe schema。Ascend适配须显式产出新的本机readiness/contract，不能直接修改B300原回执SHA或假称通过原硬件验收。Human完整文件SHA、map SHA、任务计数、base pins、科学参数应继续强制检查。每次run保留原GPU合同与Ascend合同的结构化差异。

## 5. 验收后才能正式启动

1. **CPU输入**：50任务文件/数字行/视频抽样与processor检查；输入tensor形状、dtype、task prompt、mask、尾部action逐项比较。
2. **单设备数值**：固定同一base、预处理batch、flow t和noise，比较forward总loss/MSE/FFT、代表层与全局梯度、一次更新。FP32参考、同后端重复与跨后端BF16误差分别记录；先规定容差，失败时不现场放宽阈值。
3. **多设备20-step短跑**：global192、多卡梯度归约、采样计数、有限loss/grad。非科学结果，独立新目录。
4. **分段恢复**：同后端连续训练与中断保存→新进程恢复→同样总step对比；实际多worker/预取启用。核验后续batch、RNG、optimizer及参数，而不只检查checkpoint能打开。跨设备world_size变更是另一个恢复测试，默认不支持随意改变。
5. **导出**：完整HF目录、逐文件SHA；重载后逐tensor检查BF16导出一致性（B300已用1121 tensor检查，NPU以同模型状态集合为准）。在冻结GPU policy server真正加载并验证动作形状/有限值。
6. **计时**：独立warmup与稳定区间计时。然后重新从base初始化并清空短跑optimizer/RNG状态，开始正式训练。

原GPU训练入口接口为：

```text
torchrun --standalone --nproc-per-node=4 training/ecr_target50_20261112/train_human.py
  --recipe <human_s1.json> --out <new-run-directory>
  --segment-end 1000 --b300-strict-runtime
```

**这只是原GPU接口，不是Ascend可复制启动命令。** 接手者应创建已验收的NPU入口（名称由移植实现决定），提供其实际命令与配置，不把尚未存在的`train_npu.py`写成已提供文件。端到端启动脚本应先核验输入/合同，再以集群分配卡数启动，并保留stdout/stderr、resolved config、git diff和env lock。

## 6. 效果与训练耗时怎么比较

先做配方对齐版；更换attention、编译、fused optimizer、并行布局的性能版本分别编号，不能把不同global batch/更新数的walltime直接比较。

每seed记录：实际设备数量N、峰值HBM、初始化/首步编译时间、至少200个稳定update的mean/p50/p90秒数、读数据时间、存盘/导出时间、总训练walltime和排队时间。计时前后用对应后端同步；记录测量区间是否包括IO。`samples/s=192/mean_step_seconds`；`device-hours=N*training_wall_hours`；剩余纯更新时间=`remaining_updates*mean_step_seconds/3600`，再单列IO及排队。

当前B300下一段10000→19200的来源估计是s1约9.9h、s2约10.2h纯更新，另加恢复/IO；它不是Ascend ETA，也不是已经完成48k的计时证据。首个NPU稳定区间出来前，不承诺NPU完成日期。

报告同时保留两种对比：相同update预算下的质量/耗时，以及达到同一预设dev门槛的训练时间。不要根据test分数反复选checkpoint或选择最有利硬件版本。两seed单列并给均值；不要只报告最高的一次。
