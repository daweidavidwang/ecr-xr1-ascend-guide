# Target50 数据、模型下载与输入验收

返回 [交接入口](../TARGET50_HANDOFF.md)。所有命令从本公开 GitHub checkout 执行。以下下载 Human50 使用公开 Hugging Face，不需要历史私有 ModelScope Human10 仓库权限。新的Target50私有源码包已云端交付，见[固定下载与解压命令](SOURCE_AND_STATUS.md)。2026-11-13入口复核见[访问验收摘要](input-access-verification.json)：12个Human metadata实下载、原始XR全部22个文件元数据及小config实下载通过；大文件仍按清单在Ascend全量下载验SHA。

## 1. 固定三个数据源

| 组 | 公开 Hugging Face dataset | revision | episodes | frames |
|---|---|---|---:|---:|
| atomic_seen，18任务 | [ember-lab-berkeley/robocasa365-target-atomic](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-target-atomic) | `ee95cbe19134b1762b2d50fba87f5406d4d59143` | 9,126 | 2,231,347 |
| composite_seen，16任务 | [ember-lab-berkeley/robocasa365-target-composite-seen](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-target-composite-seen) | `d431cfc244ecfa55865ea3bf8681efa0fd2e54b5` | 8,077 | 6,002,265 |
| composite_unseen，16任务 | [ember-lab-berkeley/robocasa365-target-composite-unseen](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-target-composite-unseen) | `33f25a67c1a3116477a6c06f20ef22db3c7797ae` | 8,104 | 6,724,287 |
| 合计 | 50 tasks | 固定版本，不读main | **25,307** | **14,957,899** |

[human50-download.jsonl](human50-download.jsonl) 列出实际训练使用的307个源文件、各自repo/revision/远端路径/本地路径/bytes/SHA256；合计 **43,253,612,263 bytes**。三个派生episode map合计32,904,734 bytes。B300“316文件”包含307源文件、3个map及6个stage/readiness/config文件，并不意味着要从HF找316个源文件。

## 2. 下载与重建索引

```bash
git clone https://github.com/daweidavidwang/ecr-xr1-ascend-guide.git
cd ecr-xr1-ascend-guide
export ECR_GUIDE="$PWD"
export ECR_ROOT=/data/ecr-target50-ascend
mkdir -p "$ECR_ROOT"
sha256sum -c PUBLIC_SHA256SUMS

# 只下载12个metadata文件，先验证网络及固定入口。
python3 target50/download_human50.py --root "$ECR_ROOT" --metadata-only

# 完整307文件；已有内容逐个验SHA，不覆盖不同版本。
python3 target50/download_human50.py --root "$ECR_ROOT" --workers 4
python3 target50/download_human50.py --root "$ECR_ROOT" --check-only

# 独立数据环境；不修改平台训练环境。
python3.12 -m venv "$ECR_ROOT/envs/data-check"
"$ECR_ROOT/envs/data-check/bin/python" -m pip install 'pyarrow==21.0.0'
"$ECR_ROOT/envs/data-check/bin/python" target50/build_episode_maps.py --root "$ECR_ROOT"
```

下载工具使用Python标准库，推荐Python3.10+；自动跟随HF文件分发重定向。失败保留partial和错误回执，重跑重新下载未完成文件；**不支持字节断点续传**，不要用很多重试填满磁盘。307文件全部通过才代表载荷完整；`--metadata-only`仅证明12个metadata通过，不是训练ready。下载日志在 `$ECR_ROOT/human50-download-receipt-*.json`。

数据环境推荐Python3.11/3.12；不要用Python3.14强行编译pyarrow21。若系统Python缺venv，可使用平台已有独立环境或`uv venv --python python3.12`，不要修改系统驱动/环境。

索引工具从各固定 `meta/episodes/chunk-000/file-000.parquet` 重建原episode identity、任务和视频时间区间，逐任务核对冻结计数，并要求生成字节与 [human50-mapping-pins.json](human50-mapping-pins.json) 的三个 SHA 完全一致。不能把分组中的全局episode_index直接当任务内source_episode_index。

```text
$ECR_ROOT/datasets/target50_human_v3_20261112/
├── atomic_seen/
│   ├── meta/info.json, stats.json, tasks.parquet
│   ├── meta/episodes/chunk-000/file-000.parquet
│   ├── data/chunk-*/file-*.parquet
│   └── videos/observation.images.<camera>/chunk-*/file-*.mp4
├── composite_seen/       # 同样结构
├── composite_unseen/     # 同样结构
└── maps/
    ├── atomic_seen-episodes.jsonl
    ├── composite_seen-episodes.jsonl
    └── composite_unseen-episodes.jsonl
```

每个视频文件可包含多个episode；必须按metadata中的from/to_timestamp取帧，20fps、256×256、RGB。三相机为 `robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand`。不重编码、不截短数字轨迹来迁就视频、不丢相机。

## 3. 原始 Xiaomi base

入口：[XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365](https://modelscope.cn/models/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365)。使用仓库已有 [22-file download map](../downloads/xiaomi-base-download.jsonl)；其每文件revision、bytes和SHA已与当前Target50实际base逐项对齐，见 [xr1-base-pins.json](xr1-base-pins.json)。该清单按文件固定版本，不能自行换成一个猜测的统一revision。

```bash
cd "$ECR_GUIDE"
python3.12 -m venv "$ECR_ROOT/envs/model-download"
"$ECR_ROOT/envs/model-download/bin/python" -m pip install 'modelscope-hub==0.6.1'
"$ECR_ROOT/envs/model-download/bin/python" downloads/download_files.py \
  downloads/xiaomi-base-download.jsonl --root "$ECR_ROOT" --workers 4
```

恢复到 `$ECR_ROOT/models/xiaomi-original/`，22文件合计 **10,122,587,755 bytes**。必须包含完整processor/tokenizer/model代码和权重分片，不能只校验config。下载器如需账号凭据，按 [DOWNLOADS.md](../DOWNLOADS.md) 的secret流程；不要将token放进参数或报告。

Target50 H50和R1都从这个原始base开始。旧 `xr1-a3-s1-cont-20k` 是10-task R2 Continue的父模型，**不是H50或R1-50起点**。

## 4. 输入语义与内容验收

- 原始state16：base位置3、base四元数4、EEF相对位置3、EEF相对四元数4、gripper_qpos2。由固定XR上游 `observation_to_state` 转成14维再填充到模型60维；保留四元数/旋转处理，不能换成简单重排。
- 原始action12：base_motion[0:4]、control_mode[4]、EEF位置[5:8]、EEF旋转[8:11]、gripper[11]。XR次序为 `[5,6,7,8,9,10,11,0,1,2,3,4]`，填充到60维，未来 **16** 步；末尾重复最后真实action，保持原mask。
- history为 `max(0,t-6), max(0,t-4), max(0,t-2), t`；三路图像crop0.95，调用原processor，`do_resize=False`，robot_type=`robocasa365`。Human指令前加 `[human] `，不要用任务类名代替原task_description。
- 原始v3 reader将state保留float64、action转float32，再走XR变换；不应用其他项目normalizer，不drop action channel。`meta/stats.json`为空，不可假装已经有可用归一化统计。
- 全量核验episode/frame/task身份、长度、连续frame_index、timestamp、state16/action12有限值及任务计数；每任务抽起点/中段/末尾和尾部padding，检查三相机PTS及实际processor输出。文件SHA与模型数值验收分别记录。

B300历史验收有全部数字数据检查及视频抽样，不宣称把全部视频逐帧与旧Box格式对照。对Ascend本机还需重做文件与实际reader检查。三个HF仓库是本次固定使用的发布者转换版本；未证明与全部旧Box原件逐字节相同。

## 5. 空间与未来数据

Human源文件43.25GB、map约33MB、原始模型10.12GB；模型SDK缓存另计。正式训练DCP目前约59.11GB/个，不能只预留输入大小。B300按最多13个DCP加HF导出与scratch规划 **约1.2TB/seed、两seed约2.4TB**，Ascend须按自己的checkpoint格式实测并保留余量。采集/评测视频、恢复数据和独立归档另算，未经测量不承诺总TB数。

**当前没有可下载的Target50 D1/D2/D3。** 旧 [ModelScope下载说明](../DOWNLOADS.md) 的R1/R2是历史9任务数据，仅供旧实验复现；不要混到新50任务主链。新轮次以后必须提供实际manifest/revision/SHA及覆盖表，再加入本指南。
