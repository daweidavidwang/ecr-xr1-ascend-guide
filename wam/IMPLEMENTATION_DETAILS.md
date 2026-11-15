# WAM 自包含实现细节

返回 [交接入口](../WAM_HANDOFF.md)。本文直接提供重建所需的数据语义、核心辅助代码和验收步骤，只依赖本目录清单与固定公开上游。代码块是工程辅助实现，**不是完成的 Ascend 训练器**。把它们接入自己的分支并记录差异；历史 GPU 检查结果不能自动归给新实现。

## 1. Human 源身份与整 episode 划分

下载并验证 [307个源文件](pins/human-hf-full-v1.json)。按 `component` 的三个组读取 `meta/info.json`、`meta/tasks.parquet` 和 `meta/episodes/chunk-000/file-000.parquet`。任务对应关系由 [任务表](pins/target50-tasks.json) 的 `human_source_prefix` 与 episode metadata 的 `source_prefix` **精确匹配**；不能只从自然语言任务字符串猜测类别。

每组按源全局 `episode_index` 排序，验证从0连续、`dataset_from_index` 与前一episode的 `dataset_to_index` 相接、两者差等于正整数 `length`、总帧数等于info。`(source_prefix, source_episode_index)` 必须唯一，每任务的 source_episode_index 从0连续。三相机分别读取 `videos/observation.images.<camera>/{chunk_index,file_index,from_timestamp,to_timestamp}`，验证20Hz、有限非负时间、区间长度与数值帧数一致，同一视频内episode区间不交叠。

下面代码精确恢复每任务99/1划分；`rows` 是同一任务的全部源 metadata 行。每次任务调用都重新建立seed42生成器，**不是一个全局随机流**。

```python
import numpy as np

def partition_task(rows):
    rows = sorted(rows, key=lambda row: row['source_episode_index'])
    assert [r['source_episode_index'] for r in rows] == list(range(len(rows)))
    order = np.arange(len(rows))
    np.random.default_rng(42).shuffle(order)
    train = set(map(int, order[:int(len(rows) * .99)]))
    validation = set(range(len(rows))) - train
    assert not train & validation
    return rows, sorted(train), sorted(validation)
```

冻结计数：train25,012ep/14,783,844frames；validation295ep/174,055frames；共50任务/25,307ep/14,957,899frames。源 task instructions 全量保留；下面固定processor诊断使用该episode的 `tasks[0].strip()`。

### Loop v2.1 视图重建

每任务独立目录 `<group>/<task>/`；保留原视频字节，可用只读源视频链接避免复制。映射后的 `episode_index=source_episode_index`，每任务 `index` 从0按episode顺序累计。按源 metadata 定位共享parquet：`data/chunk-{data/chunk_index:03d}/file-{data/file_index:03d}.parquet`；切片起点 `dataset_from_index - shard第一行index`，长度 `length`。逐行验证原episode、frame_index和timestamp，**只改变episode_index和index两列**；其余数值列要求精确相同。

数值列按原转换顺序为 `action`、`observation.state`、`timestamp`、`index`、`episode_index`、`frame_index`、`task_index`。沿用源Arrow schema，写zstd parquet到 `data/chunk-{episode_index//1000:03d}/episode_{episode_index:06d}.parquet`。镜像元数据为：

- `codebase_version=v2.1`，fps20，chunks_size1000，total_episodes/frames按本任务；total_videos=3×episodes，total_chunks=ceil(episodes/1000)。
- `data_path=data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`，`video_path=videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`；每个逻辑视频链接到metadata指定的真实共享源文件。
- `meta/tasks.jsonl` 保留源tasks.parquet行；`meta/episodes.jsonl` 每行保存本地episode_index、length、tasks、三相机 `shared_video_offsets` 和完整 `source_identity`。后者保留原行并增加group/task/target50_index/partition。
- `human-partition.json` 写排序后的 `train_episodes`、`validation_episodes`。加载器seed42、val_set_proportion=.01、max_episodes_per_dataset=None；重新对照实际读取的episode集合。
- `meta/info.json` 增加 `shared_video_offsets={"format":"wam-hf-v3-to-v2-view-v1","source_manifest_sha256":"<本次已验证源清单的SHA256>"}`。这里填写实际清单SHA，不填占位字串。
- 使用公开Loop的 `compute_episode_stats(..., is_compute_episode_stats_image=False)` 生成 `meta/episodes_stats.jsonl`，并让原生metadata类重新读入所有episode。metadata stats不等于训练normalizer；本轮从发布 `dataset_stats.json` 起步，不用validation重估normalizer。

以下函数可保存到公开Loop源码的 `src/fastwam/datasets/lerobot/lerobot/datasets/shared_video_timestamps.py`。它只平移视频查询，不能平移数值timestamp、动作索引或padding。

```python
"""Explicit timestamp mapping for byte-preserving Human v3 episode views.

Ordinary v2.1 datasets retain the original query timestamps. A shared-video view
must opt in through its metadata and supply a separate interval for every camera.
Offsets never change frame indices, numeric timestamps, action windows or padding.
"""
import math
import re


FORMAT = "wam-hf-v3-to-v2-view-v1"
FIELD = "shared_video_offsets"


def video_query_timestamps(info, episode, video_keys, video_key, timestamps, tolerance_s):
    marker = info.get(FIELD)
    intervals = episode.get(FIELD)
    if marker is None:
        if intervals is not None:
            raise ValueError("Shared video intervals require an explicit dataset format marker")
        return timestamps
    if not isinstance(marker, dict) or marker.get("format") != FORMAT:
        raise ValueError("Unrecognized shared-video view format")
    if not re.fullmatch(r"[0-9a-f]{64}", marker.get("source_manifest_sha256", "")):
        raise ValueError("A shared-video view must name its sealed source manifest")
    if not isinstance(intervals, dict) or set(intervals) != set(video_keys):
        raise ValueError("Every camera requires its own source video interval")
    if video_key not in video_keys:
        raise ValueError("Unknown camera")
    fps, length = info["fps"], episode["length"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frame rate")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("Invalid episode length")
    if not math.isfinite(tolerance_s) or not 0 < tolerance_s < 0.5 / fps:
        raise ValueError("Invalid video timestamp tolerance")
    for camera in video_keys:
        interval = intervals[camera]
        if not isinstance(interval, dict) or set(interval) != {"from_timestamp", "to_timestamp"}:
            raise ValueError("Malformed camera interval")
        start, end = interval["from_timestamp"], interval["to_timestamp"]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)):
            raise ValueError("Nonfinite camera interval")
        if start < 0 or end <= start or abs((end - start) * fps - length) > 0.01:
            raise ValueError("Camera interval and numeric episode length disagree")
    start = intervals[video_key]["from_timestamp"]
    result = []
    for timestamp in timestamps:
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < -tolerance_s
                or timestamp > (length - 1) / fps + tolerance_s):
            raise ValueError("Video query falls outside its numeric episode")
        result.append(start + timestamp)
    return result
```

在公开 `src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py` 导入该函数；在每个相机的 `decode_video_frames(...)` **之前**调用：

```python
query_ts = video_query_timestamps(
    self.meta.info, self.meta.episodes[ep_idx], self.meta.video_keys,
    vid_key, query_ts, self.tolerance_s,
)
```

普通v2.1输入保持原行为；缺marker、缺任一相机、区间错误、跨episode时间必须报错。验收选非零且各相机不同的offset，并比较直接从源解码的帧。训练读取器若有“异常后随机换样本”的兜底，验收模式必须关闭，否则错误episode会被隐藏。统计扫描限制并行及在途episode数量，例如最多4个worker；不能同时提交全部episode导致峰值内存失控。

## 2. Awomo 数值与时间窗口辅助代码

在公开 Awomo checkout 根目录/模块路径内使用；唯一原生导入为公开 `eval_robocasa365/common.py`。保持该文件的 `state_vector`、`action12` 和 `ACTION_MASK` 不变。下列辅助代码不做归一化或正式训练。

```python
"""Numeric and temporal contract for the released RoboCasa policy.

Inputs use the published Human parquet order, not simulator engine order.
This module does not normalize, decode images, select training data, or train.
"""
from __future__ import annotations

import numpy as np

from eval_robocasa365.common import ACTION_MASK, REPLAN_STEPS, action12, state_vector


def _rows(value, dimensions: int, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if rows.ndim != 2 or rows.shape[1] != dimensions or len(rows) == 0:
        raise ValueError(f"{name} must be nonempty [N,{dimensions}]")
    if rows.dtype.kind not in "fi" or not np.isfinite(rows).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return rows


def human_action_to_model(raw_actions) -> np.ndarray:
    """Human12 -> model80; reject values that native action12 cannot preserve.

    Raw order: base xyz, zero torso, mode, eef xyz, eef rotation3, gripper.
    The gripper label becomes closedness in [0,1]; mode remains signed.
    No clipping or implicit repair is applied to demonstration targets.
    """
    raw = _rows(raw_actions, 12, "Human actions")
    if np.any(np.abs(raw) > 1) or np.any(raw[:, 3] != 0):
        raise ValueError("Actions outside [-1,1] or nonzero torso cannot roundtrip")
    if not np.isin(raw[:, [4, 11]], [-1, 1]).all():
        raise ValueError("Human mode and gripper labels must be exactly -1 or +1")
    target = np.zeros((len(raw), 80), dtype=np.float32)
    target[:, 7:13] = raw[:, 5:11]
    target[:, 16] = (raw[:, 11] + 1) / 2
    target[:, 58:60] = raw[:, :2]
    target[:, 63] = raw[:, 2]
    target[:, 76] = raw[:, 4]
    if np.any(target[:, ~ACTION_MASK] != 0):
        raise ValueError("Unexpected active target slot")
    if not np.array_equal(action12(target), raw.astype(np.float32)):
        raise ValueError("Released policy action decoder failed exact float32 roundtrip")
    return target


def human_state_to_model(raw_states) -> np.ndarray:
    """Reuse the unchanged released state encoder, including quaternion handling."""
    raw = _rows(raw_states, 16, "Human states")
    encoded = np.concatenate([state_vector(row) for row in raw], axis=0)
    if encoded.shape != (len(raw), 80) or not np.isfinite(encoded).all():
        raise ValueError("Invalid native state encoding")
    return encoded


def window_plan(start: int, episode_length: int, camera_offsets: dict[str, float],
                *, history_slots: int = 9, history_period: int = 40) -> dict:
    """Plan one complete probe window without crossing an episode boundary.

    Starts lie on the native 16-control-step replanning grid. History is spaced
    by 40 control steps, not by 40 video frames. Each camera has its own offset.
    A target image at start+16 is planned, not decoded or accepted by this call.
    Training tail padding and a training sampler remain separate work.
    """
    if type(start) is not int or type(episode_length) is not int:
        raise ValueError("Window indices must be integers")
    if start < 0 or start % REPLAN_STEPS or start + REPLAN_STEPS >= episode_length:
        raise ValueError("Require a full native-grid window and its next image")
    if history_slots != 9 or history_period != 40:
        raise ValueError("Require the frozen released RoboCasa history layout")
    if len(camera_offsets) != 3 or any(not np.isfinite(v) or v < 0 for v in camera_offsets.values()):
        raise ValueError("Require three finite nonnegative camera offsets")
    history = list(range(start - history_slots * history_period, start, history_period))
    valid = [step >= 0 for step in history]
    images = [step for step in history if step >= 0] + [start, start + REPLAN_STEPS]
    return {"start": start, "action_steps": list(range(start, start + REPLAN_STEPS)),
            "history_steps": history, "history_valid": valid,
            "current_step": start, "target_step": start + REPLAN_STEPS,
            "image_steps": images,
            "camera_timestamps": {key: [float(offset) + step / 20 for step in images]
                                  for key, offset in camera_offsets.items()}}
```

Human state16原顺序：base_pos3、base_quaternion4、eef_pos3、eef_quaternion4、gripper_qpos2；四元数为xyzw，继续调用原生编码器，不另造坐标变换。模拟器执行顺序需再用公开 `to_engine_order`，不能把Human12直接当engine12。

## 3. 像素、请求压缩与原生processor

所有源视频按每相机 `from_timestamp + step/20` 解码；camera顺序固定left、right、eye_in_hand，输出uint8 `[T,3,256,256,3]`。原生压缩是**每次请求**独立编码/解码：step0一帧，后续请求为replan边界前12/8/4/0步的四帧。不能一次压缩全部历史得到“近似相同”像素。下面函数的 `decode` 接收控制步列表；`compress` 和 `compose` 分别调用公开策略的 `h264_roundtrip` 和 `compose_canvas`，不使用替代JPEG/resize。

```python
"""Assemble Human image windows with the released per-request codec semantics."""
from __future__ import annotations

import numpy as np


def request_packets(steps: list[int]) -> dict[int, list[int]]:
    """Recover the original request containing each requested control step.

    Step0 is encoded alone. A later request contains four frames ending at its
    16-step replanning boundary. Compressing a whole history together is different.
    """
    if not steps or any(type(step) is not int or step < 0 or step % 4 for step in steps):
        raise ValueError('Need nonnegative integer control steps on the4-step frame grid')
    packets = {}
    for step in steps:
        replan = (step + 15) // 16
        packets[replan] = [0] if replan == 0 else list(range(replan * 16 - 12, replan * 16 + 1, 4))
    return dict(sorted(packets.items()))


def assemble_window(plan: dict, decode, compress, compose, *, size=(320, 224)) -> dict:
    """Decode every frame of each needed request before native compression.

    Callbacks permit the real installed policy functions to remain unchanged.
    Returned video has current and future target; history contains past frames.
    This does not tokenize text, encode latents, or normalize action/state data.
    """
    steps = plan['image_steps']
    packets = request_packets(steps)
    if max(max(value) for value in packets.values()) > plan['target_step']:
        raise ValueError('Packet contains an observation beyond the target boundary')
    if size != (320, 224) or len(plan['history_valid']) != 9:
        raise ValueError('Require the frozen RoboCasa image/history configuration')
    frames = {}
    for packet in packets.values():
        raw = np.asarray(decode(packet))
        if raw.dtype != np.uint8 or raw.shape != (len(packet), 3, 256, 256, 3):
            raise ValueError('Decoder must return native uint8 three-camera frames')
        compressed = np.asarray(compress(raw))
        if compressed.dtype != np.uint8 or compressed.shape != raw.shape:
            raise ValueError('Native codec changed image shape or dtype')
        frames.update(zip(packet, compressed))
    image = np.asarray(compose(np.stack([frames[step] for step in steps]), size))
    if image.dtype != np.float32 or image.shape != (len(steps), *size, 3):
        raise ValueError('Native canvas returned the wrong shape or dtype')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Native canvas returned invalid pixels')
    history = np.zeros((9, *size, 3), dtype=np.float32)
    valid = np.asarray(plan['history_valid'], dtype=bool)
    if int(valid.sum()) + 2 != len(image):
        raise ValueError('Image/history index mismatch')
    history[valid] = image[:-2]
    return {'video': image[-2:].copy(), 'history': history, 'history_valid': valid,
            'packets': packets}
```

辅助sample验证如下。当前帧与未来图像放在 `video[0]`、`video[1]`；只有有效history和当前帧进入prompt。完整processor前置不等于Qwen/VAE/DiT前向。

```python
"""CPU sample boundary shared by preparation and future training probes."""
import numpy as np
from eval_robocasa365.common import ACTION_MASK


def validate_sample(sample):
    shapes={'video':(2,320,224,3),'history':(9,320,224,3),'history_valid':(9,),
            'state':(1,80),'action':(16,80)}
    for name,shape in shapes.items():
        value=sample[name]
        expected=np.bool_ if name=='history_valid' else np.float32
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=expected:
            raise ValueError(f'Invalid training sample shape/dtype: {name}')
        if not np.isfinite(value).all():raise ValueError(f'Non-finite sample: {name}')
    for name in ['video','history']:
        if sample[name].min()<0 or sample[name].max()>1:raise ValueError('Pixels outside native0..1 range')
    if np.any(sample['history'][~sample['history_valid']]!=0):raise ValueError('Invalid history slots must stay zero')
    if np.any(sample['action'][:,~ACTION_MASK]!=0):raise ValueError('Action target has inactive nonzero slots')


def prompt_images(sample):
    """Condition only on valid history and current image, never future target."""
    validate_sample(sample)
    return [*sample['history'][sample['history_valid']],sample['video'][0]]
```

原生参照：公开 `eval_robocasa365/awomo_05i/policy.py` 的history/codec/布局，`awomo/awomo_05i/prompt.py` 的 `PromptEncoder`。固定Qwen本地路径加载 `AutoProcessor(..., local_files_only=True)`，沿用原 `LAYOUT_SENTENCE`；`PromptEncoder.encode(instruction, video[0], list(history[history_valid]))`。

独立对照用 `build_messages` 生成相同文本，再 `apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=False)`；以有效history+当前RGB图像调用标准processor，`do_resize=False, do_rescale=False, return_tensors="pt"`。逐张量检查input_ids、attention_mask、pixel_values、image_grid_thw完全相同，若返回mm_token_type_ids也比较；condition_pos必须位于layout/task文字token，不包含image_pad。保存所有原始sample数组和完整processor数组的shape/dtype/content摘要。

用于重建的具体训练样本：

| 组/任务 | source_episode_index | start | 有效history |
|---|---:|---:|---:|
| atomic_seen / OpenCabinet | 499 | 368 | 9 |
| composite_seen / StoreLeftoversInBowl | 502 | 384 | 9 |
| composite_unseen / CategorizeCondiments | 512 | 368 | 9 |
| atomic_seen / OpenCabinet | 499 | 0 | 0 |
| atomic_seen / OpenCabinet | 499 | 176 | 4 |
| composite_seen / StoreLeftoversInBowl | 502 | 0 | 0 |
| composite_unseen / CategorizeCondiments | 512 | 0 | 0 |

先从任务prefix和source_episode_index定位原metadata，确认这些episode在train内，然后按本页原始parquet/视频规则重建。前三个完整history的历史processor检查已验；后四个仅像素/序列检查已验，新processor请求在冻结时尚未验收。不要把“代码可执行”当作这些原始payload已下载。

## 4. 全量窗口表、分布式顺序与恢复

按 `(group, task)` 字典序、任务内source_episode_index升序形成episode catalog，每行ordinal等于全局行号。字段至少包括ordinal/group/task/episode_index/length/split、相对numeric_path及bytes/SHA256、source_prefix、完整source_identity的canonical SHA256、instructions、每相机时间区间。source_identity采用 `json.dumps(..., sort_keys=True, separators=(',', ':'), ensure_ascii=False)` 后UTF-8求SHA256。

下面代码完整给出窗口、尾部统计、epoch顺序及cursor状态。保存episode catalog与两份int64窗口数组；dataset绑定必须涵盖合同、catalog和两张窗口表内容。示例catalog写 `json.dump(..., ensure_ascii=False, indent=2)`，窗口写 `np.save(..., allow_pickle=False)`，再把三个文件SHA和INDEX_CONTRACT按canonical JSON求SHA。历史dataset SHA见 [训练合同](DATA_AND_TRAINING.md)；重新写parquet/序列化可能改变文件摘要，应比较源身份/数组语义并产生新绑定，不能硬填历史hash使检查通过。

```python
"""Candidate full-chunk indexing and deterministic sampler state for Human50.

This is an explicit data-engineering contract, not an official SFT recipe.
Every target image must exist at start+16. Short tails are counted, never padded.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np

INDEX_CONTRACT = {
    'version': 1, 'control_hz': 20, 'stride': 16, 'action_horizon': 16,
    'future_image_offset': 16, 'tail_policy': 'drop_incomplete_chunk_or_missing_future_image',
    'window_columns': ['episode_ordinal', 'start_control_step'], 'window_dtype': 'int64',
    'partition': 'unchanged_whole_episode_train_validation',
    'ordering': 'group_task_episode_then_start',
    'sampling': 'uniform_over_train_windows_without_replacement_each_epoch',
    'shuffle': 'numpy_PCG64_SeedSequence_seed_epoch',
    'rank_assignment': 'global_batches_then_rank_then_local_batch',
    'distributed_tail': 'drop_incomplete_global_batch_after_shuffle',
    'validation': 'all_full_windows_in_canonical_order_without_shuffle_or_rank_tail_drop',
    'formal_recipe_selected': False,
}


def window_starts(length):
    if type(length) is not int or length < 1:
        raise ValueError('Episode length must be a positive integer')
    return np.arange(max(0, (length-1)//16), dtype=np.int64)*16


def make_tables(episodes):
    identities=set();parts={'train':[], 'validation':[]};tail={'train':0,'validation':0}
    empty={'train':0,'validation':0}
    for ordinal,episode in enumerate(episodes):
        identity=(episode['group'],episode['task'],episode['episode_index'])
        if identity in identities:
            raise ValueError('Episode occurs more than once or crosses partitions')
        identities.add(identity)
        split=episode['split']
        if split not in parts or episode['ordinal']!=ordinal:
            raise ValueError('Unknown partition or noncanonical ordinal')
        starts=window_starts(episode['length'])
        block=np.empty((len(starts),2),dtype=np.int64)
        block[:,0]=ordinal;block[:,1]=starts;parts[split].append(block)
        tail[split]+=episode['length']-16*len(starts)
        empty[split]+=int(len(starts)==0)
    tables={key:np.concatenate(value,axis=0) if value else np.empty((0,2),dtype=np.int64)
            for key,value in parts.items()}
    return tables, {'non_supervised_tail_action_rows':tail,'episodes_without_full_window':empty}


def array_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def epoch_batches(window_count, seed, epoch, world_size, local_batch):
    for value in (window_count,seed,epoch,world_size,local_batch):
        if type(value) is not int:
            raise ValueError('Sampler indices must be integers')
    if min(window_count,world_size,local_batch)<1 or min(seed,epoch)<0:
        raise ValueError('Invalid sampler dimensions or seed')
    global_batch=world_size*local_batch
    usable=(window_count//global_batch)*global_batch
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,epoch])))
    order=rng.permutation(window_count).astype(np.int64,copy=False)
    return order[:usable].reshape(-1,world_size,local_batch),order[usable:]


def cursor_state(dataset_sha256, window_count, seed, epoch, world_size, local_batch, next_global_batch):
    if len(dataset_sha256)!=64 or any(x not in '0123456789abcdef' for x in dataset_sha256):
        raise ValueError('Dataset binding must be a SHA256')
    batches,dropped=epoch_batches(window_count,seed,epoch,world_size,local_batch)
    if type(next_global_batch) is not int or not 0<=next_global_batch<=len(batches):
        raise ValueError('Cursor lies outside this exact epoch')
    state={'version':1,'dataset_sha256':dataset_sha256,'window_count':window_count,
            'seed':seed,'epoch':epoch,'world_size':world_size,'local_batch':local_batch,
            'next_global_batch':next_global_batch,'order_sha256':array_sha(batches),
            'dropped_sha256':array_sha(dropped)}
    state['state_sha256']=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return state


def restore_cursor(saved, dataset_sha256, *, world_size, local_batch):
    """Reject changed data/rank geometry rather than silently repeating/skipping data."""
    if saved['dataset_sha256']!=dataset_sha256 or saved['world_size']!=world_size or saved['local_batch']!=local_batch:
        raise ValueError('Changed dataset or distributed batch geometry')
    expected=cursor_state(dataset_sha256,saved['window_count'],saved['seed'],saved['epoch'],
                          world_size,local_batch,saved['next_global_batch'])
    if saved!=expected:
        raise ValueError('Sampler state or permutation was altered')
    batches,_=epoch_batches(saved['window_count'],saved['seed'],saved['epoch'],world_size,local_batch)
    return batches[saved['next_global_batch']:]
```

预期train910,723 / validation10,720窗口；尾部action rows分别212,276 / 2,535；无0完整窗口episode。对world1/2/8/16各自验证rank集合不重叠、batch与丢弃尾部合起来恰好覆盖全部train窗口；序列化cursor37（若不足37取终点）后恢复与原剩余序列逐项相同。改变数据、world_size、local_batch必须拒绝旧cursor。CPU rank模拟通过不表示HCCL/DDP训练通过。

## 5. 训练诊断需要保留的计算与恢复语义

Loop固定配方及原两卡恢复误差见 [训练合同](DATA_AND_TRAINING.md)。使用公开原生trainer、发布normalizer和真实Human读取路径完成3个更新；按同一完整scheduler horizon运行连续与恢复分支，记录全部参数/optimizer/RNG/cursor。只比较打印到四位小数的loss无法证明恢复相同。

Awomo从发布FP32 DiT master开始，训练全部DiT及proprio encoder，冻结Qwen+发布LoRA+VAE，配方详见 [训练合同](DATA_AND_TRAINING.md)。计算必须为：

```python
# clean_image_latents、clean_actions、两个独立的噪声张量均保留作跨设备fixture。
sigma = 0.5
noisy_image = (1 - sigma) * clean_image_latents + sigma * image_noise
noisy_action = (1 - sigma) * clean_actions + sigma * action_noise
image_velocity = image_noise - clean_image_latents
action_velocity = action_noise - clean_actions
# 直接调用同一DiT的可微forward；不要调用带no_grad的策略velocity方法。
# timestep = sigma * 1000；沿用原生tokens/conditioning/state/mask。
image_loss = (image_prediction.float() - image_velocity.float()).square().mean()
action_loss = (action_prediction.float()[..., action_mask]
               - action_velocity.float()[..., action_mask]).square().mean()
loss = image_loss + action_loss  # action_mask严格只有11个有效槽
```

使用前三个完整历史样本依次执行update1/2/3。update1保存；连续2/3后，再从独立新进程加载update1重做2/3，总计5次真实更新。保存完整model、optimizer、CPU与每rank设备RNG、completed_updates=1、next_sample=1和模型/输入/配方绑定（如使用scheduler/scaler也保存）。NPU入口存NPU RNG，不能把CUDA RNG伪装成NPU状态。恢复分支必须读取相同下个样本、噪声与学习率；比较所有参数、Adam状态、RNG、cursor及loss，有限性与frozen参数不变。原GPU同设备合同是exact equality；尚未GPU验收，NPU若需要容差必须独立报告，不能宣称原合同通过。

## 6. 直接重建 ECR dev1000 / test2500 请求

在本指南仓库根执行下面Python代码；先将环境变量 `WAM_SCENES_OUT` 设为一个**尚不存在**的本地输出目录。只读取公开任务表，不下载、不调用模拟器。生成器检查完整输出bytes/SHA256，然后写文件。

```python
import hashlib
import json
import os
from pathlib import Path

tasks = json.loads(Path('wam/pins/target50-tasks.json').read_text())['tasks']
assert [t['target50_index'] for t in tasks] == list(range(50))
expected = {
    'dev': (20, 510000000, 367380, '4f4af1f0a1baa628b4f10ef9d906459d2bd955c4cbb573778be90fc6c4237cee'),
    'test': (50, 520000000, 924200, 'c312526053e6042b0d9963a91ebd5dea7c9cec12d21b20f16daba7c86ab53d3e'),
}
output = Path(os.environ['WAM_SCENES_OUT'])
output.mkdir(parents=True, exist_ok=False)
for phase, (count, base, size, digest) in expected.items():
    rows = []
    for task in tasks:
        index = task['target50_index']
        for slot in range(count):
            seed = base + index * 1000 + slot
            row = {k: task[k] for k in ('task', 'group', 'target50_index', 'horizon', 'historical10')}
            row.update(phase=phase, split='target', slot=slot, attempt=0,
                       scene_id=f'target50-v1/{phase}/{index:02d}/{slot:03d}',
                       requested_seed=seed, actual_seed=seed,
                       initial_state_sha256=None, layout_id=None, style_id=None,
                       rollout_entry_smoke=False, runtime_scene_verified=False)
            rows.append(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
    payload = ''.join(rows).encode('utf-8')
    assert len(rows) == 50 * count and len(payload) == size
    assert hashlib.sha256(payload).hexdigest() == digest
    with (output / f'{phase}.jsonl').open('xb') as stream:
        stream.write(payload)
    print(phase, len(rows), digest)
```

**`actual_seed` 在未运行的请求中只是预填请求值，不是实测。** `runtime_scene_verified=false`、空initial hash必须保留，实际运行后另记录observed身份。请求不证明物理初态独立，也不证明与Human无交叠。

## 7. ECR evaluator 与官方包装器的实现合同

官方两模型的seed/horizon/native reset/停止规则完整列在 [评测合同](EVALUATION.md) 和 [100条官方协议](pins/official-pretrain-protocol.json)。共同ECR入口按下列规格实现：

1. 每个不可变请求先用新进程原生reset取得zero-action参考，再用独立policy worker重建同一请求；策略推理前比较XML、episode metadata、instruction、layout/style、MuJoCo integration state、state14/state16、三相机原始图像，保存dtype/shape/content hash和原始payload。
2. 两模型使用相同ECR环境/资产与请求，保留各自原生动作封装、normalizer、重规划和历史。Loop预测64/执行48/quality5；Awomo执行16、每4步送帧、9历史/40步间隔、启用原生压缩。
3. Awomo共同target策略seed=`(7+50*target50_index+slot)*10000+replan_index`，与环境大seed分开；Loop保留原policy_seed未指定语义。固定noise只用于独立数值对照，不能静默改正式评测。
4. 保存每个请求、返回chunk、实际执行动作、控制步、停止原因和完整horizon。原生早停规则依各模型协议；64步probe只能是probe，不能作为900/3000步任务的正式结论。
5. 环境关闭失败、GL异常、counter不符、错误模型/scene身份、部分输出都作为无效或错误记录，不变成success/failure分数；同一身份至多保留一个有效正式结果，重试保留attempt链。
6. 只有50类实际环境覆盖、同场景包装、长历史和完整dev前置验收后才放行test2500/model。接手快照时这些尚未完成；本页生成请求通过不提供正式test启动结论。

Loop CLI按原脚本 `__main__` 启动，避免Hydra入口被普通import改变。分离NPU policy与GPU evaluator时，参数化网络地址，保留HTTP/msgpack原协议和会话reset/RNG；记录网络与模型耗时。每份run结果必须绑定本机公开base、适配diff/commit、环境、模型、输入和实际完整命令。

## 8. 原生 metadata 与 JSON 参考的严格比较

2026-11-14 GPU2 两模型的原版共同场景策略诊断中，NavigateKitchen 均完成 64 步；OpenDrawer 均在首次推理前被 metadata 比较阻止。离线检查确认，两者的 XML、integration/state16/state14、三相机初始图像，以及**保存后的完整 metadata** 都与参考完全相同。这次失败不能归因于上一节之外的 Counter 区域顺序问题。

公开 RoboCasa commit `456174f62b89b8fca99eaaf33949c29fec9cfc2a` 的 `OpenDrawer._get_obj_cfgs` 使用 tuple，例如 `obj_groups=("tool", "utensil")`、`max_size=(None,None,0.10)`；`Kitchen.get_ep_meta` 的复制函数保留这些 tuple。JSON 写出后再读取会得到 list。直接用内存原生 dict 与读取后的 dict 比较，会把相同持久化内容误判为不同。已用这两个固定上游方法的原始代码在 CPU 测试重现；原失败运行未采集内存 tuple 路径，新的现场验证应保存它们。

比较两侧完整的 JSON 表示，保持所有字段、数组顺序、标量类型与数值精度。不得通过删除字段、排序数组、四舍五入或增加容差让比较通过：

```python
import json

def serialized_metadata(value):
    def check_keys(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Metadata JSON object keys must be strings")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                check_keys(child)
    check_keys(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

# actual_metadata: 原生 get_ep_meta() 返回值
# reference_metadata: 从固定参考 JSON 读取的完整值
# assert serialized_metadata(actual_metadata) == serialized_metadata(reference_metadata)
```

只有 metadata 的表示方式按已有存储协议统一；XML、物理状态、相机图像、scene request、instruction、layout/style 仍各自严格检查。保留原失败记录，使用新代码版本与新运行身份重新验收。修复已有 17 项本地 CPU 合同测试通过，包括固定原生方法重现、标量类型/数组顺序/微小数值变化仍拒绝、非有限值和非法键拒绝；两模型修订版的现场 CPU 前置也已分别通过 17 项测试、固定源码/环境与原生客户端检查；修订版 GPU 两模型各两个场景、每场景 64 步已验收，所有初态数组与完整 metadata 精确匹配；OpenDrawer 实际记录 15 条 tuple 路径。完整 horizon 仍待验收。此修复不应用稳定 Counter 候选，也不改变原 v1 场景身份或策略语义。

## 9. CPU 与渲染初态差异的独立诊断规格

**状态（2026-11-14，已更新）：** 原实现的 10 项本地隔离测试通过；首轮两个 CPU 现场准备的同 10 项测试也通过，但均在第一个 control 进程的模型大小检查失败，尚无初态诊断通过结果。第 9.1–9.4 节保留原合同，第 9.5 节记录此次失败与修订。替身测试不能证明真实 MuJoCo 调用、渲染或 Ascend 兼容；原 CPU/渲染物理差异事实仍见总入口第 7.4 节。

本节给出接手者独立实现所需合同，不要求取得原运行端的工程代码或任务回执。仍固定本指南的公开 RoboCasa/robosuite commits、MuJoCo 3.3.1 和相同资产。候选协议为 `ecr-target50-stable-counter-v2-candidate`，只应用总入口第 7.3 节的稳定去重；不修改安装文件，不与原 v1 或官方 pretrain 合并计分。

### 9.1 问题、固定输入与假设

只使用开发集 slot 0 的两个输入：

| task index | 任务 | 原 scene_id | environment seed |
|---:|---|---|---:|
| 1 | CloseFridge | `target50-v1/dev/01/000` | 510001000 |
| 27 | SearingMeat | `target50-v1/dev/27/000` | 510027000 |

实际身份为 `(候选 protocol_id, 原 scene_id)`；完整请求按第 6 节生成，不改 task、layout/style 选择、seed 或原生 reset 次序。

一个待检验的线索是：固定公开 [robosuite binding_utils.py](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/utils/binding_utils.py) 中，`MjRenderContext.__init__` 在 `gl_ctx.make_current()` 后调用 `sim.forward()`；`MjSim.forward()` 调用 MuJoCo 的 `mj_forward`。关闭 offscreen renderer 的 CPU 路径可能没有这次调用。**这只是源码线索，不是已证明的根因。** 需要定位最早差异、比较模型与状态，再设计后续因果实验；本轮不主动增删 forward 调用。

### 9.2 四进程对照与原生执行

对每个输入、每条路径分别运行四个全新进程：`control/0`、`control/1`、`traced/0`、`traced/1`。CPU 路径仅关闭 `has_offscreen_renderer` 与 `use_camera_obs`，禁止创建渲染上下文或调用 render；渲染路径保留原生三相机及实际分配 GPU 的 EGL 映射。两条路径保留原生构造、reset、随机种子和 reset 内部 settling；不执行策略或额外控制动作。

四个进程都在原 worker 唯一一次最终 `mj_getState(..., mjSTATE_INTEGRATION)` 读取处保存相同最终状态和编译模型。先调用原 getter 一次，保留原输出缓冲、参数和返回对象；额外只读快照须使用保存的原 getter，不能递归调用被包裹的入口。拒绝额外的最终读取或不同掩码。

仅 `traced` 进程观察下列原方法的调用前/后边界：`MujocoEnv.__init__`、`MujocoEnv.reset`、`MujocoEnv._get_observations`、`MjSim.forward`、`MjSim.render`、`MjRenderContext.__init__`。方法本体每次恰好调用一次，参数、返回值和原异常保持不变。构造前 sim 尚不存在时，只记录 `available=false`；不为补观测创建 sim。异常路径尽量保留最后状态及观察器错误，随后原样抛出原异常；正常或异常退出均恢复钩子。

观察器不得调用额外的 forward、reset、step、renderer query、随机抽样或状态恢复。它有文件和读取开销；是否影响被测结果由四进程最终状态对照验证，不能仅凭实现意图宣称无干扰。

### 9.3 状态、模型与原始产物

每个可用边界直接读取完整 `mjSTATE_INTEGRATION`。从 `mjtState` 中枚举落在此掩码内的正数单 bit 成员，拒绝重复 bit，按 bit 数值升序排列，要求 bit 之和等于完整掩码。每一项先用 `mj_stateSize` 分配 float64 缓冲，再用原 `mj_getState` 读取；各项拼接必须与完整向量逐元素精确相等且有限。这样可将差异归到具体状态分量，不能只比较 qpos/qvel 或末端状态。

同时只读复制 `qacc`、`qacc_warmstart`、`qfrc_bias`、`qfrc_constraint`、`qfrc_passive`、`efc_force`、`xpos`、`xquat`、`xmat` 中实际存在的字段，并检查类型与有限性。这些派生量帮助定位 forward 边界，不替代完整 integration state。

最终状态另外保存完整编译模型：通过 `mj_sizeModel(model)` 分配 uint8 缓冲，以 MuJoCo 3.3.1 的 `mj_saveModel(model, None, buffer)` 写出 MJB。再保存模型暴露的数值标量/数组，以及 `opt`、`stat`、`vis` 下可访问的数值字段，明确列出未覆盖属性。数值字段检查与 MJB 原件相互补充：字段遍历不等于全模型覆盖，MJB 字节差异也不能单独证明物理模型不同。

每进程至少保留：

- 按发生顺序追加的 `events.jsonl`，每条含序号、方法、before/after/exception、sim 是否可用和快照引用；每个可用事件一份原始 NPZ。
- 最终状态 NPZ、模型数值 NPZ、完整模型 MJB、数值属性遗漏清单、观察器异常和最终捕获次数。
- 原 worker 的 XML、完整序列化 metadata、request、instruction、layout/style、integration/state16/state14；渲染路径另外保留三相机初始图像。
- 每文件 bytes/SHA256，每数组 dtype、shape 和按 C 顺序完整 bytes 的 SHA256，以及实际软件、公开源码/本机适配版本、GPU/EGL 身份、退出状态与 stderr。

### 9.4 验收、预算与下一步

先独立执行两个 CPU 准备项，每项 0 GPU、4 CPU、32 GiB RAM、30 分钟，保留产物最多 10 GiB。每个输入的四次最终 integration、state16/state14、派生状态及模型数值数组须完全一致；XML、metadata 和请求身份也须一致。最终物理状态还须与本节点之前独立生成并冻结的 CPU 参考精确匹配。接手端没有参考 payload 时须先独立生成、保存和冻结；不能从失败运行反填参考或只依赖本指南中的历史摘要。

两个 CPU 项的完整产物经独立验收且实现版本一致，才执行渲染诊断：1 GPU、4 CPU、32 GiB RAM、60 分钟，两个输入共八个新进程，保留产物最多 18 GiB。单进程最长 300 秒、最多 8,192 个事件、观察器产物最多 2 GiB、单份编译模型最多 512 MiB；写入前检查预算，超过即失败，保留已有证据。

渲染路径同样要求每个输入四次最终状态与数值模型精确一致。CPU 与渲染之间的差异作为诊断输出保存，不将差异本身当作通过或放宽验收。对齐原方法调用序列，逐状态分量寻找最早不同的边界，并检查模型数值是否已变化；有事件缺失、观察器异常或路径内不一致时，先判诊断无效或未完成。

本地 10 项测试覆盖状态 bit 完整性/顺序/有限性、最终 getter 输入输出与恢复、重复最终读取拒绝、原方法单次调用与 RNG 不变、原异常保留、模型微小数值差异识别、写前预算和 CPU 依赖记录约束。现场还须重新验证真实 API、资产、初态和进程退出。该诊断贡献 **0 正式 episode、0 策略评测、0 SFT 更新**；不得增加容差、额外 reset 或覆盖旧参考来解除原来的物理初态阻塞。

### 9.5 首轮 CPU 失败与先读取旧快照的修订

两个输入均在首个 `control/0` 的 `0 < mj_sizeModel(model) <= 536870912` 检查停止，完成 worker 数为 0。错误文字只有 `Compiled model budget exceeded`，未保存实际返回值，因此**不能区分非正数与大于 512 MiB，更不能据此确认原 CPU/render 差异的原因**。模型 MJB 尚未生成。每项已有 11 份部分产物，分别 205,019,991 与 147,412,193 bytes；节点端完整文件清单和 SHA256 已核验，小回执/日志与 27 份源码前后状态已审计，完整 NPZ 尚未回传至研究端。原件继续保留，不能把清单当逐数组验收。

其中已保存的模型数值 NPZ 为：

| 输入 | NPZ bytes | SHA256 |
|---|---:|---|
| dev01 / slot0 | 204,988,824 | `1c1fb65b778631224534f61c17190f7d6afe668f650d7255ff44c97668201bbe` |
| dev27 / slot0 | 147,383,783 | `9bf29d87c475036b21c3fe3b16838f0265de0e715328cca0b421255dd958e616` |

后继首先分析这两份旧文件，不重新构造场景。接手者可对自己的失败快照实现同样的只读工具：先流式核对完整压缩文件大小和 SHA256；使用 ZIP 目录逐项打开 `.npy`，只读取 NPY v1/v2 头中的 dtype、shape、存储次序，再检查头大小加数组字节数恰好等于 ZIP 声明的未压缩大小。拒绝重复成员、路径分隔符、非 `model.` 字段、object dtype、不合法 shape 和过大 header。仅 shape 为标量且数据不超过 16 bytes 时读取实际数值；大数组保持压缩，禁止 `np.load` 后遍历并实例化全部数组。

输出全部字段的压缩/未压缩大小、dtype/shape、可用标量（包括 `nbuffer`）及最大的 20 个字段，再复核输入未改变且完整 SHA256 相同。成员最多 4,096 个、单 header 最多 10 KB、结果 JSON 最多 4 MiB；每个输入独立 0 GPU、1 CPU、2 GiB RAM、10 分钟，保留证据最多 16 MiB。工具不导入模拟器或模型，不做 reset、forward、渲染或 MJB 保存。读取旧快照需要真实 payload；上表摘要本身不能替代文件，也不要求接手者取得额外的工程仓库。

**解释限制：** 暴露字段可能包含别名，数组字节总和不是 MJB 大小。原进程没有记录的 `mj_sizeModel` 数值仍未知；不能从压缩文件大小反推该值。公开 [MuJoCo 3.3.1 实现](https://github.com/google-deepmind/mujoco/blob/3.3.1/src/engine/engine_io.c) 使用 `int` 累计并返回此大小，这是额外的检查线索，不构成已发生溢出的证据。

完整观察器另已修正日志顺序：原 getter 返回后，先读取实际 `mj_sizeModel`，在任何大型数值快照复制前写 `model-budget.json`，包含实际有符号数值、`limit_bytes=536870912`、可用的 `model_nbuffer`、`full_model_snapshot_started=false`，并将状态区分为 `invalid_nonpositive`、`over_budget`、`within_budget`。只有最后一种继续原快照路径，上限没有放宽。修订观察器的 11 项本地测试及旧快照工具的 7 项本地测试通过；后者尚无现场测量结果。旧完整任务不自动重试，原 GPU 条件任务仍等待新的完整 CPU 验收；这两个小型测量任务不满足该条件。

## 10. Awomo 原生九槽历史的独立准备

2026-11-14 新增的是代码与预注册合同。三输入初态 CPU 检查已有调度完成记录，完整原件尚未验收；历史组装的首轮现场运行在会话重置检查失败，已定位并修正检查器形状错误，修订版 15 项本地合同测试通过，新 CPU 运行已有调度完成记录（退出码 0），但完整 receipt、日志和五份原始边界仍待回传验收。真实模型长回合入口与服务器观察器现已完成代码，19 项本地测试通过；新增匹配 CPU 检查已提交，完整验收及 GPU 实测仍未完成。本节不增加正式评测、训练或 Ascend 完成量。此前共同 ECR 的 Awomo 两个完整 dev 回合最多运行 201 步，尚不能证明该入口覆盖了九个历史槽。

### 固定输入与终止规则

从第 6 节冻结的 dev1000 清单选择 NavigateKitchen，`target50_index=4`、slot 为 1、2、3。对应 ID 是 `target50-v1/dev/04/001`、`002`、`003`，环境 seed 分别为 `510004001`、`510004002`、`510004003`；每局 horizon 450，最多 1,350 个环境步。选择在运行前固定，只使用原 v1 初始化，不应用第 9 节的稳定 Counter 候选，也不读取 test 成绩选场景。

后续真实共同 ECR 回合沿用已声明的 success、terminated、truncated 或 horizon 终止；成功后不继续动作以凑历史长度。如果三局全部提前结束，就报告历史覆盖不足，不自动换 seed 或追加场景。

九槽索引为 `current − 360, current − 320, …, current − 40`，非负槽才有效；当前帧另列。Awomo 每 16 控制步重规划，因此完整九槽首次可能出现于 **368 步**，对应历史帧 `[8,48,88,128,168,208,248,288,328]` 和当前帧 368；352 步仍只有八槽。不能仅凭回合超过 360 步宣布通过。

### 两项相互独立的 CPU 准备

**初态对照：** 每个固定输入各用两个全新进程，合计六次初始化。按第 9.2 节的 CPU 原生构造/reset/seed/reset 顺序，只关闭 offscreen renderer 与 camera observations；禁止渲染、策略、控制动作和 Counter 替换。每对比较完整 XML、完整 JSON metadata、layout/style、环境参数、integration mask，以及 integration/state16/state14 的 dtype、shape 和全部字节。保存每个进程的原始 XML、metadata、NPZ、日志与比较记录。预算为 0 GPU、2 CPU、32 GiB RAM、30 分钟，每个子进程最多 120 秒、证据最多 2 GiB。该检查只证明选定输入的 CPU 重复性，不能证明与渲染初态相同。

**原生历史组装：** 使用[公开源码清单](pins/source-pins.json)中固定的原版 Awomo 策略版本，直接调用其 `parse_request`、`h264_roundtrip`、`RoboCasaPolicy._update`、`_sample` 和 `compose_canvas`，保持各函数原内容。读取固定 checkpoint 中 738 bytes 的 `awomo_config.json`（Git blob SHA1 `5f6d1ecf06367c54bcba406ef8bf1fece4d77864`），确认 9 槽、周期 40、动作长度 16，并使用其实际 video_size。这里只构造持有 video_size 的轻量对象；prompt 替身仅记录输入，不加载权重、processor、encoder，也不调用 predict、模型 encode 或采样动作。

输入是可确定性重建的测试图像，不是仿真观测。每个相机为 256×256 RGB，像素值为 `(3*x + 5*y + 13*step + 67*camera + 41*channel) % 256`，camera/channel 从 0 编号，dtype 为 uint8。请求在 `0,16,…,448` 步产生，共 29 次；首请求只有 step0，此后每次含 current−12、−8、−4、0 四个新观测。每个请求单独经过原生 H264 压缩/解压，再送入原生缓存。不能将全部历史一次压缩。

测试 state16 为 float32 全零向量，将从 0 编号的第 6、13 元素设为 1，以表示两个单位四元数；session 固定为 `cpu-history-fixture-v1`。instruction 使用两端各两个空格的 `synthetic history fixture`，以检查原生去空白行为。请求 seed 为 `2080000+replan_index`，本 CPU 检查没有动作采样，不能据此声明策略 RNG 已验证。

另存一份独立的压缩后帧时间线，逐次核对：

- 缓存恰好保留 `max(0,current−360)` 到 current 的所有 4 步间隔帧；每帧完整 dtype/shape/字节相同。
- 在 `compose_canvas` 外只包一层观察，原函数仅调用一次并返回同一个结果对象。记录实际选入的缓存帧与输出画布，按上述独立索引核对来源和顺序。
- `history_valid` 是精确的九元素 bool 掩码；无效槽全部为 float32 零，合法历史槽、当前 video 与实际画布逐数组相同。state 与原生 state_vector 一致。
- prompt 边界实际收到去首尾空白的 instruction、当前画布及全部合法历史画布，顺序不变。此检查不等于真实 processor 或 VLM 验收。
- 末尾用新 session 的 replan0 重置，确认旧帧清空、缓存持有输入副本；旧 session 的后续 replan 必须被原生检查拒绝。观察钩子在异常后也恢复。

保留 29 份逐请求元数据与数组摘要，在 step 0、160、352、368、448 另存原始请求、压缩后请求、实际选中缓存、画布、video/history/mask/state 的 NPZ。预期全九槽请求恰好为 368、384、400、416、432、448。预算为 0 GPU、2 CPU、16 GiB RAM、20 分钟、最多 2 GiB 证据；意外 CUDA 初始化立即失败，保留错误与部分产物。

两项准备分别验收，均不自动放行 GPU。后续真实长回合还必须保存服务器实际缓存和 `_sample` 输入，并与客户端 session/replan、原始观测来源和请求 seed 对齐。没有真实模型与原生终止下的这些证据，就继续标记九槽实机覆盖未完成。本节已完整写出重建方法，不需要取得原运行端的工程仓库或调度记录。

### 首轮现场失败与修订

首轮历史组装的 12 项现场合同测试通过，随后 producer 以错误 `Array dtype/shape/bytes differ` 退出。限定只读诊断取得完整原 JSON 和 traceback，原文件 9,170 bytes、读取前后 SHA256 相同。失败位于会话重置后的缓存复制检查：`ep['frames'][0]` 是 `[3,256,256,3]` 的单帧，而检查器拿它与 `fixture_frames([0])` 产生的 `[1,3,256,256,3]` 整个请求序列比较。该错误不能据此归因于原生模型或缓存实现。

修订先保存 `expected_frame = reset_frames[0].copy()`，再将输入序列清零，检查缓存单帧仍与这个独立副本的 dtype、shape 和所有字节相同。新增回归分别验证正确单帧比较可通过、缓存引用输入而未复制时失败、旧帧或旧 session 未隔离时失败；共 15 项本地测试通过。原生 `_update`、`_sample`、H264、canvas、29 请求时间线和严格数组比较规则均不变。

原失败回执列出 29 份请求记录及 5 份边界 NPZ，全部 38 份产物合计 72,559,067 bytes；这些完整原件仍待回传与独立验收，不能把清单当作通过结果。修订使用独立输出和新运行身份，保留原失败；已经调度完成的初态检查不重做。此次修订仍为 CPU 测试，不加载真实模型或 processor，也不放行 GPU 或正式评测。


### 实际服务器观察器与三场景入口（2026-11-14 更新）

入口继续使用本节固定的三个原 v1 NavigateKitchen dev 身份。每局先在全新进程生成零动作渲染参考，再在另一全新进程执行原生策略；推理前严格比较完整 XML、JSON 化元数据、instruction、layout/style、积分状态、两模型状态和相机像素。使用同一个原生服务器顺序处理三局，各局 replan 0 重置缓存。每局最多 450 步，保留原生 success、terminated、truncated 和 horizon 终止；三局都提前结束时记录“历史覆盖不足”，不能延长或补选种子。

服务器在已加载的原生策略对象外添加观察器，包装原 `predict`。一次请求内临时包装以下七个真实调用：`h264_roundtrip`、`_update`、`compose_canvas`、`prompt.encode`、`_sample`、`model.encode`、`sample_actions`。每个必须恰好调用一次，委托原函数后返回原来的同一对象；成功或异常都恢复包装。不得额外编码、再调用模型、改变随机状态或修改传入数组。`model.encode` 必须收到 `_sample` 返回的同一个字典，`sample_actions` 必须收到原编码函数返回的同一个对象与原请求 seed。异常向原服务器和客户端传播，该回合停止并保留失败记录。

每次保存请求前的 intent 和完成后的 observation JSON，包含 scene 身份、session、replan、seed、观测步、instruction、原始相机与 state 的 dtype/shape/SHA256、各原始帧摘要、真实压缩后各帧摘要、实际缓存摘要、选中帧、canvas、video/history/mask/state、原始动作和实际返回动作摘要。为缓存中每一步额外记录来源 `(replan, request_frame_index)`。缓存检查使用从每次真实压缩输出独立复制的时间线；保留窗口、九槽顺序、缺失槽精确零值及 prompt 输入检查沿用上面的严格规则。

在实际到达的 step 0、160、352、368、448 保存 NPZ，字段为 `raw_request`、`compressed_request`、`selected_cache`、`composed_canvas`、`raw_state`、`video`、`history`、`history_valid`、`state`。每局结束后逐请求连接客户端与服务器记录：session/replan/seed/观测步必须相同；原相机/state 与实际发送 NPZ 的类型、形状和字节相同；每个请求帧摘要必须来自初始观测或对应环境步记录；动作摘要必须等于客户端实际收到的动作；缓存来源必须能追溯到原请求压缩后的对应帧。所有请求与 intent 数量、原生步数和终止原因须一致，任何额外、缺失或失败记录均不能算完整通过。

新 CPU 检查专门检验这些观察钩子：第一个固定身份执行 29 次合成图像请求，后两个身份各执行一次 replan 0，共 31 次。实际调用公开原生解析、H264、缓存、取样、canvas、state/action 转换；prompt、model encode 和 action sampling 使用明确的替身，不加载真实模型或 processor，不允许 CUDA 初始化。这样得到的结果只能用于观察器工程验收。19 项本地回归覆盖单次委托和对象身份、随机状态/输入不变、缓存错误、来源错配、错误 seed/session/timeline、异常恢复、预算停止、自然提前成功以及缺失现场验收时拒绝 GPU；19 项已通过，31 次现场 CPU 结果尚未验收。

资源上限：该新 CPU 检查为 0 GPU、2 CPU、16 GiB RAM、30 分钟、3 GiB 证据；后续三局 GPU 验证为 1 GPU、8 CPU、128 GiB RAM、2 小时、最多 1,350 环境步和 8 GiB 证据。真实 GPU 执行前须将三个准备结果——三输入初态重复性、修订后的原生历史组装、新观察入口 CPU 检查——完整原件分别验收，再生成绑定实现版本、三个固定身份、各证据大小/SHA256 和自然终止规则的放行文件。调度完成或文件清单不能替代此验收。此更新没有增加正式评测、正式 SFT 或 Ascend 完成量。

## 11. 渲染器析构修复与三个旧失败分片的固定复测

### 11.1 观察、原因边界与修复算法

公开固定 robosuite 的 `MjRenderContext` 由自己的 `gl_ctx` 持有 EGL context。较旧的 renderer 被延迟析构时，另一个 renderer 的 context 可能正在当前线程上激活。历史 Loop 实机轨迹记录到：旧 renderer 的原生析构在不同的当前 context 下执行，随后当前 renderer 的三个 framebuffer 消失并出现 GL 1286。这支持该次失败由错误 context 下的资源清理引起。三个旧 Awomo 失败都记录了严格 GL 检查失败，但没有逐一保存同等因果轨迹，不能声称三次错误原因都已直接证明。

修复只在评测进程内显式安装，保存原 `MjRenderContext.__del__` 和原渲染外 GL 检查函数。接手端按以下顺序实现，保留公开原生析构本身：

1. 从待析构对象的 `gl_ctx._context` 读取 owner 指针。没有 owner 或没有 `con` 的半初始化对象仍直接调用原析构；保留其原生错误行为。
2. 在切换前用 `eglGetCurrentContext`、`eglGetCurrentDisplay`、`eglGetCurrentSurface(EGL_DRAW=0x3059)`、`eglGetCurrentSurface(EGL_READ=0x305A)`、`eglQueryAPI` 记录当前 context、display、draw/read surface 和 API。日志保存这些值及待析构 owner，日志写入后刷新。
3. 调用该对象的 `gl_ctx.make_current()`，重新读取当前 context 并断言等于 owner，再调用原 `__del__` 恰好一次。只选择 owner 仍不足以保护被中断的 renderer：原析构还可能释放当前线程的 EGL 绑定，因此必须执行下一步。
4. 无论原析构成功或抛异常，在 `finally` 中处理之前保存的状态。只有保存的 context 非零且与被析构 owner 不同时，先要求保存的 display 非零，再调用 `eglBindAPI(saved_api)` 和 `eglMakeCurrent(saved_display, saved_draw, saved_read, saved_context)`；每个调用必须成功，再读回五个字段逐项精确比较。不得重新绑定刚被销毁的 owner，也不得人为创建一个“替代”context。
5. 保存 after 事件、当前五字段、实际清理次数和恢复次数。切换、原析构或恢复异常都保留第一条错误并继续抛出。由于 Python 可能忽略析构异常，后续每次原渲染外检查前必须先断言该错误槽为空；worker 结束时也检查一次，有错误则整次运行记为 incomplete，不能留下成功回执。
6. 结束或异常退出时还原两个原函数。不得增加额外 `glGetError` 读取、吞掉未知 GL 错误、修改 shininess、材质、XML、物理状态或策略输入。`ctypes` 调用须显式声明 EGL 指针和整型的参数/返回类型，避免 64 位指针被截断。

这个实现适用于使用 EGL 的渲染进程，不是 `torch_npu` 的设备适配。Ascend 策略若通过另一台 GPU 机器运行模拟器，应在该模拟器机器独立验证；不要在没有 EGL renderer 的 NPU 进程中强行安装。

### 11.2 必须保留的验证与已有证据

先分别在独立进程比较三个小型渲染对照：原析构、仅切换到 owner 再原析构、切换到 owner 且恢复此前 context。记录清理前后当前 context、framebuffer 存在性、后续渲染和像素。仅切换 owner 的对照可在后续显式重新绑定后恢复渲染，不能据此证明被中断的调用可直接继续；完整修复还必须证明此前五字段在析构返回时已恢复。保留负对照的实际结果，不把预期失败改写为通过。

已有 Awomo NVIDIA 诊断完成 14/14 个固定 BreadSelection 回合、18,876 环境步与 56,922 次渲染，28 次清理中捕获 1 次真实的其他 context 恢复；错误槽为空，没有场景或材质修改，14 个初始状态和相机输入与旧固定参考相同。这 14 局是工程诊断，不计正式成绩。GPU2 的固定原生入口也已单独验收。新硬件或重新实现仍需自己的 CPU、合成渲染和真实完整回合验证，不能继承这些结果。

### 11.3 2026-11-14 准备的三项正式后继

三项旧原生策略评测因 GL 检查失败而不完整。修复后分别使用新的运行身份，从原固定 trial 0 开始执行完整 30 局。旧部分回合继续保留，不拼接、不按成功选择样本，也不把上述 14 局诊断加入分母。

| 任务 | 旧有效部分回合 | 新 trial | constructor seed | 新 reset seeds | 原 horizon |
|---|---:|---|---:|---|---:|
| BreadSelection | 13 | 0–29 | 7 | 107–136 | 1950 |
| CategorizeCondiments | 11 | 0–29 | 7 | 157–186 | 1650 |
| CuttingToolSelection | 29 | 0–29 | 7 | 207–236 | 1200 |

继续使用公开固定 Awomo 官方策略、同一发布 checkpoint、`split=pretrain`、16 步动作执行、原 H264/历史组装与成功终止。每任务先做精确任务和节点的 CPU 前置：0 GPU、4 CPU、16 GiB RAM、20 分钟；检查模型、50 任务 horizon、公开源码、策略/模拟器包版本、清理证据和前后环境，实际不创建 EGL context、不执行 episode。每项正式后继最多 1 GPU、4 小时，worker 上限 10,800 秒；三项共 90 局、最多 12 GPU·小时。GPU 分配前须验收完整 CPU 原件，并核对旧身份没有活动、模糊或已经完成的重复运行；保留已知失败，不能误写成从未执行。

本节是已准备并提交的执行合同，不是新通过结果。CPU 完整回传、独立验收和 GPU 准入仍待完成；正式累计仍按主文第 7.28 节，Human/recovery SFT 与 Ascend 实测没有新增计数。

## 12. 真实 Human 读取器与模型训练状态的联合恢复

**状态（2026-11-15）：** 数据读取游标恢复与三窗口模型/AdamW/RNG 恢复已有各自独立的 NVIDIA 证据；两者在同一训练入口内的联合恢复尚未验收。新增联合入口及保存合同已实现，本地共 69 项检查，67 项通过、2 项因本机缺少 Torch 跳过。两个 B300 节点的 CPU 执行端现已分别报告 69 项通过，部署包装器退出码均为 0；48 项源码执行后摘要与固定计划一致，环境 freeze 未变。B300 节点 1 的调度终态为 COMPLETED、退出码 0:0；节点 2 的调度记录已过期，调度退出码未知。上述终态元数据已核对，但完整 receipt、preview、unit 结果和日志尚未回传验收；两项单卡诊断未放行，没有新增正式 SFT、benchmark 或 Ascend 计数。下列内容直接给出接手端重建所需的输入、执行顺序和验收要求。

### 12.1 固定输入与训练目标

使用第 4 节的原始 910,723 个 train 窗口，数据摘要 `a514224e5a149effa98279627b91fa25d3b19f33bff1b3fc9832dd76b2294b49`。按 `PCG64(SeedSequence([42, 0]))` 生成 epoch 0 的排列，world size 1、rank 0、batch size 1，仅取前五批，不能自行补选窗口：

| 更新 | 窗口索引 | 源任务组 / 任务 | episode ordinal / episode index | start |
|---|---:|---|---|---:|
| 1 | 830901 | composite_unseen / SeparateFreezerRack | 23728 / 433 | 992 |
| 2 | 357660 | composite_seen / SearingMeat | 13847 / 180 | 624 |
| 3 | 267255 | composite_seen / PackIdenticalLunches | 11487 / 349 | 624 |
| 4 | 893913 | composite_unseen / WeighIngredients | 24845 / 42 | 64 |
| 5 | 418544 | composite_seen / SteamInMicrowave | 15405 / 218 | 64 |

五批必须均来自 train；后两批包括早期历史。每次真实读取原视频、按逐相机 offset 解码，保留原生 H264、canvas 与 prompt processor。采用两个 `spawn` worker、每 worker `prefetch_factor=1`、batch size 1、不启用随机数据增强和梯度累积。DataLoader 使用独立 CPU `torch.Generator`，seed 42；创建 iterator 前后比较全局 CPU RNG，不能让 worker 初始化改变训练 RNG。

模型从原始发布权重初始化；训练范围和数值目标沿用总入口第 7.1 节：全部 DiT/proprio 的 171 个 FP32 参数张量、3,876,166,656 个元素可训练，原生 Qwen、发布 LoRA 与 VAE 冻结；BF16 autocast、math SDPA、确定性算法、关闭 TF32。AdamW `lr=1e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0`，clip 1.0，训练 seed 20261113，sigma 0.5、timestep 500，图像 velocity MSE 加 11 个有效动作坐标的 velocity MSE。每批重新执行冻结编码器；不能直接复用历史编码缓存。这仍是工程诊断配方。

### 12.2 保存边界与新进程重放

1. 单独维护“已发给 worker 的批次数”和“已成功完成优化器更新的批次数”。每批严格核对预注册窗口索引、episode 身份、split 和原生 processor 输出；训练执行 forward、有限 loss 检查、backward、全部参数梯度检查、裁剪、`optimizer.step()`、设备同步，全部成功后才提交该批消费游标。失败不提前推进游标。
2. 更新 2 完成时，要求已发出的批次数大于 2，已消费数恰好为 2。完整 checkpoint 同时包含：格式版本、代码与全部输入/配方绑定、`completed_updates=2`、已消费 sampler 状态、全部模型参数、完整 AdamW 状态、CPU RNG 和全部 CUDA RNG。sampler 状态必须绑定数据摘要、窗口数、seed、epoch、world size、rank、batch size、排列摘要和尾部丢弃摘要；恢复时重新计算并比较。
3. 检查 `completed_updates` 是整数且与已消费游标精确相等；固定单 rank/batch 1，不允许把不同布局的状态强行恢复。缺失模型、优化器、任意 RNG，或数据/源码/配方/排列不一致时立即拒绝。
4. 先将所有张量复制成独立 CPU 快照，再以 exclusive create 写同目录 `.partial` 文件；flush 和文件 fsync 后，用不可覆盖的原子硬链接发布最终文件，删除临时硬链接，并 fsync 目录。写入失败保留 partial，最终文件不出现；已有最终文件或 partial 均不得覆盖。不要分别保存一个新游标和一个旧模型。
5. 连续进程完成更新 1–5 后退出。另起全新进程重新加载原生冻结组件和 FP32 DiT/AdamW，校验更新 2 checkpoint 的完整大小与 SHA256，严格加载模型、优化器及 sampler；重新创建使用独立 RNG 的数据 worker，最后恢复训练 CPU/CUDA RNG。从更新 3 开始读取真实数据并重放 3–5，不从数据开头读两批再丢弃。
6. 连续与恢复分支逐更新精确比较：17 个 CPU 输入数组的 dtype/shape/完整内容摘要、冻结编码结果、图像和动作噪声、各项 loss、梯度范数、全部 171 个参数张量、完整优化器状态、CPU/CUDA RNG 和消费游标。两进程均检查冻结编码器前后不变且无梯度，保留峰值显存。首次不一致即失败，不提高容差、不挑选成功批次。

逻辑更新共 5 次，实际更新共 8 次（连续 5 次加重放 3 次）。只保留更新 2 和恢复后的更新 5 诊断保存点；收到停止信号时在当前完整更新后保存自己的中断点并退出。所有诊断权重保持独立，不能作为正式 SFT 或 benchmark 候选的初始化权重。Ascend 实现还须替换并验证对应设备 RNG、同步及保存恢复 API，不能认为 CUDA RNG 字节可以直接用于 NPU。

### 12.3 CPU 与 GPU 验收边界

每节点 CPU 前置最多 0 GPU、8 CPU、32 GiB RAM、45 分钟，真实数据子进程最多 1,800 秒。要求 69 项检查真实通过且零 skip；核对原始 21 个模型文件和配置、已有读取依赖的完整 wheel/member 摘要、原视频与数值身份、五个真实 batch、原生 processor 输出及全局 CPU RNG。此阶段保存的 checkpoint metadata 只是数据边界预览，没有模型加载或优化器更新，不能算联合恢复通过。

该节点完整 CPU receipt、所有原始日志、源码和环境前后检查经独立验收后，才生成本节点放行文件。放行文件绑定同一源码/输入、CPU 运行身份、CPU receipt SHA256、全部 CPU 产物大小/摘要及指定 GPU 运行身份；GPU 入口须验证所有字段与文件。调度完成、producer 摘要或缺少原件的清单不能代替放行。

每节点 GPU 诊断最多 1 GPU、8 CPU、192 GiB RAM、120 分钟；连续和恢复子进程各最多 3,000 秒。checkpoint 峰值预算 200 GB，开始前至少有 200 GB 可用空间，并仍满足集群磁盘门槛；完整小型 JSON/日志回交上限 100 MB，大 checkpoint/partial 保留源节点并记录完整大小和 SHA256。两站合计最多 4 GPU·小时，是预算而非已消耗量。尚未验收真实联合恢复、多卡、梯度累积、随机增强、全量训练或 Ascend；现有单独模块的成功不能代替这些结果。


## 13. Checkpoint 完成边界与失败传播

**状态（2026-11-15）：** 这是通用训练器的保存/恢复候选修复。20 项本地控制流程测试通过，模型与存储使用明确的替身；测试执行从实际训练器提取的方法体，没有实例化完整训练器、运行真实 DCP 或加载模型。相同测试在修复前的候选实现中复现 6 项失败。一个 B300 节点的执行端现已报告 20 项检查零失败/零跳过、九种两进程 CPU Gloo 对照全部通过；调度完成且退出码为 0:0，9 份源码及环境前后摘要相同。48 件、31,763 bytes 的节点产物清单已核对，18 个 rank 结果文件均列在其中；但原始 rank JSON、完整 producer receipt 和日志尚未回传独立验收，不能把执行端摘要当作完整通过。此修复没有替换已有联合恢复诊断的固定版本，不增加正式训练或 Ascend 结果。

### 13.1 可复制的文件系统边界

在 Linux 目标文件系统上，完成标记必须恰好为 `complete\n`，且标记不能是 symlink。已有原生完整 checkpoint 使用这一格式，仍可读取。扫描自动恢复位置时，只接受带有效标记的目录；更大 step 的不完整目录或指向它的 `LATEST` 不能遮住较早的完整保存点。下面的代码只判定发布边界，不能代替模型、优化器、游标、RNG 或逐文件摘要验证：

```python
"""Filesystem completion boundary; this does not validate model payload contents."""
from pathlib import Path
import os
import re


def is_complete_checkpoint(path):
    path = Path(path)
    marker = path / '.complete'
    try:
        return path.is_dir() and not marker.is_symlink() and marker.read_bytes() == b'complete\n'
    except OSError:
        return False


def require_complete_checkpoint(path):
    if not is_complete_checkpoint(path):
        raise FileNotFoundError(f'Checkpoint is missing its valid completion marker: {path}')


def find_latest_completed_checkpoint(checkpoint_dir):
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return None
    latest = root / 'LATEST'
    target = None
    try:
        if latest.is_symlink():
            target = os.readlink(latest)
        elif latest.with_name('LATEST.txt').is_file():
            target = latest.with_name('LATEST.txt').read_text().strip()
    except OSError:
        pass
    if target:
        candidate = Path(target) if Path(target).is_absolute() else root / target
        if is_complete_checkpoint(candidate):
            return candidate
    candidates = []
    for path in root.glob('*step_*'):
        match = re.search(r'step_(\d+)$', path.name)
        if match and is_complete_checkpoint(path):
            candidates.append((int(match.group(1)), path.name, path))
    return max(candidates)[-1] if candidates else None


def publish_complete_marker(path):
    """Called by rank0 only, after all ranks have completed every save future."""
    path = Path(path)
    temporary, marker = path / '.complete.partial', path / '.complete'
    if marker.exists():
        raise FileExistsError(marker)
    with temporary.open('xb') as stream:
        stream.write(b'complete\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, marker)
    temporary.unlink()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
```

文件系统须实际支持同目录 exclusive create、原子硬链接和 fsync；不要把本机临时目录检查当作目标共享存储已经验证。已有最终标记或 `.complete.partial` 均不得覆盖。路径中的模型文件、优化器与其他状态仍需单独完整读取和校验。

### 13.2 保存顺序与所有恢复入口

按下列顺序接入公开 SFT 基底的训练器，所有仍参与的 rank 使用相同顺序和相同检查组：

1. 等待前一次模型及优化器 upload future。即使模型 future 抛错，也必须继续等待优化器 future；两者结束后清除 future 引用，再抛出第一条错误。将每个 rank 的成功或异常类型/信息通过 `all_gather_object` 汇总；任意 rank 有错，全部在同一边界退出。
2. rank 0 检查目标是否已有有效完成标记，并广播是否跳过。未完成时，rank 0 用 `exist_ok=False` 建立本次目录；已有 partial 目录保留并失败，不能续写覆盖。目录创建结果在开始保存前同步给各 rank。
3. rank 0 写 loss metadata 与普通本地 tracked state，并在进入后续分布式保存前同步写入错误。验证指标字典只能读取，不能 `pop` 修改调用方传入的对象；多个指标直接各自保存，不写入 `dict.update()` 的返回值。
4. 运行模型、分布式状态、每个 DP rank 的数据/RNG 状态及优化器保存。随后等待**本次**模型和优化器 upload future 完成，并汇总各 rank 错误。单纯 barrier 不能代表后台写盘完成。
5. rank 0 保存完整配置和训练配方附件；同步结果。全部成功后，才调用上面的 `publish_complete_marker`，再次同步发布结果，最后更新 `LATEST` 和最佳验证保存点指针。指针写入失败仍报错；已完成的数据目录保留，之后可按有效标记扫描找到。
6. 恢复检查必须覆盖显式 `load_checkpoint`、自动选择以及训练 `setup` 中提前加载模型权重的路径，放在第一次状态读取之前。完整训练恢复缺少有效标记就拒绝。显式仅加载模型权重的 warm start 与完整训练恢复分别配置；前者保留对发布权重目录的兼容性，不能冒充优化器或数据游标恢复。

该顺序会等待当前保存完成再返回，牺牲异步保存与后续更新的重叠。不能宣称已经恢复多卡吞吐。这里同步的是仍存活并到达边界的 rank；DCP 内部 collective 卡住、进程崩溃、共享文件系统损坏及通信组不匹配仍需真实运行、明确超时和独立故障处理验证。

### 13.3 接手端的最小检查集合

20 项本地检查覆盖：较大 step/错误 LATEST 的排除、旧标记及文本指针、损坏或 symlink 标记、已有 marker/partial 的保留、既有不完整目录、当前 future 未结束时禁止发布、模型失败后仍等待优化器、前一次 future 失败、优化器失败、附件失败、配置失败、rank 0 普通状态失败、loss 写入失败、验证指标保持不变、指针失败后完整目录可发现、显式恢复拒绝 partial、完整保存点自动恢复、setup 提前读权重前的门槛、仅权重 warm start、完整 setup 恢复。

两进程对照依次测试：正常完成、前一次 future 失败、当前模型 future 失败、当前优化器 future 失败、rank 0 附件失败、配置失败、普通状态失败、loss metadata 失败、已有 partial 目录。前三类 future 故障在 rank 1 注入，后四种写入故障在 rank 0 注入；每种失败都要求两个 rank 报告同一汇总错误，且没有完成标记或 LATEST。正常情形在 rank 1 的当前 future 内短暂等待，并确认这时尚未发布。

固定预算：0 GPU、4 CPU、16 GiB RAM、15 分钟；每个 Gloo group 超时 20 秒，每组两个子进程总等待上限 55 秒，总 driver 预算 600 秒，文件输出不超过 10 MB。保存每个 rank 的错误、实际后端、future 等待事件、源码/环境前后摘要及退出码；父进程只终止自己超时的子进程。该对照即使通过，也只证明上述控制流程和 CPU Gloo 传播；真实 DCP、实际模型保存恢复、NCCL/HCCL、跨节点及 Ascend 都仍需另验。

## 14. 真实 DCP 的独立 CPU 保存与恢复诊断

**状态（2026-11-15）：** 18 份相关源码的语法检查通过。B300 执行端已报告单/双 rank 的连续运行与新进程恢复共四阶段通过，六个 rank 进程均退出 0，driver 用时 24.05 秒；调度终态为 COMPLETED、退出码 0:0。18 份源码及环境前后摘要已核对。节点索引列出 45 件、678,601 bytes 的原件，包括六份 rank JSON、15 份 oracle、四份 DCP metadata、六个 DCP shard 和三份 rank 辅助状态；完整原件仍待正常磁盘容量下回传，尚未完成独立内容验收。本节直接给出接手端可独立实现的完整实验规格。它使用真实 `Checkpointer`、PyTorch DCP 和 AdamW，不使用模型/存储替身；模型是下述 61 参数的合成小模型，不能计作 WAM、Human SFT、完整训练器、FSDP、GPU 或 Ascend 验收。

### 14.1 固定模型、运行路径与比较对象

使用本指南公开 SFT 基底的 checkpoint 模块，并先按第 13 节接入保存完成边界。以下模型、优化器与调度器用于诊断：

```python
import torch
from awomo.training.checkpoint.checkpointing import Checkpointer, CheckpointingConfig


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 7), torch.nn.Tanh(),
            torch.nn.Dropout(0.25), torch.nn.Linear(7, 3),
        )
        self.unused = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer('batches', torch.zeros((), dtype=torch.int64))

    def forward(self, x):
        self.batches.add_(1)
        return self.layers(x)


def make_checkpoint_probe(checkpoint_root, rank, world_size, resuming=False):
    torch.manual_seed(9876 if resuming else 1541)
    model = TinyModel()
    assert sum(p.numel() for p in model.parameters()) == 61
    train_model = (torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True) if world_size == 2 else model)
    optimizer = torch.optim.AdamW(
        train_model.parameters(), lr=0.003, betas=(0.9, 0.95), foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    config = CheckpointingConfig(
        enabled=True, checkpoint_dir=checkpoint_root, model_save_format='torch_save',
        model_cache_dir=checkpoint_root.parent / 'unused-cache', model_repo_id='tiny-cpu-fixture',
        save_consolidated=False, is_peft=False, is_async=False,
    )
    checkpointer = Checkpointer(config, dp_rank=rank, tp_rank=0, pp_rank=0)
    return model, train_model, optimizer, scheduler, checkpointer
```

当前对照环境固定 `torch==2.8.0+cu129`，实际只使用 CPU，`CUDA_VISIBLE_DEVICES` 为空；不安装依赖、不下载权重。所有进程设 Torch intra-op/inter-op 线程各为 1，启用 deterministic algorithms。使用真实 Gloo，分别运行 world size 1 和 2。双 rank 训练通过 DDP 平均梯度；保存与加载传入未包装的 `model`，优化器持有同一组参数。该实验不验证 DDP 前缀转换或 DTensor 分片。

训练前分别设置 Torch、NumPy、Python 随机种子为 `2000+rank`、`3000+rank`、`4000+rank`。每次更新生成 `x=torch.randn(5,4)`、`target=torch.randn(5,3)`，以及 `scale=float(np.random.uniform(0.8,1.2)+random.random()/10)`；loss 为 `((train_model(x)*scale-target)**2).mean()`。依次清梯度、前向、反向、检查有限值、AdamW step、scheduler step，最后推进已完成更新计数。每次须有参数改变；四个参与训练的参数张量有梯度，`unused` 不得出现 AdamW 状态。其他优化器默认值由上述固定 Torch 版本确定，实际完整 `param_groups` 必须保存并比较。

### 14.2 连续分支与全新进程恢复

每个 world size 独立使用新目录，先连续完成更新 1–5，再退出全部进程；随后重新 spawn 同样数量的进程，从更新 2 恢复并重放更新 3–5。总计四个运行阶段、六个 rank 进程，每个 rank 的逻辑更新为 5、实际更新为 8。两种 world size 分别比较各自连续与恢复分支，不要求不同 world size 得到相同权重。

1. 每次更新后保存独立 oracle，包含全部模型参数与 buffer、完整 optimizer state/param_groups、scheduler state、rank/world size/已完成计数、Torch CPU RNG、NumPy RNG、Python RNG，并保存本次输入、target、scale、loss 与四份梯度。
2. 连续分支更新 2 后，rank 0 新建保存目录，各 rank 同步。调用真实 `save_model(model, path)`、`save_on_dp_ranks(rank_state, state_name, path)`、`save_optimizer(optimizer, model, path, scheduler)` 和 `async_wait()`。辅助状态的 `state_name` 使用包含本次运行与 world size 身份的唯一名字，避免该写入器的临时路径与其他运行相撞。
3. 所有 rank 的保存返回后同步，再由 rank 0 发布第 13 节的完成标记，之后再次同步。必须存在真实 `model/.metadata`、`optim/.metadata` 及其 DCP 文件；保存动作本身不得改变参数、优化器、schedule 或任何 RNG。保留更高 step 的不完整目录，确认选择器忽略它且显式完成检查拒绝它。
4. 恢复进程采用不同的模型初始化 seed（上例 `9876`），证明恢复确实改变了初始化参数。依次调用 `load_model(model, path/'model')`、`load_optimizer(optimizer, model, path, scheduler)`，最后调用 `load_on_dp_ranks(rank_state, state_name, path)` 恢复 rank 身份、计数和三类 RNG。辅助状态先核对 world size/rank 及计数恰为整数 2，再设置 RNG。
5. 在下一次更新前，将完整恢复状态与更新 2 的 oracle 逐叶比较。每次重放更新 3–5 后，再逐叶比较全部状态和本次输入、loss、梯度。张量与数组的 dtype、shape、每个元素必须完全一致，字典键、列表/tuple 类型及标量也须一致；不能只比较 loss 或参数摘要。
6. 恢复结束后重新计算 checkpoint 目录下全部普通文件的大小与 SHA256，要求与连续分支结束时完全相同。原 checkpoint 和 oracle 保留，恢复过程不能修改它们。

### 14.3 预算、失败与解释边界

资源上限为 0 GPU、4 CPU、16 GiB RAM、15 分钟；每个 Gloo group 超时 30 秒，每个阶段等待全部子进程最多 120 秒，总 driver 预算 600 秒，完整小型输出不超过 10 MB。导入失败即停，不用临时模块替身或额外安装把失败改成通过。父进程保留各子进程退出码、日志、JSON 和完整失败栈；超时时只结束自己的子进程，不原样重试。源码和环境在运行前后均核验。

回交材料包括四阶段结果、六个 rank 的原始日志/JSON、全部 oracle、实际 DCP 目录及逐文件索引。只有完整原件经独立读取、校验和内容比较后，才能记作本项通过。该实验的计数器不是真实 DataLoader/prefetch 游标；也没有实例化完整 `Trainer`、执行异步 DCP、测试 HCCL/NCCL 或恢复真实 WAM 权重。现有联合数据/训练恢复及完整训练器验收仍须单独完成。


## 15. 真实 Trainer 导入前置与独立依赖环境

截至 2026-11-15，首次现场准备在安装清单的路径校验失败；改用完整 prefix 后，执行端报告真实 Trainer 导入及下列组件检查通过，完整原件验收仍待完成。第 14 节只调用实际 Checkpointer，不会导入完整 Trainer；不能据此认定训练器可启动。固定推理环境缺少 `torchdata` 和 `wandb`，Trainer 在模块导入时即引用两者。其 `setup()` 还明确要求 CUDA，所以本节 CPU 检查仅覆盖真实模块导入、构造、状态登记与组件内存恢复，不调用 `setup()`，也不执行训练更新或 Trainer checkpoint 保存。

### 15.1 固定依赖和隔离方法

NVIDIA 对照使用 [GPU 环境](GPU_ENVIRONMENTS.md) 中的 Python 3.11 / Torch 2.8.0+cu129 基础环境，先比较完整包名/版本清单。仅增加下表八个包；基础环境不安装、升级或覆盖。前两项版本来自公开训练依赖，另外六项满足其有效依赖闭包。全部 17 条有效依赖边已与基础环境的 68 个固定包核对；这不等于实际导入兼容性已经通过。

| 包 | 版本 | 精确 wheel 文件 | 字节数 | SHA256 |
|---|---|---|---:|---|
| torchdata | 0.11.0 | `torchdata-0.11.0-py3-none-any.whl` | 61,968 | `52b940fbbe0e00fb21cabddf528449d1bec5bfb0d0823b7487b15f951658ee33` |
| wandb | 0.27.0 | `wandb-0.27.0-py3-none-manylinux_2_28_x86_64.whl` | 27,256,204 | `f772da7005cc26a2a32b729a16982a583dc68b3d493df6a09d0aa5c5ca5a2060` |
| gitpython | 3.2.1 | `gitpython-3.2.1-py3-none-any.whl` | 231,548 | `d66b0832aa8755c1a84892937282df0befb16c97e5a64aeebd68be9c5e323f53` |
| gitdb | 4.0.12 | `gitdb-4.0.12-py3-none-any.whl` | 62,794 | `67073e15955400952c6565cc3e707c554a4eea2e428946f7a4c162fab9bd9bcf` |
| smmap | 5.0.3 | `smmap-5.0.3-py3-none-any.whl` | 24,390 | `c106e05d5a61449cf6ba9a1e650227ecfb141590d2a98412103ff35d89fc7b2f` |
| platformdirs | 4.14.5 | `platformdirs-4.14.5-py3-none-any.whl` | 35,402 | `8e778c6fa421516f4d3369449f8df3b0e1b72c0e0005fe88e3ec772d0a14e750` |
| protobuf | 7.36.2 | `protobuf-7.36.2-py3-none-any.whl` | 179,806 | `bdb3a345d48db958e6ce1f18e508beb0cc981d64f24088427549c866cd039f1e` |
| sentry-sdk | 2.75.0 | `sentry_sdk-2.75.0-py3-none-any.whl` | 555,165 | `9a10e97de2cbd42ad7e4a6e02a0d5af19de9811eb02445077d41ba14c08e170c` |

来源为各版本的官方 PyPI 元数据，地址规则为 `https://pypi.org/pypi/<包名>/<版本>/json`。直接从该 JSON 的 `urls` 中选择上表**完全相同的 filename**，并要求 `size`、`digests.sha256` 与表中相等、`yanked=false`，只接受 `https://files.pythonhosted.org/` 的文件 URL。特别是 protobuf 固定为上表 pure-Python wheel，不能默换为相同版本的 ABI wheel。总下载量 **28,407,277 bytes**；每个文件下载后再次核大小和 SHA256，任一不符即停。

1. 在新且不存在的输出目录创建 `wheelhouse/`、`home/`、`tmp/`；安装前缀为该目录下的 `overlay/`，Python 包位于 `overlay/lib/python3.11/site-packages/`，命令入口位于 `overlay/bin/`。不复用旧失败目录，不改基础环境或旧实验目录。
2. 用基础 Python 执行 `python -B -m pip --isolated install --no-input --disable-pip-version-check --no-cache-dir --no-index --no-deps --no-compile --ignore-installed --prefix <新目录>/overlay <八个已核验wheel的绝对路径>`。这里不给 pip 自行解析或下载其他版本的机会。
3. 核对 overlay 中分发包集合恰好为这八项，版本相等；遍历安装后每项 RECORD，将每个 RECORD 路径相对于 `overlay/lib/python3.11/site-packages/` 解析，要求最终路径位于整个 `overlay/` 前缀内，逐个校验带 SHA256 的成员。合法命令入口可以位于该前缀内的 `bin/`；不能把允许范围扩大到整个工作目录。另保存全部安装文件的大小和摘要，安装体积上限 300 MB。
4. 以新子进程导入真实 Trainer。仅把 `overlay/lib/python3.11/site-packages/` 与公开固定源码放进 `PYTHONPATH`；设置 `PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1`、`CUDA_VISIBLE_DEVICES=`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`WANDB_MODE=disabled`，CPU 数学库线程设为 1。HOME/TMPDIR 指向上述新目录，不继承服务令牌。子进程拒绝 socket 连接和 DNS 解析，不调用 W&B 登录、初始化或上传。
5. 保存完整导入日志、真实类所属模块，以及所有加载的 `awomo` 模块相对路径和源文件摘要。导入必须来自本次固定源码；不能注入同名假模块或抽取 AST 方法来替代。

**首次失败与修正依据：** 执行端报告首次准备耗时 6.12 秒，八个 wheel 大小与 SHA256 全匹配，源码和基础环境前后不变；在 RECORD 路径包含性检查停止，未开始真实导入，完整原件验收仍待完成。候选原因是 `pip --target` 搬移命令入口后，RECORD 仍使用临时安装布局的相对路径。仅凭这一断言不能认定发生了外部写入。后继诊断只读取旧目录内的 metadata 和已限定的 `bin/wandb`、`bin/wb`，比较它们与 RECORD 的摘要；只记录解析出的外部路径字符串，绝不打开这些路径。若异常不是这两个已限定入口则立即停止。旧目录保留原状，新尝试使用上面的完整 `--prefix` 布局。

五项标准库回归检查已通过：合法 prefix 内入口通过；旧 target 布局的路径问题可重现且找到摘要匹配的内部入口；真正越界拒绝；入口内容变化拒绝；未识别的旧越界项拒绝。这些本地文件测试不等于现场 Trainer 导入通过。原生 Python 3.11 的 metadata 行为与本地 Python 3.14 对不存在 RECORD 目标的过滤行为不同，测试显式覆盖了这一差异。

本表只定义现有 NVIDIA 对照的 CPU 准备环境。Ascend 接手端必须另行选择与本机 CANN/torch_npu 相符的 Torch 栈；不能把 CUDA wheel 当作 NPU 环境。相同的导入和状态登记检查仍应在该目标栈真实执行，并明确记录依赖差异。

### 15.2 真实对象检查的直接实现

在完成上述隔离和输入校验的子进程内执行以下核心检查。这里只构造一个 10 参数线性层；没有真实 WAM 权重、Human 样本、前反向或优化器更新。

```python
import random
import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from awomo.training.trainer import Trainer, is_dataloader, is_lr_scheduler, is_model, is_optimizer
from awomo.training.config.loader import ConfigNode
from awomo.training.optim.scheduler import OptimizerParamScheduler
from awomo.training.rng import StatefulRNG

assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
trainer = Trainer(ConfigNode({}))
trainer.model = torch.nn.Linear(4, 2)
trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.001, foreach=False)
trainer.lr_scheduler = OptimizerParamScheduler(
    trainer.optimizer, init_lr=.001, max_lr=.001, min_lr=.001,
    lr_warmup_steps=0, lr_decay_steps=5, lr_decay_style='constant',
    start_wd=.01, end_wd=.01, wd_incr_steps=5, wd_incr_style='constant')
dataset = torch.utils.data.TensorDataset(torch.arange(12).reshape(6, 2))
trainer.train_dataloader = StatefulDataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)
trainer.rng = StatefulRNG(42, ranked=False)
assert sorted(trainer.__dict__['__state_tracked']) == [
    'cfg', 'lr_scheduler', 'model', 'optimizer', 'rng', 'train_dataloader']
assert is_model(trainer.model) and is_optimizer(trainer.optimizer)
assert is_lr_scheduler(trainer.lr_scheduler) and is_dataloader(trainer.train_dataloader)
iterator = iter(trainer.train_dataloader)
first = next(iterator)[0]
loader_state = trainer.train_dataloader.state_dict()
expected = next(iterator)[0]
restored_loader = StatefulDataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)
restored_loader.load_state_dict(loader_state)
assert torch.equal(expected, next(iter(restored_loader))[0])
rng_state = trainer.rng.state_dict()
expected_rng = (random.random(), np.random.random(), torch.rand(4))
trainer.rng.load_state_dict(rng_state)
restored_rng = (random.random(), np.random.random(), torch.rand(4))
assert expected_rng[:2] == restored_rng[:2]
assert torch.equal(expected_rng[2], restored_rng[2])
assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
```

保留 first/continuous-next/restored-next 三个 batch 的实际值，预期分别为 `[[0,1],[2,3]]`、`[[4,5],[6,7]]`、`[[4,5],[6,7]]`；记录 RNG 三项逐值比较结果、实际包版本和来源。保存源码与基础环境前后清单；任何变化、缺包、导入错误、意外 CUDA 初始化或游标/RNG 不一致均失败，不忽略异常后继续。

预算：0 GPU、4 CPU、16 GiB RAM、总计 15 分钟；安装与原生子进程各最多 120 秒，完整小证据最多 10 MB。保留 wheel、overlay、完整日志、文件索引、前后清单与首个错误；完整证据按正常容量条件回传验收。本节通过后才准备真正导入 Trainer 的保存/恢复集成；多进程预取、完整 setup、FSDP、WAM/真实数据及 GPU/NPU 仍须分别验证。既有正式评测、SFT 和 Ascend 计数不因此增加。

**修正后现场状态（2026-11-15 07:51 UTC）：** 执行端报告 9.04 秒完成，调度终态 `COMPLETED / 0:0`；旧目录的限定入口诊断与八包共 1,407 个带 SHA256 的 RECORD 成员校验通过，六类状态登记、下一批 `[[4,5],[6,7]]` 恢复及三类 CPU RNG 精确比较通过。报告加载了 49 个真实 Awomo 模块；93 份固定源码和基础环境前后不变。节点 10 份原始证据共 302,262 bytes 已列索引，但完整原件尚未回传，维护者没有独立读取完整模块记录和旧目录诊断 JSON。该结果只支持执行端报告的导入与组件进展，不计完整 Trainer、checkpoint 集成、正式 SFT 或 Ascend 通过；首次失败保持原结论，不重跑。


## 16. 真正导入 Trainer 后的训练批次与 checkpoint 集成

本节为 2026-11-15 新增的独立 CPU 验证方案。执行端已报告单rank连续训练完成、恢复后第3次更新不一致；完整状态见第16.3节。第 15 节只验证导入与组件内存状态；本节实际调用 `Trainer._run_train_batch_group`、`_clip_gradients`、`save_checkpoint`、`load_checkpoint`，以及真实 StepScheduler、StatefulDataLoader、StatefulRNG 和 DCP。子类**仅实现 `train_step`**，其他上述方法必须与导入的 Trainer 方法是同一个对象。不从 AST 提取方法，不替换存储或优化器。

### 16.1 固定初始化与批次语义

使用第 15 节已经核验的独立 prefix，不再安装或下载。先将整个 prefix 的逐文件清单与此前冻结的清单字节/SHA256比较，再核全部安装成员和基础环境；执行后再次比较，禁止写坏共享推理环境或已有 overlay。子进程显式把 overlay 的 `lib/python3.11/site-packages` 加入 Python 搜索路径，因为 multiprocessing spawn 会恢复父进程的 `sys.path`，仅设置 `PYTHONPATH` 不足以保证它生效。

- 按第 14 节初始化 CPU/Gloo，分别验证 world size 1、2，每组先连续运行、再启动全新进程恢复，共四阶段、六个 rank 进程。单进程组仍使用真实 Gloo；两 rank 模型使用真实 DDP 梯度同步。
- 模型沿用第 14 节的 61 参数网络：4→7→3、Tanh、Dropout(.25)、两元素 unused 参数和整型 batches buffer。连续模型初始化 seed1541，恢复模型 seed9876，确保恢复真实改变参数。随后用实际 `StatefulRNG(2000+rank, ranked=False)` 初始化三类训练 RNG。
- 每 rank 有 24 行确定性数据。`ids=arange(24)+100*rank`；`inputs=(ids[:,None]+arange(4))/31`，`targets=(ids[:,None]-arange(3))/47`，浮点列 FP32。StatefulDataLoader 的 batch size2、shuffleFalse、workers0，**保留 generator=None**。每个优化器更新累积两个 microbatch。
- 使用实际 AdamW：lr.003、betas(.9,.95)、foreachFalse；实际 OptimizerParamScheduler 放在单元素列表中：max/init lr.003、min lr.0003、warmup0、linear decay5、weight decay恒定.01。最大梯度范数2.0，device为cpu，device_mesh=None，EMA禁用。
- StepScheduler：global batch=4×world size、local batch2、dp size=world size、max steps5、checkpoint周期2。通过其真实迭代器取得 batch group，再调用 Trainer 的真实批次方法，不自行重写 backward、clipping、optimizer或scheduler更新。

下面是本节唯一的模型任务子类；`micro_records` 在每个 batch group 开始前设为空列表，只记录比较材料，不修改训练计算。

```python
import random
import numpy as np
import torch
from awomo.training.trainer import Trainer

class TinyTrainer(Trainer):
    def train_step(self, batch, step, micro_idx):
        ids, inputs, targets = batch
        noise = torch.randn_like(inputs) * .02
        scale = float(np.random.uniform(.8, 1.2) + random.random() / 10)
        prediction = self.model(inputs + noise) * scale
        loss = ((prediction - targets) ** 2).mean()
        assert torch.isfinite(loss)
        self.micro_records.append({
            'ids': ids.clone(), 'inputs': inputs.clone(), 'targets': targets.clone(),
            'noise': noise.clone(), 'scale': scale, 'loss': loss.detach().clone(),
            'step_index': step, 'micro_index': micro_idx})
        return {'total_loss': loss, 'dynamics_loss': loss * .4, 'action_loss': loss * .6}

for name in ('save_checkpoint', 'load_checkpoint', '_load_checkpoint_tracked_state',
             '_run_train_batch_group', '_clip_gradients'):
    assert getattr(TinyTrainer, name) is getattr(Trainer, name)
```

手动构造 `Trainer(ConfigNode({'model': {'model_type': 'tiny-cpu-trainer-fixture'}}))` 并登记真实 model、optimizer、lr_scheduler、StepScheduler及两种rank状态。为避免生产辅助写入器使用固定 `/tmp` 名称造成作业间冲突，把 loader 和 RNG 属性分别命名为 `loader_<运行根摘要前20位>`、`rng_<同摘要>`；两者仍由真实类型检查自动登记，不能通过替身状态对象改变保存语义。

### 16.2 保存与新进程恢复的精确规则

1. 每组创建一个高编号的不完整目录 `epoch_0_step_9999`，保留诊断文件但没有 `.complete`。调用真实 `Trainer.load_checkpoint` 显式恢复该目录，必须在加载模型前拒绝，且模型和 RNG 不变；自动选择也不能选中它。
2. 连续完成五个更新，每个更新保存原始 PT oracle。保存全部参数/buffer、完整 optimizer state及param_groups、真实LR scheduler、三类RNG、实际loader状态、microbatch输入/噪声/target/scale/loss、全部非空裁剪前梯度、真实梯度范数和跨rank平均指标。逐tensor检查有限性，验证参数确实变化、unused参数没有获得Adam状态。
3. **原生计数约定：** StepScheduler先yield再递增；第二个已完成更新发生在内部step=1。此时调用 `save_checkpoint(epoch=0, step=1, ...)`，目录为 `epoch_0_step_1`，但真实 `step_scheduler.pt` 必须保存 `{'step':2,'epoch':0}`，表示下一更新从索引2开始。不能把目录名误当已完成更新数，也不能在循环外调用会加1的 `state_dict()` 来解释恢复后的计数。
4. 保存时传 `val_loss={'probe_metric':.25}`，要求调用后字典不变、losses.json完整，config、model、optim、loader/rng各rank状态、step_scheduler及合法完成标记齐全；LATEST与LOWEST_VAL都指向该目录。再次调用同一个已完整checkpoint的保存必须跳过，全部原始文件字节不变。
5. 全部连续进程退出后，用新模型与新对象调用真实 `load_checkpoint('LATEST')`。在创建新的loader迭代器**之前**，先比较所有模型/优化器/LR/RNG状态与第二步oracle，并要求实际step=2、epoch0。不能在此手动重置或补偿RNG。
6. 继续通过真实StepScheduler取得数据，重放完成更新3、4、5；逐叶精确比较上述完整记录。尤其核对第一次恢复后的实际数据ID与随机噪声：重新创建DataLoader迭代器可能消耗全局随机状态，本实验保留原生默认值来检验此风险。若不同即记录首次失败，不换generator、不放宽容差、不选择性忽略状态。
7. 验证恢复前后checkpoint原文件集合、大小和SHA256未改变。不同world size分别与自己的连续对照比较，不要求跨world size权重相同。

每个rank的逻辑更新5次，包含重放共8次；最多四阶段。首次失败立即停止后续阶段。预算为0GPU/4CPU/16GiB/15分钟，进程组30秒、阶段120秒、driver600秒；最多保留10MB完整小模型证据。只结束本诊断派生的超时子进程，保留原始失败、partial、oracle、DCP、逐rank日志和完整配置/源码/环境索引；完整回传仍受正常容量门槛约束。

**结论边界：** 这是实际Trainer方法的CPU集成，初始化由诊断显式构造，未调用CUDA-only `setup()` 或完整 `run_train_validation_loop()`。没有真实WAM/Human数据、FSDP/DTensor、异步DCP、CUDA/NPU和正式SFT信用。此前完成的诊断不重复执行，其原件验收和旧GPU准入要求不因本节而改变。

**完整 setup 的源码审计边界：** 它先构建模型/优化器/loader，再在恢复路径加载 RNG、步数与数据状态、优化器；完整训练循环之后才创建迭代器。通用 `load_checkpoint` 则先恢复登记的 loader/RNG/step 状态，再加载模型/优化器。因此，仅小模型DCP的“最后手动恢复RNG”顺序通过，不能替代这两种真实入口的验证。本节不修改这些顺序，先观察实际行为；CPU通过也不能绕过完整setup内的CUDA要求。

### 16.3 首轮现场失败与独立定位要求

执行时间为 **2026-11-15 08:43:24–08:43:42 UTC**，节点为 B300 GPU2，但本项申请并使用 **0 GPU、4 CPU、16 GiB**；调度终态 `FAILED / 1:0`，运行18秒，程序报告15.79秒。配置完全沿用第16.1–16.2节，没有修改默认loader generator、恢复顺序或比较容差。

| 检查 | 执行端报告的结果 |
|---|---|
| world size 1 连续分支 | 完成更新1–5；实际Trainer批次、梯度裁剪和保存方法已调用 |
| 不完整checkpoint | 连续与恢复进程均拒绝，异常类型为RuntimeError |
| 重复保存已完成checkpoint | 跳过，连续分支检查原文件字节不变 |
| 新进程恢复后、创建loader迭代器之前 | 模型、优化器、LR与RNG核心状态同更新2的oracle精确一致 |
| 恢复后的首次更新 | 第3次完成更新比较失败，首个报告位置为 `replay-update-3.core.model.layers.0.weight` |
| world size 2 | 因首个失败停止，未启动；不得写作通过或失败 |
| 源码和环境 | 95份固定源文件与预登记清单一致，基础环境和依赖prefix前后不变；自有进程已退出 |

节点保留21份原件，共316,762 bytes，包括连续分支五个oracle、真实DCP checkpoint、两阶段日志/JSON及逐文件索引。当前只收到执行端控制元数据、首错栈和文件摘要；完整原件受接收端磁盘容量门槛阻塞，**尚未独立读取全部张量并验收**。这里的失败结论有执行回执支持，但不把摘要当作完整payload交付。

首个报告的模型权重差异是比较器遇到的第一个不相等字段，**不是因果链中的最早差异**。它不能证明DataLoader迭代器消耗RNG就是根因。后续独立定位应保持上述固定模型、数据、seed和更新顺序，在新运行身份中依次记录：恢复刚完成、创建迭代器前后、取得每个microbatch后、生成noise/scale后、前向后、反向后及optimizer step后的实际状态。比较数据ID、输入、三类RNG、noise/scale、loss、裁剪前梯度及完整优化器状态，找到最早分歧。不得先补偿RNG、改generator或放宽容差再把结果当作原路径通过；任何修复另存差异，并重新验证连续与新进程重放。

本项不增加正式SFT、官方评测或Ascend通过计数。第16节的CPU方法集成仍未通过，完整setup、真实WAM/Human及NPU训练恢复保持待验。

### 16.4 首个恢复更新的分阶段观察方案

本项独立CPU诊断的执行端结果已返回，见本节末尾。模型、训练计算、seed、数据、loader默认generator与第16.1节相同；StepScheduler和LR schedule仍保留原来的5步终点，避免改变第3步的学习率。但本次只执行world size 1：连续更新1–3，保存完成更新2的checkpoint；随后新进程恢复并只执行更新3，共4次实际更新。不会运行更新4/5或world size 2。

**观察器不修复任何状态。** Trainer登记与保存的仍是真实StatefulDataLoader。只在传给StepScheduler的入口加一个转发对象：`__len__`返回真实loader长度，`__iter__`使用下面的生成器；不得替换sampler、generator、batch内容或checkpoint状态。所有迭代器和next调用均只发生一次。

```python
def observed_batches(loader, observe):
    observe('loader_iter_before')
    iterator = iter(loader)
    observe('loader_iter_after')
    while True:
        observe('loader_next_before')
        try:
            batch = next(iterator)
        except StopIteration:
            observe('loader_exhausted')
            return
        observe('loader_next_after', {'batch': batch})
        yield batch
```

`observe(point, payload=None)`记录当前StepScheduler内部step、三类完整RNG状态和payload的独立深拷贝。记录前后再次读取RNG并逐叶精确比较，观察本身不得消耗随机数；张量在记录前detach，不能保存仍连着计算图的对象。额外观察点为：每个microbatch开始、生成noise/scale后、前向及loss计算后、全部累积反向完成但裁剪前、优化器与LR更新后，以及保存和恢复前后。每阶段最多100条事件，事件原件保存在独立PT文件中，即使后续失败也保留。

连续分支的更新1–3必须与第16.3节原失败实验保留的完整oracle逐叶一致，先检查旧文件大小及SHA256，再加载比较，运行后再次核旧文件不变。这是观察器不改变原计算的对照；仅对比loss不足以通过。已有三份参考文件均为24,768 bytes，其SHA256依次为：

| 完成更新 | 原始oracle SHA256 |
|---|---|
| 1 | `794913cbb797bffae08b633fcfd5f1a423076a5616116fe9b93544c44d84c126` |
| 2 | `3659104b48a380a4c86ffa67b2d63399218ac264c81f297ff22779a99031b0e9` |
| 3 | `668fb69d7d41666e50803c0573b4a6041cb069a5a70a8f6854371e7384809ea5` |

这些摘要标识未随指南交付的历史原件，不是下载入口。独立Ascend实现需在自己的固定GPU对照环境先运行第16.1–16.2节，保留真实oracle，再用同一环境执行有观察器的对照；跨设备另按数值验收规则比较，不能要求新文件的序列化摘要与历史文件相同。

恢复分支分别报告迭代器创建前后Python、NumPy、Torch CPU及CUDA RNG字段是否相同；按事件顺序比较连续与恢复分支在内部step2的共同观察点。每个事件同时比较完整记录与单独的payload，防止先遇到RNG差异就看不到数据ID是否相同。分别记录保存前后核心状态是否不变、首个完整状态差异、数据差异、noise/scale差异、梯度差异和更新后差异。若观察完整但仍存在差异，只能称“诊断完成”，不能称恢复通过。任何源码、原始oracle、环境不符或连续计算偏离对照均立即停止。

预算沿用0GPU、4CPU、16GiB、15分钟及10MB输出上限；只读复用已校验依赖prefix，无安装或下载。此方案已完成93份Python语法检查，并通过AST比较确认模型及去除观察语句后的train_step计算不变；原生执行仍交由执行端完成，本地检查不代表现场通过。

**现场诊断结果（2026-11-15 09:28 UTC）：** 执行端报告15.91秒完成，连续与恢复两进程均退出0。连续更新1–3与前述原始oracle全部精确相同；保存前后核心状态不变，恢复后迭代前核心状态也相同。连续/恢复分别记录40/16个观察点。重新创建真实迭代器时，Python和NumPy RNG不变，**Torch CPU RNG发生变化**；两个实际microbatch的数据均精确相同，而noise、prediction、裁剪前梯度和更新后权重依次不同，恢复更新3仍未通过。CUDA状态为空，本项没有CUDA测试。

95份源码与基础环境核对通过，依赖prefix和三个旧oracle前后不变，自有进程已退出。节点21份新原件共1,181,548 bytes仍未完整回传；以上为已核对执行端元数据支持的定位，原始trace张量尚待独立读取。结果把首个观察到的差异定位到迭代器重建阶段，不能推及完整setup、其他worker配置或NPU。

### 16.5 零worker恢复保护候选与生产loader复验

已有生产loader在恢复后重建迭代器时保护Torch CPU RNG，但原条件同时要求`num_workers`为非零，因此漏掉第16.4节观察到的零worker路径。修复候选只去掉这个worker数量条件：正常首次创建迭代器仍保留原始随机数消耗；只有`load_state_dict`之后的首次重建会保存并恢复主进程Torch CPU RNG。以下给出完整核心实现，不能把补偿操作偷偷加入第16.4节的原诊断。

```python
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

class ResumeSafeStatefulDataLoader(StatefulDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resume_pending = False

    def load_state_dict(self, state):
        # 先完成本节下文的配置身份校验，再调用真实恢复。
        super().load_state_dict(state)
        self._resume_pending = True

    def __iter__(self):
        saved_rng = torch.get_rng_state() if self._resume_pending else None
        try:
            iterator = super().__iter__()
        finally:
            if saved_rng is not None:
                torch.set_rng_state(saved_rng)
        # 若创建失败，此行不执行，保留pending以免把失败当作完成恢复。
        self._resume_pending = False
        return iterator
```

生产RoboDojoDataLoader继续使用其真实TorchData状态读写。保存时额外附带配置身份：版本1、num_workers、prefetch_factor、persistent_workers，以及batch sampler的length、seed、local_batch_size、rank、world_size；恢复时复制状态字典、取出此身份并要求完全相等，之后把其余完整状态交给父类，不静默丢弃任何字段。该配置校验不能由上面的核心示例省略。

实际DenseBatchSampler使用独立`torch.Generator().manual_seed(seed+epoch)`生成全局randperm；按全局batch大小补齐，再reshape为`[-1, world_size, local_batch_size]`并取本rank连续分区。补齐不足时重复原索引到所需长度。其迭代器保存下一批position并检查恢复位置范围；sampler另保存epoch与data_step，初始offset为`max(0, data_step-epoch*len(sampler))`。这些状态仍由真实loader保存，不用一个手写计数器代替。

新的复验调用**真实RoboDojoDataLoader、DenseBatchSampler及Trainer方法**。本次全局数据有`24×world_size`行，ID为`arange(24×world_size)`，input/target公式沿用第16.1节，sampler seed1541、local batch2、workers0、每次更新累积2批。world size 1与2分别建立新的shuffle连续对照，各连续5更新、保存完成更新2、全新进程重放3–5。因为输入顺序已切换为实际生产sampler，本项不与第16.3节旧未shuffle轨迹比较；同一world size的完整连续/恢复记录仍须逐叶精确一致。没有第16.4节的观察转发对象。

为单独导入loader/sampler，数据集类的运行时import移到真实构建函数内部；类型注解使用延迟解析，静态检查保留类型导入。构建真实HDF5数据集时仍导入并检查实际数据集类，缺少Hydra/HDF5等依赖仍须失败，不能用此变更冒充完整数据环境就绪。

四个本地真实方法体检查覆盖：零worker恢复、多worker恢复、首次迭代保留原始随机消耗，以及重建异常时恢复RNG并保留pending。旧条件在其中两个零worker用例失败，候选四项通过；这些检查使用假RNG，未运行Torch。96份源码固定、94份Python语法通过；现场复验结果见下文。多worker真实预取、完整setup、真实Human/WAM和Ascend仍需分别验收。

**现场复验（2026-11-15 09:35 UTC）：** 执行端报告world size 1连续5更新与新进程恢复3–5全部精确通过；world size 2的两个rank也各完成连续5更新，但两个恢复rank都在创建loader迭代器前失败，位置为`restored-core-before-loader-iteration.optimizer.state`的字典键集合比较。整体调度终态为`FAILED / 1:0`，运行34秒，程序报告32.39秒。双进程恢复尚未到达本次RNG保护位置，不能据单进程结果推定双进程已通过。96份源码、基础环境与prefix前后不变，自有进程已退出；54份节点原件共754,657 bytes待完整回传。下一项独立修复候选见第16.6节。

### 16.6 DDP优化器状态名称匹配候选

源码审计发现，优化器加载末尾会删除checkpoint中不存在的参数状态。DCP的规范名称不带DDP包装层，而直接对DDP对象调用`named_parameters()`会多出包装前缀；用两者直接匹配，可能把已经恢复的有效状态误删。当前失败位置与这一问题一致，但仅凭字典键断言还不能认定现场因果链已完全验证。

候选修复保持DCP实际加载顺序，只在后续名称到Parameter对象的匹配中识别真正的`DistributedDataParallel`并访问其`.module`。底层Parameter身份不变，数字参数ID的fallback仍保留；不要对所有名字盲目删除字符串`module.`，因为真实模型也可能有同名子模块。完整清理规则如下：

```python
import torch

def prune_unrecorded_optimizer_state(model, optimizer, checkpoint_state_keys):
    prefixes = {key.rsplit('.', 1)[0] for key in checkpoint_state_keys}
    parameter_model = (model.module if isinstance(
        model, torch.nn.parallel.DistributedDataParallel) else model)
    recorded = {
        parameter for name, parameter in parameter_model.named_parameters()
        if 'optim.state.' + name.replace('_checkpoint_wrapped_module.', '') in prefixes
    }
    native_groups = optimizer.state_dict()['param_groups']
    for native, live in zip(native_groups, optimizer.param_groups, strict=True):
        for parameter_id, parameter in zip(native['params'], live['params'], strict=True):
            if f'optim.state.{parameter_id}' in prefixes:
                recorded.add(parameter)
    for _, parameter in parameter_model.named_parameters():
        if parameter not in recorded:
            optimizer.state.pop(parameter, None)
```

`checkpoint_state_keys`只接受经过原加载器完整性检查的`optim.state.*`键。其余模型/优化器元数据、dtype、shape及调度器验证不变，不能用此函数跳过前置检查。三项真实清理尾部的本地替身检查覆盖普通模型、DDP命名状态和DDP数字键fallback，均要求保留活动参数状态并删除未参与更新的状态；旧代码在DDP命名用例失败，候选三项通过。这些检查没有执行原生Torch。

后继现场复验固定97份源码和95份Python语法检查，保持第16.5节完全相同的模型、数据、shuffle、loader与预算，重新建立world size 1/2各自的连续5更新和新进程恢复3–5对照。原失败与全部原件保留，使用新运行身份，不放宽任何比较。

**修正后现场结果（2026-11-15 09:44:32–09:45:06 UTC）：** 执行端报告四个阶段、六个rank结果全部通过：world size 1/2各自连续更新1–5，保存完成更新2，再从新进程精确重放3–5；恢复后、迭代前核心状态也精确一致。调度终态`COMPLETED / 0:0`，运行34秒，程序报告约32.27秒。97份源码逐文件与固定版本及计划一致，基础环境和prefix前后不变，自有进程已退出。

节点54份原件共757,879 bytes仍待正常容量下完整回传；维护者已核结果元数据、源码、环境与索引，**尚未独立读取全部PT/DCP原件**。该结果支持执行端报告的生产loader/sampler及真实Trainer方法的CPU恢复进展，不等于完整Trainer setup、真实HDF5/Human/WAM、多worker真实预取、FSDP/DTensor、CUDA/NPU或正式SFT通过。接手者须保留前两项失败，并在自己的目标栈重新验证这些边界。
