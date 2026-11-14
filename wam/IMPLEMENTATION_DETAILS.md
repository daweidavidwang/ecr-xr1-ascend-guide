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

只有 metadata 的表示方式按已有存储协议统一；XML、物理状态、相机图像、scene request、instruction、layout/style 仍各自严格检查。保留原失败记录，使用新代码版本与新运行身份重新验收。修复已有 17 项本地 CPU 合同测试通过，包括固定原生方法重现、标量类型/数组顺序/微小数值变化仍拒绝、非有限值和非法键拒绝；两模型修订版的现场 CPU 前置也已分别通过 17 项测试、固定源码/环境与原生客户端检查；修订版 GPU 结果仍待验收。此修复不应用稳定 Counter 候选，也不改变原 v1 场景身份或策略语义。
