# 数据下载入口与接手要求

这是 Ascend 接手 agent 查找训练输入的入口。此前公开版把真实仓库地址替换成了占位符；本页补齐已确认的仓库、版本、文件位置和可从本仓库执行的下载命令。核对范围为 2026-11-10 的已完成上传回执。

**Human 和 R1 可定位并下载；R2 recovery 与固定 20k 父模型还缺经核验的云路径。** 本页公开的是访问地址和校验元数据，数据仓库仍然私有，需要有读取权限的 ModelScope 凭据。链接不能匿名访问，不表示文件不存在。

| 必需输入 | 去哪里找 | 状态 |
|---|---|---|
| Human 十任务 | 下表十个 ModelScope 仓库；仓库根目录的 `lerobot.tar` 和 `SHA256SUMS` | 固定 revision 与哈希已提供 |
| 第一轮 recovery | [R1 数据仓库](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-g16-recovery-unified-9task-v1-20261022)；主目录 `v1-20261020/` | 按冻结云映射逐文件恢复，共 205,003 文件、49,789,441,549 B |
| R1 全量文件的实际位置 | [归档映射仓库](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-g16-archive-by-model-v2-20261023) 中的 `source-cloud-map.jsonl.gz`，准确路径和版本见下文 | 已有完整映射；含历史位置与增量位置 |
| 第二轮 recovery | 尚未取得对应 ModelScope repo / revision / 文件映射 | 缺失输入，不能猜地址或拿 rollout 代替 |
| A3 s1 累计 20k 父模型 | 尚未取得完整 22 个 HF 文件的云端位置 | 缺失输入，不能拿最终 30k best 或 A3 s2 代替 |

## 1 Human 仓库与版本

每个仓库下载 `lerobot.tar` 和 `SHA256SUMS`。固定文件哈希和目标路径在 [human10-download.jsonl](downloads/human10-download.jsonl)；根目录映射和计数在 [human10.json](downloads/human10.json)。

| 任务 | ModelScope dataset 仓库 | revision |
|---|---|---|
| TurnOnSinkFaucet | [dvdnh2026/ds-robocasa365-target-turnonsinkfaucet-human-d9e51f2a4069](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-turnonsinkfaucet-human-d9e51f2a4069) | `941edf30d47ac657f743d898999ff09b9b419fc0` |
| TurnOffStove | [dvdnh2026/ds-robocasa365-target-turnoffstove-human-51f148772b84](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-turnoffstove-human-51f148772b84) | `35aec5baa9032b2febfa884714980d7f82543074` |
| CoffeeSetupMug | [dvdnh2026/ds-robocasa365-target-coffeesetupmug-human-005055b941c3](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-coffeesetupmug-human-005055b941c3) | `98f55b23734d31d482e5f2a5d9c86ba8a1b5ace9` |
| NavigateKitchen | [dvdnh2026/ds-robocasa365-target-navigatekitchen-human-bff6c60a3e27](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-navigatekitchen-human-bff6c60a3e27) | `4195b6828c73fb4836cb37e3a923f2a9b2029335` |
| KettleBoiling | [dvdnh2026/ds-robocasa365-target-kettleboiling-human-ed11182fe5c0](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-kettleboiling-human-ed11182fe5c0) | `15ba07ce3ea8d8160cdb379a42c37362e57622c8` |
| LoadDishwasher | [dvdnh2026/ds-robocasa365-target-loaddishwasher-human-7d743c2cce3b](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-loaddishwasher-human-7d743c2cce3b) | `2f6240d9f000fbb5e8c140d5fc3d25c536d1075e` |
| SteamInMicrowave | [dvdnh2026/ds-robocasa365-target-steaminmicrowave-human-d268e63968dd](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-steaminmicrowave-human-d268e63968dd) | `53269267066597594dac76c4d4582cc481e7dad8` |
| WashFruitColander | [dvdnh2026/ds-robocasa365-target-washfruitcolander-human-22bf894534d4](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-washfruitcolander-human-22bf894534d4) | `f797c513d7f88e1d5ffff721e44f6693bf1d94a1` |
| ArrangeTea | [dvdnh2026/ds-robocasa365-target-arrangetea-human-a59bb61b515c](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-arrangetea-human-a59bb61b515c) | `e709f9684324e66d86c3aba2134fdcdd05126c89` |
| MakeIceLemonade | [dvdnh2026/ds-robocasa365-target-makeicelemonade-human-885a1174f656](https://modelscope.cn/datasets/dvdnh2026/ds-robocasa365-target-makeicelemonade-human-885a1174f656) | `2a114677f94c0c5ecc2f122a928b00a678edcedc` |

## 2 从本公开仓库开始下载

这些命令只依赖本 GitHub 仓库中的 `downloads/`，不要求先取得私有训练源码包。下载和内容校验通过后，再进行完整交接包的输入检查。

```bash
git clone https://github.com/daweidavidwang/ecr-xr1-ascend-guide.git
cd ecr-xr1-ascend-guide
set -euo pipefail
export ECR_ROOT=/data/ecr-ascend  # 换成集群上有权限且空间充足的目录
python3.12 -m venv .venv-download
.venv-download/bin/python -m pip install 'modelscope-hub==0.6.1'
read -rsp 'ModelScope read token: ' MODELSCOPE_API_TOKEN
export MODELSCOPE_API_TOKEN

.venv-download/bin/python downloads/download_files.py \
  downloads/human10-download.jsonl --root "$ECR_ROOT" --workers 4
.venv-download/bin/python downloads/unpack_human.py --root "$ECR_ROOT"
```

若已经 clone，先 `git pull --ff-only`，从 `set -euo pipefail` 开始。token 使用该集群正常的凭据提供方式，不写进仓库或日志。401/403 表示要检查读取权限；私有资源也可能返回 404，不能据此断言数据不存在。

## 3 R1 的逐文件位置与完整下载

归档映射仓库：`dvdnh2026/ds-ecr-g16-archive-by-model-v2-20261023`，`repo_type=dataset`。

- 固定 metadata revision：`d530e9b77c290307643c517aaf09dab27e1b2480`
- 仓库内目录：`snapshots/20261109/ecr-g16-ms-incremental-r1/_manifests/`
- 关键文件：`source-cloud-map.jsonl.gz`
- 关键文件 SHA256：`8dac3099cfad185ad859e308eb2af1af29cbf4834a23f1cec2ff3e85901f24a3`

先取约 40.39 MB 的五个 metadata 文件，再从经过哈希核验的 map 中生成 R1 下载表。不要将 metadata revision 填到 R1 数据仓库：旧数据文件使用 `master` 加逐文件 SHA256 和字节数锁定。

```bash
.venv-download/bin/python downloads/download_files.py \
  downloads/archive-metadata-download.jsonl --root "$ECR_ROOT" --workers 4
.venv-download/bin/python downloads/make_round1_map.py \
  --source "$ECR_ROOT/cloud-manifests/ecr-g16-ms-incremental-r1/source-cloud-map.jsonl.gz" \
  --output "$ECR_ROOT/input-maps/round1-download.jsonl"
.venv-download/bin/python downloads/download_files.py \
  "$ECR_ROOT/input-maps/round1-download.jsonl" --root "$ECR_ROOT" --workers 8
unset MODELSCOPE_API_TOKEN
```

输出恢复到 `$ECR_ROOT/datasets/ecr-recovery-unified-9task/v1-20261020/`，保留 18 个 LeRobot 根目录，包括 π0.5 与 XR‑1 两个来源。每个文件核验大小和 SHA256；已有不同内容不会被覆盖。生成器核对固定云映射哈希、205,003 个唯一文件和总字节数。当前公开工具已做离线映射对账，但未在 Ascend 集群进行私有下载验收；首次运行需保留实际下载回执。

## 4 小米原版模型与缺失输入

公开原版模型入口：[XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365](https://modelscope.cn/models/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365)。仅做 Fresh、完整 30k 复现或算子诊断时需要它；原版不是本次 Continue 的父模型。

```bash
.venv-download/bin/python downloads/download_files.py \
  downloads/xiaomi-base-download.jsonl --root "$ECR_ROOT" --workers 4
```

R2 的目标相对路径为 `datasets/ecr-recovery-unified-9task/v3-round2-20261101/`；父模型的目标相对路径为 `models/xr1-a3-s1-cont-20k/`。这些是本机恢复目录，不是下载地址。当前缺失的云端映射必须由上传方补交；验收内容见 [MISSING_INPUTS.md](MISSING_INPUTS.md)。

## 5 完整训练交接包在哪里

训练源码、NPU 适配工具、配置生成器和 GPU 比较基准在 owner 持有的 **`XR1-Ascend训练复现指南-v3.zip`** 中。该 ZIP 不在本 GitHub 仓库，也没有已发布的 HTTP 下载链接；owner 需要通过双方已有的文件传输渠道把文件交给集群 agent。不要让接手 agent 在 GitHub 的 Releases 中寻找不存在的附件。

- ZIP 大小：11,563,326 bytes
- ZIP SHA256：`9e52a2342ad542af952376326bafdecd2fc43704301efbe461922e71fabe1466`
- 解压后先读 `START_HERE.md`，运行包内 `sha256sum -c SHA256SUMS`。
- 公开仓库提供数据定位与基础下载工具；完整包提供固定实验执行与验收工具。两个入口中的已知下载 pins 一致。

**给接手 agent 的明确任务：** 先按本页下载 Human 与 R1、核验目录和文件，并做环境准备；取得完整 v3 包后继续适配。仅在 R2、固定 20k 父模型和 NPU 验收均通过后启动正式 Continue 训练。缺项写出确切名称，不改成另一条实验链。
