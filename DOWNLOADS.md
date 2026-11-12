# 数据下载入口与接手要求

这是 Ascend 接手 agent 查找训练输入的入口。此前公开版把真实仓库地址替换成了占位符；本页补齐已确认的仓库、版本、文件位置和可从本仓库执行的下载命令。数据映射依据 2026-11-10 完成回执；2026-11-12 已补齐完整 ZIP 的私有云端入口，并实测接手下载流程。完整状态见 [HANDOFF_STATUS.md](HANDOFF_STATUS.md)。

**Human 和 R1 可定位并下载；R2 recovery 与固定 20k 父模型还缺经核验的云路径。** 本页公开的是访问地址和校验元数据，数据仓库仍然私有，需要有读取权限的 ModelScope 凭据。链接不能匿名访问，不表示文件不存在。

| 必需输入 | 去哪里找 | 状态 |
|---|---|---|
| Human 十任务 | 下表十个 ModelScope 仓库；仓库根目录的 `lerobot.tar` 和 `SHA256SUMS` | 固定 revision 与哈希已提供 |
| 第一轮 recovery | [R1 数据仓库](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-g16-recovery-unified-9task-v1-20261022)；主目录 `v1-20261020/` | 按冻结云映射逐文件恢复，共 205,003 文件、49,789,441,549 B |
| R1 全量文件的实际位置 | [归档映射仓库](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-g16-archive-by-model-v2-20261023) 中的 `source-cloud-map.jsonl.gz`，准确路径和版本见下文 | 已有完整映射；含历史位置与增量位置 |
| 第二轮 recovery | 尚未取得对应 ModelScope repo / revision / 文件映射 | 缺失输入，不能猜地址或拿 rollout 代替 |
| A3 s1 累计 20k 父模型 | 尚未取得完整 22 个 HF 文件的云端位置 | 缺失输入，不能拿最终 30k best 或 A3 s2 代替 |

## 0 先取得完整包并验证读取权限

完整 ZIP 现在可以从 ModelScope 下载，按本文第 5 节执行。只有 GitHub 仓库时也可先进行小文件访问检查，避免开始大文件下载后才发现无权限。这里使用的是接手 agent 所在机器的凭据；网页登录和另一台机器的登录状态不会替代它。

```bash
git clone https://github.com/daweidavidwang/ecr-xr1-ascend-guide.git
cd ecr-xr1-ascend-guide
set -euo pipefail
sha256sum -c PUBLIC_SHA256SUMS
export ECR_GUIDE="$PWD"
export ECR_ROOT="${ECR_ROOT:-$HOME/ecr-ascend}"
mkdir -p "$ECR_ROOT"
python3.12 -m venv .venv-download
.venv-download/bin/python -m pip install 'modelscope-hub==0.6.1'
read -rsp 'ModelScope read token: ' MODELSCOPE_API_TOKEN; echo
export MODELSCOPE_API_TOKEN
.venv-download/bin/python downloads/download_files.py \
  downloads/access-check.jsonl --root "$ECR_ROOT" --workers 4
```

若已有 checkout，先 `git pull --ff-only`，不重复 clone；平台未提供 Python 3.12 时，下载工具可用 Python 3.10+ 的独立环境。本次实际验证使用 Python 3.11 与 `modelscope-hub==0.6.1`；它不决定训练环境的 Python 版本。

访问检查读取 12 个私有仓库的小文件：十个 Human 仓库的 `SHA256SUMS`、R1 的一个 100-byte metadata 文件、归档仓库的一个固定 metadata 文件，全部核验冻结哈希。它证明所列入口当前可访问，不是所有训练数据完整性验收。完整 ZIP 另按第 5 节实下载校验。成功回执为 `$ECR_ROOT/download-verified-*.json`；只读凭据通过平台的 secret 注入或上面的不回显输入提供，不把 token 写入 GitHub、聊天、URL 或命令行参数。

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

## 2 从本公开仓库开始下载 Human

完成第 0 节后，从 GitHub checkout 执行。若刚完成 ZIP 解压，下面的 `cd` 会切回正确目录；按需重新提供读取凭据。

```bash
cd "$ECR_GUIDE"
if [ -z "${MODELSCOPE_API_TOKEN:-}" ]; then
  read -rsp 'ModelScope read token: ' MODELSCOPE_API_TOKEN; echo
fi
export MODELSCOPE_API_TOKEN
.venv-download/bin/python downloads/download_files.py \
  downloads/human10-download.jsonl --root "$ECR_ROOT" --workers 4
.venv-download/bin/python downloads/unpack_human.py --root "$ECR_ROOT"
```

下载和内容校验通过后，再进行完整交接包的输入检查。`ECR_ROOT` 沿用第 0 节选定的位置，不在中途切换恢复目录。

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
```

输出恢复到 `$ECR_ROOT/datasets/ecr-recovery-unified-9task/v1-20261020/`，保留 18 个 LeRobot 根目录，包括 π0.5 与 XR‑1 两个来源。每个文件核验大小和 SHA256；已有不同内容不会被覆盖。生成器核对固定云映射哈希、205,003 个唯一文件和总字节数。当前公开工具已做离线映射对账和本机小文件访问实测，尚未在 Ascend 集群完成全量数据下载验收；首次运行需保留实际下载回执。

## 4 小米原版模型与缺失输入

公开原版模型入口：[XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365](https://modelscope.cn/models/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365)。仅做 Fresh、完整 30k 复现或算子诊断时需要它；原版不是本次 Continue 的父模型。

```bash
cd "$ECR_GUIDE"
if [ -z "${MODELSCOPE_API_TOKEN:-}" ]; then
  read -rsp 'ModelScope read token: ' MODELSCOPE_API_TOKEN; echo
fi
export MODELSCOPE_API_TOKEN
.venv-download/bin/python downloads/download_files.py \
  downloads/xiaomi-base-download.jsonl --root "$ECR_ROOT" --workers 4
```

R2 的目标相对路径为 `datasets/ecr-recovery-unified-9task/v3-round2-20261101/`；父模型的目标相对路径为 `models/xr1-a3-s1-cont-20k/`。这些是本机恢复目录，不是下载地址。当前缺失的云端映射必须由上传方补交；验收内容见 [MISSING_INPUTS.md](MISSING_INPUTS.md)。

## 5 完整训练交接包：已上传，按固定版本下载

**2026-11-12 已完成上传和完整回下载校验。** 文件保持原 v3 字节不变，包含训练源码、NPU 适配候选生成器、执行工具、配置和 GPU 比较证据；不包含 Human/R1/R2 大数据和 20k 父模型权重。

- [ModelScope 私有仓库](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-g16-archive-by-model-v2-20261023)
- [完整 v3 ZIP 固定版本下载链接](https://modelscope.cn/api/v1/datasets/dvdnh2026/ds-ecr-g16-archive-by-model-v2-20261023/repo?Revision=6bbf6a71afc2c39f78fc3e2a35e584c67be875cf&FilePath=handoffs%2Fxr1-ascend-v3-20261112-9e52a2342ad5%2FXR1-Ascend%E8%AE%AD%E7%BB%83%E5%A4%8D%E7%8E%B0%E6%8C%87%E5%8D%97-v3.zip)（浏览器需登录有权限的账号；agent 推荐用下方命令）
- 仓库：`dvdnh2026/ds-ecr-g16-archive-by-model-v2-20261023`；`repo_type=dataset`
- revision：`6bbf6a71afc2c39f78fc3e2a35e584c67be875cf`
- 仓库内文件：`handoffs/xr1-ascend-v3-20261112-9e52a2342ad5/XR1-Ascend训练复现指南-v3.zip`
- 文件大小：`11563326` bytes
- SHA256：`9e52a2342ad542af952376326bafdecd2fc43704301efbe461922e71fabe1466`
- 机器可读下载清单：[handoff-v3-download.jsonl](downloads/handoff-v3-download.jsonl)

从本 GitHub checkout 执行，使用第 0 节的下载环境和凭据：

```bash
cd "$ECR_GUIDE"
if [ -z "${MODELSCOPE_API_TOKEN:-}" ]; then
  read -rsp 'ModelScope read token: ' MODELSCOPE_API_TOKEN; echo
fi
export MODELSCOPE_API_TOKEN
.venv-download/bin/python downloads/download_files.py \
  downloads/handoff-v3-download.jsonl --root "$ECR_ROOT" --workers 1
unset MODELSCOPE_API_TOKEN

# 只有首次解压执行这两行；已有目录时直接进入目录做校验，勿覆盖修改。
test ! -e "$ECR_ROOT/handoff/xr1-ascend-reproduction-20261110-v3"
.venv-download/bin/python -m zipfile -e \
  "$ECR_ROOT/handoff/XR1-Ascend训练复现指南-v3.zip" "$ECR_ROOT/handoff"

export ECR_BUNDLE="$ECR_ROOT/handoff/xr1-ascend-reproduction-20261110-v3"
cd "$ECR_BUNDLE"
sha256sum -c SHA256SUMS > bundle-check.log
```

下载器核验 ZIP 的大小和 SHA256；解压后的 `SHA256SUMS` 再核验包内文件。完成后按 GitHub 最新 [START_HERE.md](START_HERE.md) 执行，所有 `scripts/`、`manifests/`、`reference/` 相对路径均在 `$ECR_BUNDLE` 内解析。公开仓库中的 `downloads/` 命令则在 `$ECR_GUIDE` 内执行。包内文档冻结于 2026-11-10；云端访问和交接状态以本仓库的 2026-11-12 更新为准，原训练配方和参考源码保持不变。

### 权限与常见失败

| 现象 | 检查和处理 |
|---|---|
| GitHub 可打开，但 ZIP 或数据无法下载 | GitHub 权限不包含 ModelScope 权限。由仓库 owner 给接手者使用的账号授予读取权限，或通过集群 secret 机制提供已授权的读取凭据 |
| 401 / 403，或私有链接返回 404 | 先核验凭据是否有效、账号是否被授权、repo_type 是否为 `dataset`，再核验固定 revision 和路径；单次 404 不能证明文件不存在 |
| ZIP 可以下载，Human 或 R1 拒绝访问 | ZIP 所在归档仓库、R1 仓库和十个 Human 仓库权限独立；需全部通过第 0 节的小文件访问检查 |
| 命令报 SDK 参数错误 | 确认实际执行的是独立下载环境的 Python，且 `modelscope-hub` 精确为 `0.6.1`；不要直接使用旧系统 SDK |
| 文件 SHA256 不符或已有路径内容不同 | 保留错误和原文件，使用新的恢复目录排查；不覆盖或接受不同版本 |
| 只能访问 GitHub 的云 agent | 需为其运行环境提供 ModelScope 读取凭据及外网访问；这个私有 URL 不是匿名分享链接，仓库权限并未因本次上传自动授予其他账号 |
| 找不到 `scripts/` 或 `reference/` | 先下载并解压整个 ZIP，切换到 `$ECR_BUNDLE`；只 clone 公开仓库不足以运行训练 |

**给接手 agent 的执行顺序：** 先做权限探测和 ZIP 校验，下载 Human/R1 并做环境准备；按 [HANDOFF_STATUS.md](HANDOFF_STATUS.md) 逐项关闭 R2、20k 父模型、NPU 验收和 GPU 评测环境缺口，再启动对应阶段。ZIP 的访问问题已解决，不代表训练输入和硬件验收已全部完成。
