# 当前源码、checkpoint 与交付边界

返回 [Target50交接](../TARGET50_HANDOFF.md)。科学实验状态保留2026-11-13原交接快照；源码云端交付更新于 **2026-11-13 22:15 UTC / 17:15 Toronto**。

## 1. 新的私有源码补充包：已上传并完整回下载核验

**新的Target50源码ZIP已交付ModelScope。** 现有授权账号 `dvdnh2026` 实测可读取；仓库仍为私有，公开GitHub只包含文档与下载元信息。下方链接不是匿名公开下载，Ascend环境需通过平台secret或隐藏输入提供现有账号的有效读取凭据。

- [固定版本ZIP下载（需登录授权账号）](https://modelscope.cn/api/v1/datasets/dvdnh2026/ds-ecr-g16-archive-by-model-v2-20261023/repo?Revision=26401275c1f3cfbaeb621e73f682c93c3a70e9d8&FilePath=handoffs%2Fxr1-target50-ascend-20261113-v1-be53dddf20f6%2Fxr1-target50-ascend-20261113-v1.zip)
- [ModelScope私有仓库](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-g16-archive-by-model-v2-20261023)
- [可执行下载清单](source-bundle-download.jsonl) · [验收摘要](source-delivery.json)

| 字段 | 固定值 |
|---|---|
| 文件名 | `xr1-target50-ascend-20261113-v1.zip` |
| 大小 | **329,314 bytes** |
| SHA256 | `be53dddf20f67763a873d9a9998c6653b32ba6ba17c26db1fde8ba037658e9b7` |
| ModelScope repo / type | `dvdnh2026/ds-ecr-g16-archive-by-model-v2-20261023` / `dataset` |
| 固定云revision | `26401275c1f3cfbaeb621e73f682c93c3a70e9d8` |
| 仓库内路径 | `handoffs/xr1-target50-ascend-20261113-v1-be53dddf20f6/xr1-target50-ascend-20261113-v1.zip` |
| 私有源快照 | `4bf26d9831a355651b820eb116d8b02ea44f300d` |
| 文件验收 | ZIP共104成员，103项内部SHA全部通过；其中101个source文件 |
| 内容 | Target50 Human reader/sampler/trainer/restart、B300运行参考、评测与采集准备代码、必要历史loss/数据混合实现；SOURCE_MANIFEST和SHA256SUMS |
| 不包含 | 数据载荷、权重、凭据、git历史、已验证NPU实现 |

### 在Ascend下载、校验、解压

先取得最新公开GitHub checkout：已有checkout执行 `git pull --ff-only`，首次使用按[DATA.md](DATA.md) clone。下面从该checkout根目录执行；`ECR_ROOT` 沿用数据下载所选目录。数据/模型下载环境与训练环境分开。

```bash
set -euo pipefail
export ECR_GUIDE="${ECR_GUIDE:-$PWD}"
export ECR_ROOT="${ECR_ROOT:-$HOME/ecr-target50-ascend}"
cd "$ECR_GUIDE"
test -f TARGET50_HANDOFF.md
sha256sum -c PUBLIC_SHA256SUMS
mkdir -p "$ECR_ROOT/envs" "$ECR_ROOT/private"
if [ ! -x "$ECR_ROOT/envs/model-download/bin/python" ]; then
  python3.12 -m venv "$ECR_ROOT/envs/model-download"
fi
"$ECR_ROOT/envs/model-download/bin/python" -m pip install 'modelscope-hub==0.6.1'

# 优先由平台secret注入。需要交互输入时不回显；不要将token写进命令参数或文档。
if [ -z "${MODELSCOPE_API_TOKEN:-}" ]; then
  read -rsp 'ModelScope read token (hidden): ' MODELSCOPE_API_TOKEN
  printf '\n'
fi
export MODELSCOPE_API_TOKEN
"$ECR_ROOT/envs/model-download/bin/python" downloads/download_files.py \
  target50/source-bundle-download.jsonl --root "$ECR_ROOT" --workers 1

printf '%s  %s\n' \
  be53dddf20f67763a873d9a9998c6653b32ba6ba17c26db1fde8ba037658e9b7 \
  "$ECR_ROOT/private/xr1-target50-ascend-20261113-v1.zip" | sha256sum -c -
export ECR_TARGET50_BUNDLE="$ECR_ROOT/private/xr1-target50-ascend-20261113-v1"
if [ ! -e "$ECR_TARGET50_BUNDLE" ]; then
  python3 -m zipfile -e "$ECR_ROOT/private/xr1-target50-ascend-20261113-v1.zip" "$ECR_ROOT/private"
fi
cd "$ECR_TARGET50_BUNDLE"
sha256sum -c SHA256SUMS
export ECR_TARGET50_SOURCE="$ECR_TARGET50_BUNDLE/source"
# 保留原ZIP与未改动源码，另建私有working copy做NPU适配。
```

SDK下载器逐项核验固定版本、329,314 bytes和外层SHA；已有不同文件不会被覆盖。下载回执在 `$ECR_ROOT/download-verified-*.json`。**该下载器的 `--check-only` 只检查清单格式，不下载或验收本地载荷**，不能代替上面的实际下载。平台没有Python3.12时可用独立Python3.10+环境；不要修改训练环境或系统驱动。

SDK命令是推荐入口；浏览器直链如提示登录，请使用现有授权账号，不要把token拼进URL。另一台机器的网页登录状态不会自动带到Ascend进程。本次未将凭据写进文档、源码ZIP或GitHub，也未更改仓库公开性。

### 已完成的云端验收与范围

Monitor在上述固定revision从空缓存通过网络下载完整ZIP，外层SHA和103项内部SHA全部一致；owner再独立读取该下载副本，复核104个成员、CRC和全部内部SHA。源ZIP及旧包均保留。此前缺凭据状态已解除，历史blocked记录不代表当前交付状态；机器可读状态以[source-delivery.json](source-delivery.json)为准。

这是**新的Target50包**，旧10-task v3 ZIP仍保留，不互相替换。新包里的采集研究代码存在未完成部分；文件存在不意味着已经实机验收。继续阅读[评测与录制状态](EVALUATION_AND_ROLLOUT.md)及[Ascend适配步骤](TRAINING_AND_ASCEND.md)。本次交付不宣称已完成NPU训练或仿真录制验收。

## 2. 源码和运行版本关系

| 对象 | revision / 身份 |
|---|---|
| XR上游模型/评测辅助代码 | `0dd7aef8dc87296246aae812a1f59ccb708e5546` |
| RoboCasa | `456174f62b89b8fca99eaaf33949c29fec9cfc2a` |
| 实际Human50核心11文件 | `d0fcba205f5dcfdd502159ae8d6cdd8a80b17b59` |
| 1000→10000衔接controller | `288259d0337c79e6b68272a7c826faa2e4c29322` |
| 10000→19200衔接controller | `074f1755dcb2055ea397c5dd9797d14919fd285b` |
| 当前完整准备源快照 | `4bf26d9831a355651b820eb116d8b02ea44f300d` |

私有maintained repo为 `vla_sft`，工作分支 `work/ecr-target50-20261112`；这里只给身份，不承诺远端分支已经包含上述commit。不要依赖公开指南能克隆这个私有分支。补充包按commit导出必要文件，不包含历史；`SOURCE_MANIFEST.json`对每个文件记录SHA/bytes。训练时再保存实际run contract，防止后来准备代码替代原运行核心。

## 3. GPU参考模型路径

两台B300项目根相同为 `/data/ubuntu/embodiedclaw-robocasa`，但磁盘独立。以下是源节点路径，接手者不会自动获得挂载或访问权限。

| 模型 | 路径（相对项目根） | 状态 |
|---|---|---|
| original base | `models/rc365_xr1_20261006/Xiaomi-Robotics-1-RoboCasa365` | 22文件pins，与公开base下载清单一致 |
| H50 s1 step10000 | `runs/target50_human_20261113/s1_segment1000_10000_r1/restart/step_10000` | 10文件，59,109,513,074 bytes，完整DCP已验 |
| H50 s2 step10000 | `runs/target50_human_20261113/s2_segment1000_10000_r1/restart/step_10000` | 10文件，59,109,512,498 bytes，完整DCP已验 |
| H50 19200/33600/48000 HF | 尚未产生 | 没有best可冻结 |
| Target50 R1/R1-cont/R2/R3 | 尚未训练 | 依赖新recovery生产与验收 |

step10000 `COMPLETE.json` SHA：

- s1：`ae8b372ba735b018731b89e757bfff92aba064fc7da17d2a3a09fee9ec1aede5`
- s2：`246138cf0184ed7d24a1998023264b1b7d857ec8e3b1a9be819ef4fb20424b3c`

上述是完整性manifest的SHA，不是单一权重文件SHA。完整DCP还需逐个payload检查。B300原件存在不等于独立g16或ModelScope备份已经完成。Ascend从base重训无需先复制这些中间状态。

训练数据根 `datasets/target50_human_v3_20261112`；实际GPU配置在根下 `human_s1.json` / `human_s2.json`、`data.json`、`readiness.json`。GPU环境 `envs/target50_xr1_cu130_r2`；模拟器客户端 `envs/target50_robocasa_client_r1`。这些环境路径仅说明来源，不是Ascend可复用的venv。

## 4. 当前排队和未完成项

20:38 UTC回执：B30016卡被其他项目占用，ECR训练GPU=0；`358`为1GPU渲染读回诊断，`359`/`383`分别为s1/s2的4GPU续训，依赖`afterany:358`。无可信获卡时间；获卡后到19200约9.9h/10.2h纯更新，另加恢复IO。这是GPU队列，不要求Ascend等待它才开始Human SFT。

π0.5为低优先级并行准备：数据/部分CPU输入和checkpoint边界检查已做，真实模型GPU训练恢复还未验收。本次Ascend交接主线是XR‑1；π0.5不借用XR参数直接启动。

接手者首份状态回执应明确：Human50下载/内容验收是否完成、私有新包是否收到、实际Ascend配置、数值与恢复gate、资源布局、首个稳定step-time、下一项实验及条件ETA。不要把“下载完成”称为“训练已开始”。
