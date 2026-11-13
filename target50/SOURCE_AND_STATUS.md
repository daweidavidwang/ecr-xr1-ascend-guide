# 当前源码、checkpoint 与交付边界

返回 [Target50交接](../TARGET50_HANDOFF.md)。时间：2026-11-13；本页是交接时快照，不是实时调度页。

## 1. 新的私有源码补充包

**包已在owner工作站生成；尚无已验证的云端下载地址。请由owner通过私有渠道交给接手者。** 不要把旧ModelScope v3 ZIP的下载链接改名后当作Target50包。公开GitHub不包含私有训练实现或git历史。

| 字段 | 固定值 |
|---|---|
| 文件名 | `xr1-target50-ascend-20261113-v1.zip` |
| 大小 | **329,314 bytes** |
| SHA256 | `be53dddf20f67763a873d9a9998c6653b32ba6ba17c26db1fde8ba037658e9b7` |
| 私有源快照 | `4bf26d9831a355651b820eb116d8b02ea44f300d` |
| source文件数 | 101 |
| 内容 | Target50 Human reader/sampler/trainer/restart、B300运行参考、评测与采集准备代码、必要历史loss/数据混合实现；包内SOURCE_MANIFEST和SHA256SUMS |
| 不包含 | 数据载荷、权重、凭据、git历史、已验证NPU实现 |
| 云下载 | **未交付；不可填写猜测URL/revision** |

owner本地交付文件：

```text
/home/eai/Documents/Codex/2026-10-26/files-pasted-by-the-user-embodiedclaw/outputs/handoffs/xr1-target50-ascend-20261113-v1.zip
```

这是owner工作站地址，不是Ascend下载URL。接手者收到包后放在私有目录；校验再解压，不上传到本公开仓库：

```bash
# 在接收ZIP的私有目录执行。
printf '%s  %s\n' \
  be53dddf20f67763a873d9a9998c6653b32ba6ba17c26db1fde8ba037658e9b7 \
  xr1-target50-ascend-20261113-v1.zip | sha256sum -c -
test ! -e "$ECR_ROOT/private/xr1-target50-ascend-20261113-v1"
mkdir -p "$ECR_ROOT/private"
python3 -m zipfile -e xr1-target50-ascend-20261113-v1.zip "$ECR_ROOT/private"
export ECR_TARGET50_BUNDLE="$ECR_ROOT/private/xr1-target50-ascend-20261113-v1"
cd "$ECR_TARGET50_BUNDLE"
sha256sum -c SHA256SUMS
export ECR_TARGET50_SOURCE="$ECR_TARGET50_BUNDLE/source"
# 保留这个原始包，另建私有working copy做NPU适配。
```

新私有源包还包含未完成的采集研究代码；存在文件不意味着它已经实机验收。读 [评测状态](EVALUATION_AND_ROLLOUT.md) 后再选择生产入口。

当前Target50独立私有归档路线此前回执为 `blocked_missing_configured_openapi_token`（2026-11-13 11:43 UTC），尚无本次新源码包上传成功证据。账号在别的机器登录，不代表Monitor进程已有可用API凭据。凭据通过既有私有secret机制配置，禁止写到Git或公开文档。原有v3云包已经交付，但内容冻结于旧10-task；其可下载不解决新源码缺口。

## 2. 源码和运行版本关系

| 对象 | revision / 身份 |
|---|---|
| XR上游模型/评测辅助代码 | `0dd7aef8dc87296246aae812a1f59ccb708e5546` |
| RoboCasa | `456174f62b89b8fca99eaaf33949c29fec9cfc2a` |
| 实际Human50核心11文件 | `d0fcba205f5dcfdd502159ae8d6cdd8a80b17b59` |
| 1000→10000衔接controller | `288259d0337c79e6b68272a7c826faa2e4c29322` |
| 10000→19200衔接controller | `074f1755dcb2055ea397c5dd9797d14919fd285b` |
| 当前完整准备源快照 | `4bf26d9831a355651b820eb116d8b02ea44f300d` |

源码交付以本页上方固定的补充包为准，无需访问开发仓库。补充包按表中快照导出必要文件，不包含历史；`SOURCE_MANIFEST.json`对每个文件记录SHA/bytes。训练时再保存实际run contract，防止后来准备代码替代原运行核心。

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
