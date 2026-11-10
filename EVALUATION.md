# Ascend 训练后的固定 GPU 评测流程

> **Public guide with download entry points / 公开指南与下载入口。** Known ModelScope addresses, fixed download pins and generic download tools are available in [DOWNLOADS.md](DOWNLOADS.md). Dataset repositories remain private. Frozen training source, NPU execution tools and GPU comparison evidence remain in the separately transferred v3 handoff ZIP; see [PRIVATE_INPUTS.md](PRIVATE_INPUTS.md).

为了比较训练后端，使用原 GPU RoboCasa evaluator。固定模型服务、仿真代码、场景 seeds 和视频设置；不要同时迁移仿真环境并把差异全部归因于训练。

评测方需要本包 `reference/source-round2/`、公开 XR‑1 commit `0dd7aef8dc87296246aae812a1f59ccb708e5546`，以及已验证的 RoboCasa `456174f62b89b8fca99eaaf33949c29fec9cfc2a` / robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` 环境与资产。原 `env.sh` 仅适配旧主机：在独立副本中修改本地路径/环境入口，不能原样在 Ascend 调用。原 `build_envs.sh` 是历史参考，其中 CUDA wheel 不能装到 NPU 训练环境。

将 Ascend 的 `hf/step_4000`、`step_7000`、`step_10000` 传给 GPU 评测方；传输前后核对每个导出全部文件 SHA256。使用独立 ARM 名，例如 `ecr_ascend_r2_continue_a1`，放入评测端 `RUNS/sft/${ARM}_s1/hf/`。传输和作业提交由有该节点权限的执行方按其平台规则完成。

在 GPU allocation 内运行（变量来自评测方已经配置好的 `env.sh`）：

```bash
set -euo pipefail
export RC365_STRICT_ONE_PASS=1
export PROMPT_TAG='[human]'
export SERVER_IMPL=orig
export SAVE_VIDEOS=1
unset RC365_BANK RC365_TAKEOVER RC365_QUEUE_KEEP
TASKS='TurnOnSinkFaucet,TurnOffStove,CoffeeSetupMug,NavigateKitchen,KettleBoiling,LoadDishwasher,SteamInMicrowave,WashFruitColander,ArrangeTea,MakeIceLemonade'
bash "$EXP_DIR/scripts/sft_eval.sh" ecr_ascend_r2_continue_a1 1 "$TASKS"
```

该入口自动执行三个 dev200、按最大成功数选点（并列选较晚步）及 test500。原 GPU 的 dev 173 / 175 / 175 选 10k；新模型必须重新按同规则选点，不能直接指定 10k 并称为 dev-selected。开发集 base seed=100007，测试 base seed=7；单局 seed 使用冻结 target50 index，而非 10 任务内的新索引。生成的 500 task/seed 可与本包 `reference/gpu-test500-episodes.jsonl` 核对。

保持 history=4、interval=2、crop=0.95、replan/action chunk=16、5 个 Euler flow steps；视频为原三视角横拼、stride=4、10 fps。保留逐局结果、技术错误、视频、模型哈希、版本和 selection。技术错误不静默改 seed 或减少分母；诊断与补跑保留独立 attempt，不覆盖首次记录。

评测完成后在本包根目录运行：

```bash
python scripts/compare_test500.py \
  "$RUNS/eval/ecr_ascend_r2_continue_a1_s1_test/episodes.jsonl" \
  --output "$RUNS/eval/ecr_ascend_r2_continue_a1_s1_test/gpu-vs-ascend.json"
```

报告全部 500 局、NAV 50 局、非 NAV 450 局和逐任务成功率，同时报告训练 seed 和 dev 选点。GPU 对照为 Continue s1 455/500、NAV 46/50、非 NAV 409/450；另一个训练 seed 的历史 Continue s2 为 442/500，说明 91.0% 不是任意后端/seed 必须精确达到的保证。若做跨平台平均效果结论，按相同训练 seeds 扩展两端重复，不能各挑最高一次。
