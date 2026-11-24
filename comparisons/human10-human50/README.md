# XR-1 Human-only SFT: NVIDIA reference for Ascend comparison

Updated 2026-11-24. **Human10: all five seeds complete; Human50: both seeds complete. Ascend training and full evaluation results have not yet been measured.** This package contains results, data sizes, checkpoint selection, measured training-log samples and a matching Ascend reporting template.

These are Human-only runs (`[human]`, no recovery mixture). The repository's older Round2 Continue 91% result uses a different Human/R1/R2 recipe and must not be used as the Human-only baseline.

## Training workload and data

| Item | Human10 | Human50 |
|---|---:|---:|
| Training tasks | 10 | 50 |
| Unique demonstration episodes | 5,040 | 25,307 |
| Unique demonstration frames | 3,176,983 | 14,957,899 |
| Completed training seeds | 1–5 | 1–2 |
| Optimizer updates per seed | 10,000 | 48,000 |
| Global batch | 192 | 192 |
| Sampled observations per seed (updates × batch) | 1,920,000 | 9,216,000 |
| Devices per training run | 4 NVIDIA GPUs; exact SKU unverified here | 4 NVIDIA B300 |
| Microbatch × devices × accumulation | 12 × 4 × 4 | 12 × 4 × 4 |
| Dev checkpoint candidates | 4,000 / 7,000 / 10,000 | 19,200 / 33,600 / 48,000 |
| Dev episodes per candidate | 200 (10 × 20) | 1,000 (50 × 20) |
| Final test per seed | 500 (10 × 50) | 2,500 (50 × 50) |

Sampled observations are not unique frames or epochs. Every seed starts from the original XR-1 base. Select the largest dev success count; ties choose the later checkpoint. Only the selected checkpoint receives the full test. Do not select a training seed based on test score.

Human10 uses 500 warmup updates, peak LR 2e-5 and cosine decay to 5e-6, with FSDP2 full SFT. Human50 uses 2,400 warmup updates; the detailed optimizer, precision, architecture, data loader and version contract is in [Target50 training configuration](../../target50/TRAINING_AND_ASCEND.md). Human10 provenance is code revision `86e8b0009e9455611bec34f48bc22ca54bc149ce`, recipe `training/rc365_xr1_20261006/scripts/train_job.sh`, job `rc365-xr1-e2e-h-s5-20261010`. The public source handoff's recovery recipe is not a substitute for this Human-only configuration.

## Measured NVIDIA training speed

| Measured log window | Samples | Mean s/update | Median s/update | Global observations/s | Budget extrapolation |
|---|---:|---:|---:|---:|---:|
| human10, seed 5, steps 9180–10000 | 83 | 4.3453 | 4.2410 | 44.19 | 12.07 h/seed |
| human50, seed 2, steps 19301–22589 | 3289 | 5.0978 | 4.0569 | 37.66 | 67.97 h/seed |

**These are partial measured windows, not measured full training wall time.** Human10 has one logged sample every ten updates. Human50 excludes the first 100 updates of a resumed segment and retains every later row, including slow rows; its mean therefore exceeds its median. The ~68 h Human50 number extrapolates that window across 48k updates; it is not a replacement for a complete job accounting record. Neither full training elapsed time nor total training GPU-hours is established by this package. Earlier ~50–60 h estimates based on faster windows are not measured completion times.

Raw timing samples and exact filters are supplied in `human10-timing-samples.csv`, `human50-timing-samples.csv` and `nvidia-training-timing.json`. Do not compare the two workload speeds as a GPU-generation benchmark. For Ascend, report the same global batch, precision, optimizer, data and timing window; separately report startup, checkpoint I/O, restarts, and end-to-end elapsed time.

## Complete test results

| Suite | Training seed | Selected update | Success / completed | Success rate | Allocated test GPU-hours |
|---|---:|---:|---:|---:|---:|
| human10 | 1 | 7000 | 424/500 | 84.80% | unavailable |
| human10 | 2 | 7000 | 406/500 | 81.20% | unavailable |
| human10 | 3 | 7000 | 397/500 | 79.40% | unavailable |
| human10 | 4 | 10000 | 407/500 | 81.40% | unavailable |
| human10 | 5 | 10000 | 411/500 | 82.20% | unavailable |
| human50 | 1 | 48000 | 2086/2500 | 83.44% | 71.885 |
| human50 | 2 | 48000 | 2074/2500 | 82.96% | 79.091 |

Human10 pooled across seeds: **2,045/2,500 = 81.80%**. Human50 pooled across seeds: **4,160/5,000 = 83.20%**. These are repeated seeded evaluations, not 2,500 or 5,000 independent scene draws. Human50 allocated test cost totals **150.976 GPU-hours** (one allocated GPU × group wall time), not GPU active-compute time; it excludes dev, training, failed earlier runs and environment preparation.

Human10 historical complete summary/episode counts have been rechecked. Human50 results remain **provisional (暂定)**: automatic artifact acceptance, coverage and identity checks passed, but independent full artifact/recovery review is pending. Completion of policy testing does not establish failure-recovery correctness.

## Protocol boundaries and original XR-1

Human10 is the frozen final ten-task list (includes TurnOnSinkFaucet, not OpenDrawer). Test uses target kitchens, original evaluator reset(seed), base seed 7; per-task seed offsets follow the original target50 task index. Dev base seed is 100007. Historical summaries explicitly caution that reset(seed) does not guarantee scene reproducibility across different runs/software. Use matching environment/assets and validate scenes when comparing Ascend.

Human50 test uses target50-v1/test identities, 50 tasks × 50 episodes, requested seeds 520000000 + task index × 1000 + episode. Per-episode checkpoint and protocol hashes are included. Each training seed is compared on the same scene IDs. NVIDIA evaluation used 16 simulator processes per GPU, 30 CPU cores and 240 GiB host memory per group; this is not a claim that the same concurrency fits an Ascend device.

The historical original XR-1 50-task result is **1,443/2,500 = 57.72%**, on **pretrain kitchens**, while the Human50 results above use target kitchens. `original50-pretrain-historical.csv` preserves its per-task counts. The difference is descriptive only; it is not a controlled SFT gain or a same-protocol NV/Ascend comparison.

## Files and Ascend completion contract

- [Training workload and data](training-summary.json), [Human10 data counts and pins](human10-data.json).
- [Human10 dev selections](human10-selection.json), [Human50 dev selections](human50-selection.json).
- [All final summaries](test-summary.json), [all 150 task/seed rows](per-task.csv), [all 7,500 test episode rows](test-episodes.csv).
- [NVIDIA timing definitions](nvidia-training-timing.json) and both timing CSVs.
- [Ascend result template](ascend-results-template.json), [source evidence hashes](source-manifest.json).

For each Ascend suite/seed, record exact device model/count, software versions, input/code/checkpoint hashes, microbatch/accumulation/global batch, precision, optimizer schedule, full training elapsed time and the same steady-window statistics. Evaluate all three dev candidates; select by the frozen rule; evaluate only that checkpoint on the complete 500- or 2,500-episode test. Publish per-task k/n and per-episode identities, infrastructure failures, selected checkpoint and protocol hashes. Speedup = matched NVIDIA seconds/update ÷ Ascend seconds/update; compare end-to-end time separately. Leave missing values null. No Ascend speedup or accuracy delta is currently available.

This upload contains compact reports and result records, not model weights, private credentials, videos, or recovery trajectories. It does not resolve checkpoint distribution or certify the Ascend execution environment.
