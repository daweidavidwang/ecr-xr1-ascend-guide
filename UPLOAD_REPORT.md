# Archive upload status and training readiness

## Handoff bundle update — 12 November 2026

The original v3 ZIP is now stored in the existing private ModelScope archive at revision `6bbf6a71afc2c39f78fc3e2a35e584c67be875cf`. Its 11,563,326 bytes were fully downloaded again and SHA256-verified against `9e52a2342ad542af952376326bafdecd2fc43704301efbe461922e71fabe1466`. [Download and access instructions](DOWNLOADS.md) include the exact file path, pinned manifest, permission checks and extraction steps. The archive is unchanged; no dataset or parent-model weights were added to it.

See [HANDOFF_STATUS.md](HANDOFF_STATUS.md) for the current handoff audit and its limits. This upload resolves delivery of the code/evidence bundle; it does not establish full training readiness.

## R2 and fixed 20k parent supplement — upload verified, readback pending

All 24,270 files / 15,683,968,143 bytes have been uploaded and checked against source pins at immutable cloud revisions. The complete mapping was independently downloaded and checked against the frozen file set. **Full payload cloud readback has not yet passed.** [Download instructions](DOWNLOADS.md#6-r2-和固定-20k-父模型补传下载与独立验收) and [current verification status](downloads/round2-parent-verification.json) are available now; final acceptance will be recorded when complete.

Mapping revision: `203223f41d3de33efdc1c0587de28075ddac963a`. Private repository: [dvdnh2026/ds-ecr-xr1-ascend-r2-parent-20261112](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-xr1-ascend-r2-parent-20261112). Recipient permissions and hardware readiness are separate conditions.

## Historical data archive operation — 10 November 2026

Status reviewed on **10 November 2026**, from a completion report timestamped **18:41 UTC / 13:41 Toronto**.

The reviewed incremental archive operation completed 284 batches, adding 71,995 cloud paths and 44,787,578,352 bytes. The full mapping covers 370,075 logical source paths, including previously archived content and content aliases. Five final metadata files total 40,393,249 bytes. New payload verification used per-file sizes/hashes and 284 actual download samples; existing content relied on its frozen listing evidence, not a new full readback.

| Training input | Status in the reviewed material |
|---|---|
| Human demonstrations | [Public download entry](DOWNLOADS.md) contains pins for all ten tasks |
| Round 1 recovery | Complete mapping: 205,003 files, 49,789,441,549 bytes |
| Round 2 recovery | Not included in this archive operation; separate upload mapping required |
| Fixed 20k parent model | Weight shards not present in the reviewed mapping; separate complete model mapping required |
| Xiaomi original base | Download pins available in `downloads/xiaomi-base-download.jsonl` for diagnostics or alternative training tracks |
| Ascend port | Candidate preparation is available; hardware validation remains pending |

This completion report establishes archival progress, not full readiness for the Continue experiment. Environment setup and available-input downloads can begin; formal training requires target-side input checks and NPU validation described in [START_HERE.md](START_HERE.md).

Known repository identifiers and download instructions are published in [DOWNLOADS.md](DOWNLOADS.md). The complete archive file map is downloaded from its private ModelScope repository. Training source and operational evidence remain in the [private handoff bundle](PRIVATE_INPUTS.md).
