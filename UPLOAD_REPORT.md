# Archive upload status and training readiness

## Handoff bundle update — 12 November 2026

The original v3 ZIP is now stored in the existing private ModelScope archive at revision `6bbf6a71afc2c39f78fc3e2a35e584c67be875cf`. Its 11,563,326 bytes were fully downloaded again and SHA256-verified against `9e52a2342ad542af952376326bafdecd2fc43704301efbe461922e71fabe1466`. [Download and access instructions](DOWNLOADS.md) include the exact file path, pinned manifest, permission checks and extraction steps. The archive is unchanged; no dataset or parent-model weights were added to it.

See [HANDOFF_STATUS.md](HANDOFF_STATUS.md) for the current handoff audit and its limits. This upload resolves delivery of the code/evidence bundle; it does not establish full training readiness.

## R2 and fixed 20k parent supplement — 12 November 2026

The two missing inputs are now delivered in the [private supplement repository](https://modelscope.cn/datasets/dvdnh2026/ds-ecr-xr1-ascend-r2-parent-20261112). All 24,270 files / 15,683,968,143 bytes were checked at source, uploaded with immutable payload revisions, downloaded again into an empty directory, and checked against the frozen R2 manifest and all 22 parent pins. Source files were retained and rehashed after upload.

The complete mapping is separately pinned at `203223f41d3de33efdc1c0587de28075ddac963a` with SHA256 `fcce42bbf84728ca0b3545360e33be50353a24368e60ed8c8903d7bcb12d6b91`. [Executable download instructions](DOWNLOADS.md#6-r2-和固定-20k-父模型补传下载与独立验收), [mapping metadata](downloads/round2-parent-metadata-download.jsonl), and [verification details](downloads/round2-parent-verification.json) identify the exact paths, counts and verification methods. Hardware readiness and recipient permissions remain separate conditions.

## Historical data archive operation — 10 November 2026

Status reviewed on **10 November 2026**, from a completion report timestamped **18:41 UTC / 13:41 Toronto**.

The reviewed incremental archive operation completed 284 batches, adding 71,995 cloud paths and 44,787,578,352 bytes. The full mapping covers 370,075 logical source paths, including previously archived content and content aliases. Five final metadata files total 40,393,249 bytes. New payload verification used per-file sizes/hashes and 284 actual download samples; existing content relied on its frozen listing evidence, not a new full readback.

| Training input | Status in the reviewed material |
|---|---|
| Human demonstrations | [Public download entry](DOWNLOADS.md) contains pins for all ten tasks |
| Round 1 recovery | Complete mapping: 205,003 files, 49,789,441,549 bytes |
| Round 2 recovery | Not included in this historical operation; delivered by the 12 November supplement above |
| Fixed 20k parent model | Weight shards absent from this historical mapping; all 22 files delivered by the 12 November supplement above |
| Xiaomi original base | Download pins available in `downloads/xiaomi-base-download.jsonl` for diagnostics or alternative training tracks |
| Ascend port | Candidate preparation is available; hardware validation remains pending |

This completion report establishes archival progress, not full readiness for the Continue experiment. Environment setup and available-input downloads can begin; formal training requires target-side input checks and NPU validation described in [START_HERE.md](START_HERE.md).

Known repository identifiers and download instructions are published in [DOWNLOADS.md](DOWNLOADS.md). The complete archive file map is downloaded from its private ModelScope repository. Training source and operational evidence remain in the [private handoff bundle](PRIVATE_INPUTS.md).
