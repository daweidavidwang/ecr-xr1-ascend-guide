# Private inputs required for execution

This public repository contains documentation. Authorized collaborators need the **XR1 Ascend reproduction v3 handoff bundle** from the project owner to execute its commands. Keep that bundle outside this public Git repository.

The private supplement supplies:

- `scripts/`: download, extraction, integrity checks, configuration generation, environment probing, NPU candidate generation, stage launching, evaluation comparison and timing tools.
- `manifests/`: exact private ModelScope repository/file mappings and content hashes for Human and first-round recovery, plus public-base model pins.
- `reference/source-round2/`, `reference/a3-first/`, `reference/a3-continue/`: frozen project training source and configurations.
- `reference/model-pins-base-parent.json` and related files: exact model identity checks, including all 22 files of the 20k parent.
- `reference/gpu-test500-episodes.jsonl`, selection and timing evidence: the frozen GPU comparison records.
- `SHA256SUMS`: the private package checksum list. It is distinct from this repository's `PUBLIC_SHA256SUMS`.

The public Markdown links to private evidence intentionally lead here. Their filenames in command examples refer to the supplementary bundle, not to files shipped in this public repository. Historical paths beginning with `$LEGACY_WORK` are logical placeholders; actual server paths are supplied privately.

## Outstanding requirements as of 10 November 2026

The latest reviewed archive completion report covers first-round recovery, but not the second-round recovery directory or the 20k parent model's weight shards. Their upload mappings must be supplied and verified before formal Continue training. See [required inputs](MISSING_INPUTS.md).

Ascend hardware validation is also pending. The provided candidate is a starting point for a reviewed port, not a tested NPU release. Start with [the execution guide](START_HERE.md), and use the allocated cluster environment and private repository access provided to you.

No credentials, dataset contents, model weights, private file listings or internal receipts are published here.
