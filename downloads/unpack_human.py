#!/usr/bin/env python3
"""Restore the ten pinned Human tar archives and verify every payload file."""
import argparse
import json
from pathlib import Path
import tarfile
import tempfile
from download_files import sha256, relative_path


def verify_payload(base, expected):
    actual = {p.relative_to(base).as_posix() for p in (base / "payload").rglob("*") if p.is_file()}
    if actual != set(expected):
        raise ValueError(f"Payload file set mismatch: {base}")
    for rel, digest in expected.items():
        p = base / rel
        if p.is_symlink() or sha256(p) != digest:
            raise ValueError(f"Payload SHA256 mismatch: {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    pack = Path(__file__).resolve().parents[1]
    catalog = json.loads((pack / "downloads/human10.json").read_text())
    root = Path(args.root).expanduser().resolve()
    for task, item in catalog.items():
        base = root / "datasets/rc365_xr1_20261006" / item["name"] / item["version"]
        archive = base / "transport/lerobot.tar"
        sums = base / "SHA256SUMS"
        if sha256(archive) != item["archive_sha256"] or sha256(sums) != item["manifest_sha256"]:
            raise ValueError(f"Pinned archive/manifest mismatch: {task}")
        expected = {}
        for line in sums.read_text().splitlines():
            digest, rel = line.split("  ", 1)
            rel = str(relative_path(rel))
            if not rel.startswith("payload/") or rel in expected:
                raise ValueError("Unexpected payload manifest path")
            expected[rel] = digest
        if not (base / "payload").exists():
            stage = Path(tempfile.mkdtemp(prefix=".unpack-", dir=base))
            (stage / "payload").mkdir()
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    if member.isdir() and member.name.rstrip("/") in (".", ""):
                        continue
                    relative_path(member.name.rstrip("/").removeprefix("./"))
                    if not (member.isfile() or member.isdir()):
                        raise ValueError("Unexpected tar link or special file")
                tf.extractall(stage / "payload", filter="data")
            verify_payload(stage, expected)
            (stage / "payload").rename(base / "payload")
            stage.rmdir()
        verify_payload(base, expected)
        info = json.loads((base / "payload/lerobot/meta/info.json").read_text())
        if info["total_episodes"] != item["episodes"] or info["total_frames"] != item["frames"]:
            raise ValueError(f"Unexpected dataset counts: {task}")
        print(f"verified {task}: {item['episodes']} episodes, {item['frames']} frames", flush=True)


if __name__ == "__main__":
    main()
