#!/usr/bin/env python3
"""Download an explicit JSONL file map; verify SHA256 and restore logical paths.

No cloud discovery, uploads or directory flattening. Exact revisions are preferred;
legacy master reads require explicit sha256_locked_branch mode and frozen file size.
Requires modelscope-hub==0.6.1 only when a file actually needs downloading.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Missing relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or value == ".":
        raise ValueError(f"Unsafe relative path: {value}")
    return p


def load_manifest(path):
    rows, seen = [], set()
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        lines = stream.readlines()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("repo_id", "repo_type", "revision", "remote_path", "local_path", "sha256"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Line {n}: missing {field}")
        if row["repo_type"] not in ("dataset", "model"):
            raise ValueError(f"Line {n}: invalid repo_type")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", row["revision"]):
            if not (row['revision'] == 'master' and row.get('revision_mode') == 'sha256_locked_branch'
                    and type(row.get('bytes')) is int):
                raise ValueError(f"Line {n}: needs an exact revision or explicit SHA256-locked master with byte size")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Line {n}: invalid SHA256")
        for key in ("remote_path", "local_path"):
            relative_path(row[key])
        if row.get("bytes") is not None and (type(row["bytes"]) is not int or row["bytes"] < 0):
            raise ValueError(f"Line {n}: invalid byte size")
        if row["local_path"] in seen:
            raise ValueError(f"Line {n}: duplicate logical destination")
        seen.add(row["local_path"])
        rows.append(row)
    if not rows:
        raise ValueError("An empty manifest is not a complete download")
    return rows


def verify(path, row):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular file: {path}")
    if row.get("bytes") is not None and path.stat().st_size != row["bytes"]:
        raise ValueError(f"Byte size mismatch: {path}")
    if sha256(path) != row["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--root", required=True)
    ap.add_argument("--check-only", action="store_true", help="Validate manifest only; no network or writes")
    ap.add_argument('--workers', type=int, default=4, help='Concurrent files, 1-16; SDK client isolated per thread')
    args = ap.parse_args()
    rows = load_manifest(args.manifest)
    if not 1 <= args.workers <= 16:
        ap.error('--workers must be 1-16')
    if args.check_only:
        print(json.dumps({"manifest_valid": True, "files": len(rows), "files_downloaded": 0}))
        return
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    clients = threading.local()
    def download_one(row):
        dst = root / row["local_path"]
        if not dst.resolve().is_relative_to(root):
            raise ValueError(f"Destination escapes root: {dst}")
        if dst.exists() or dst.is_symlink():
            verify(dst, row)  # Refuse to overwrite an existing different file.
        else:
            if not hasattr(clients, 'api'):
                from modelscope_hub import HubApi
                clients.api = HubApi(token=os.environ.get("MODELSCOPE_API_TOKEN"))
            cached = Path(clients.api.download_file(
                row["repo_id"], repo_type=row["repo_type"], file_path=row["remote_path"],
                revision=row["revision"], expected_sha256=row["sha256"], file_size=row.get("bytes"),
                cache_dir=root / ".modelscope-cache")).resolve()
            verify(cached, row)
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".download-", dir=dst.parent)
            os.close(fd)
            try:
                shutil.copyfile(cached, tmp)
                verify(Path(tmp), row)
                # Link the verified temporary file without replacing another writer's destination.
                try:
                    os.link(tmp, dst)
                except FileExistsError:
                    verify(dst, row)
            finally:
                os.unlink(tmp)
        return row['local_path']
    pool = ThreadPoolExecutor(max_workers=args.workers)
    remaining = iter(rows); pending = set(); i = 0
    try:
        for _ in range(args.workers):
            row = next(remaining, None)
            if row is not None: pending.add(pool.submit(download_one, row))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            paths = [future.result() for future in done]  # Check every completed result before enqueueing more.
            for path in paths:
                i += 1
                if i == 1 or i % 1000 == 0 or i == len(rows):
                    print(f'{i}/{len(rows)} verified {path}', flush=True)
                row = next(remaining, None)
                if row is not None: pending.add(pool.submit(download_one, row))
    finally:
        for future in pending: future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    receipt = {"complete_for_manifest": True, "files": len(rows),
               "manifest_sha256": sha256(args.manifest), "root": str(root),
               "scope": "Only the listed files; not proof that the input manifest covers every required dataset",
               "sha256_locked_branch_files": sum(r.get('revision_mode') == 'sha256_locked_branch' for r in rows)}
    path = root / f"download-verified-{receipt['manifest_sha256'][:16]}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
