#!/usr/bin/env python3
"""Read-only verification of the WAM public input manifests (standard library).

No network access, extraction, model loading, input modification or credentials.
Prints one JSON receipt to stdout; a nonzero exit means incomplete/failed checks.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def verify(manifest, root, component=None):
    raw = manifest.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1 or not isinstance(data.get("files"), list):
        raise ValueError("Expected a version 1 files manifest")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Input root is not a directory")
    seen = set()
    selected = []
    for item in data["files"]:
        rel = PurePosixPath(item["destination"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError("Unsafe destination")
        if str(rel) in seen:
            raise ValueError("Duplicate destination")
        seen.add(str(rel))
        kind = item["verification"]
        if kind not in ("sha256", "git_blob_sha1"):
            raise ValueError("Unsupported digest type")
        digest = item["digest"]
        length = 64 if kind == "sha256" else 40
        if len(digest) != length or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Malformed digest")
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise ValueError("Expected positive file size")
        if component is None or item["component"] == component:
            selected.append(item)
    if not selected:
        raise ValueError("No files selected")
    rows = []
    for item in selected:
        row = {"destination": item["destination"], "passed": False}
        try:
            path = root / item["destination"]
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing file or final symlink")
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("File resolves outside input root")
            before = path.stat()
            if before.st_size != item["bytes"]:
                raise ValueError("File size mismatch")
            sha = hashlib.sha256()
            blob = hashlib.sha1(("blob %d\0" % before.st_size).encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    sha.update(chunk)
                    blob.update(chunk)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns
            ):
                raise ValueError("File changed during verification")
            actual = sha.hexdigest() if item["verification"] == "sha256" else blob.hexdigest()
            if actual != item["digest"]:
                raise ValueError("Content digest mismatch")
            row.update(passed=True, bytes=after.st_size, sha256=sha.hexdigest(),
                       upstream_digest_type=item["verification"], upstream_digest=actual)
        except (OSError, ValueError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "root": str(root), "component": component,
            "files_planned": len(rows), "files_passed": sum(r["passed"] for r in rows),
            "passed": all(r["passed"] for r in rows), "files": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--component")
    args = parser.parse_args()
    try:
        receipt = verify(args.manifest, args.root, args.component)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
