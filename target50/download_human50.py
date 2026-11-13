#!/usr/bin/env python3
"""Download public Target50 files at frozen HF revisions; verify bytes and SHA256.

Python standard library only. Does not download training source or model weights.
Existing mismatches and failed partials are preserved. Never follows a moving branch.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def checked_path(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe destination')
    p = root / rel
    if p.resolve() != p or not p.is_relative_to(root):
        raise ValueError('Symlink or escaped destination')
    return p


def transfer(row, root, check_only=False):
    path = checked_path(root, row['local_path'])
    if path.exists():
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Existing content differs; retained: ' + row['local_path'])
        return {'path': row['local_path'], 'status': 'verified_existing'}
    if check_only:
        raise FileNotFoundError(row['local_path'])
    if not re.fullmatch(r'[0-9a-f]{40}', row['revision']):
        raise ValueError('Must use a fixed revision')
    if not row['repo_id'].startswith('ember-lab-berkeley/robocasa365-target-'):
        raise ValueError('Unexpected repository')
    url = ('https://huggingface.co/datasets/' + row['repo_id'] + '/resolve/'
           + row['revision'] + '/' + urllib.parse.quote(row['remote_path'], safe='/'))
    path.parent.mkdir(parents=True, exist_ok=True)
    checked_path(root, row['local_path'])
    fd, temp = tempfile.mkstemp(prefix=path.name + '.partial-', dir=path.parent)
    temp = Path(temp)
    count = 0
    h = hashlib.sha256()
    with os.fdopen(fd, 'wb') as out:
        request = urllib.request.Request(url, headers={'User-Agent': 'ecr-target50-pinned-downloader/1'})
        with urllib.request.urlopen(request, timeout=120) as response:
            while True:
                block = response.read(8 << 20)
                if not block:
                    break
                count += len(block)
                if count > row['bytes']:
                    raise ValueError('Response exceeds frozen size')
                h.update(block)
                out.write(block)
        out.flush()
        os.fsync(out.fileno())
    if count != row['bytes'] or h.hexdigest() != row['sha256']:
        raise ValueError('Downloaded content differs; partial retained: ' + str(temp))
    os.link(temp, path)  # exclusive publication; never overwrites an existing file
    temp.unlink()
    return {'path': row['local_path'], 'status': 'downloaded_verified'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=Path(__file__).with_name('human50-download.jsonl'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--metadata-only', action='store_true')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.workers <= 8:
        ap.error('--workers must be between 1 and 8')
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    if len(rows) != 307 or sum(r['bytes'] for r in rows) != 43253612263:
        raise ValueError('Expected frozen 307-file payload')
    if len({r['local_path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate destination')
    if args.metadata_only:
        rows = [r for r in rows if r['remote_path'].startswith('meta/')]
    receipt = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'manifest_sha256': sha(args.manifest), 'metadata_only': args.metadata_only,
               'planned_files': len(rows), 'verified': [], 'errors': []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(transfer, r, root, args.check_only): r for r in rows}
        for future in concurrent.futures.as_completed(pending):
            row = pending[future]
            try:
                receipt['verified'].append(future.result())
            except Exception as exc:
                receipt['errors'].append({'path': row['local_path'], 'error': str(exc)})
    receipt['passed'] = not receipt['errors'] and len(receipt['verified']) == len(rows)
    receipt['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fd, name = tempfile.mkstemp(prefix='human50-download-receipt-', suffix='.json', dir=root)
    with os.fdopen(fd, 'w') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps({'passed': receipt['passed'], 'files': len(receipt['verified']), 'receipt': name}))
    if not receipt['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
