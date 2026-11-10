#!/usr/bin/env python3
"""Recover R1 download locations from the pinned private archive metadata."""
import argparse
import gzip
import json
from pathlib import Path

from download_files import sha256, relative_path

SOURCE_SHA = '8dac3099cfad185ad859e308eb2af1af29cbf4834a23f1cec2ff3e85901f24a3'
PREFIX = 'datasets/ecr-recovery-unified-9task/v1-20261020/'
EXPECTED_FILES = 205003
EXPECTED_BYTES = 49789441549


def build(source):
    if sha256(source) != SOURCE_SHA:
        raise ValueError('Archive map does not match the frozen SHA256')
    rows = []
    seen = set()
    with gzip.open(source, 'rt') as stream:
        for line in stream:
            entry = json.loads(line)
            local_path = entry['source']
            if not local_path.startswith(PREFIX):
                continue
            relative_path(local_path)
            if local_path in seen:
                raise ValueError('Duplicate R1 logical path')
            seen.add(local_path)
            cloud = entry['cloud']
            relative_path(cloud['path'])
            rows.append({
                'repo_id': cloud['repo_id'], 'repo_type': 'dataset',
                'revision': 'master', 'revision_mode': 'sha256_locked_branch',
                'remote_path': cloud['path'], 'local_path': local_path,
                'sha256': entry['sha256'], 'bytes': entry['bytes'],
            })
    if len(rows) != EXPECTED_FILES or sum(row['bytes'] for row in rows) != EXPECTED_BYTES:
        raise ValueError('R1 file count or total bytes differs from the frozen handoff')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.source)
    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text() != content:
            raise ValueError('Refusing to overwrite a different output map')
    else:
        with args.output.open('x') as stream:
            stream.write(content)
    print(json.dumps({'files': len(rows), 'bytes': EXPECTED_BYTES, 'output': str(args.output),
                      'scope': 'Frozen R1 only; no R2 or parent model mapping'}))


if __name__ == '__main__':
    main()
