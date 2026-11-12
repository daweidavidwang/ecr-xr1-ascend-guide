#!/usr/bin/env python3
"""Verify complete R2 + fixed 20k parent coverage and downloaded bytes.

Requires only Python 3.10+ and the verified original v3 bundle. Human/R1 are
not required for this two-input check; they remain required for training.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import re

from download_files import load_manifest, relative_path, sha256, verify

R2 = 'datasets/ecr-recovery-unified-9task/v3-round2-20261101/'
PARENT = 'models/xr1-a3-s1-cont-20k/'
CONTROLS = {
    'MANIFEST.sha256': 'aeb4d7271114522d7dc0e2c85ac3f1b26a93ff8b3d40baa014f380bbae112676',
    'meta.json': '0c036a91ed278408325576d9f2ab8c96d0d89f248c9e3672b2246411b1579cb0',
    'README.md': '56997b2a4570a67c24e9d9e616fb4d67bb43629b58dd0c81d5ddd4eac1b1fa62',
}
PINS_SHA = '171aa7dd4eeb8b65a3c9a416f925d5a1e2ac9a2405447c73c84f1246720446ff'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve()
    bundle = Path(a.bundle).expanduser().resolve()
    manifest = root / R2 / 'MANIFEST.sha256'
    pins_file = bundle / 'reference/model-pins-base-parent.json'
    require(sha256(pins_file) == PINS_SHA, 'Original v3 parent pins differ')
    expected = {}
    for name, digest in CONTROLS.items():
        p = root / R2 / name
        require(p.resolve().is_relative_to(root), 'R2 control escapes restore root')
        verify(p, {'sha256': digest})
        expected[R2 + name] = (digest, p.stat().st_size)
    payload = {}
    for line in manifest.read_text().splitlines():
        digest, size, rel = line.split(None, 2)
        relative_path(rel)
        require(re.fullmatch('[0-9a-f]{64}', digest) and size.isdigit(), 'Invalid R2 manifest row')
        require(R2 + rel not in payload, 'Duplicate R2 manifest row')
        payload[R2 + rel] = (digest, int(size))
    require(len(payload) == 24245 and sum(v[1] for v in payload.values()) == 5557378683,
            'Frozen R2 count or byte total differs')
    for key, value in payload.items():
        require(key not in expected or expected[key] == value, 'Conflicting R2 control pin')
        expected[key] = value
    pins = json.loads(pins_file.read_text())['continue']
    parent = {}
    for row in pins['files']:
        rel = PurePosixPath(row['path'])
        if rel.is_absolute():
            rel = rel.relative_to(pins['root'])
        relative_path(str(rel))
        key = PARENT + str(rel)
        require(key not in parent, 'Duplicate parent pin')
        parent[key] = (row['sha256'], row['bytes'])
    require(len(parent) == 22 and sum(v[1] for v in parent.values()) == 10122587755,
            'Frozen parent count or byte total differs')
    expected.update(parent)
    rows = load_manifest(a.map)
    require({r['local_path'] for r in rows} == set(expected),
            'Download map must cover exactly all frozen R2 files, controls and 22 parent files')
    for i, row in enumerate(rows, 1):
        require(re.fullmatch('[0-9a-f]{40}', row['revision']), 'Supplement requires immutable revisions')
        require((row['sha256'], row.get('bytes')) == expected[row['local_path']],
                'Map differs from frozen identity: ' + row['local_path'])
        p = root / row['local_path']
        require(p.resolve().is_relative_to(root), 'File escapes restore root')
        verify(p, row)
        if i % 5000 == 0 or i == len(rows):
            print(f'{i}/{len(rows)} frozen files verified', flush=True)
    for directory, prefix in [(root / R2, R2), (root / PARENT, PARENT)]:
        for p in directory.rglob('*'):
            require(not p.is_symlink(), 'Symlink in restored input: ' + str(p))
            if p.is_file():
                require(prefix + p.relative_to(directory).as_posix() in expected,
                        'Unexpected file in restored input: ' + str(p))
    report = {
        'passed': True, 'map_sha256': sha256(a.map), 'files': len(rows),
        'bytes': sum(r['bytes'] for r in rows), 'r2_payload_files': len(payload),
        'r2_payload_bytes': 5557378683, 'parent_files': len(parent),
        'parent_bytes': 10122587755, 'root': str(root),
        'scope': 'Complete R2 and fixed 20k parent identity and local content; not Human/R1 or NPU acceptance',
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
