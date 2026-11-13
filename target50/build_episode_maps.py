#!/usr/bin/env python3
"""Rebuild the three frozen episode indexes from public LeRobot v3 metadata.

This is a standalone metadata utility, not the private XR training implementation.
Requires pyarrow==21.0.0. No dataset conversion, normalization or video changes.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    here = Path(__file__).resolve().parent
    root = args.root.expanduser().resolve() / 'datasets/target50_human_v3_20261112'
    catalog = json.loads((here / 'tasks50.json').read_text())['tasks']
    expected = json.loads((here / 'human50-mapping-pins.json').read_text())
    counts = json.loads((here / 'task-counts.json').read_text())
    files = [json.loads(s) for s in (here / 'human50-download.jsonl').read_text().splitlines()]
    for row in files:
        if row['remote_path'].startswith('meta/'):
            path = args.root.expanduser().resolve() / row['local_path']
            if (path.stat().st_size != row['bytes'] or
                    hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']):
                raise ValueError('Metadata bytes differ: ' + str(path))
    out = root / 'maps'
    out.mkdir(exist_ok=True)
    for pin in expected:
        group = pin['file'].removesuffix('-episodes.jsonl')
        tasks = {t['registry_key'].removesuffix('/lerobot.tar'): t for t in catalog if t['group'] == group}
        rows = pq.read_table(root / group / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
        totals = {t['task']: [0, 0] for t in tasks.values()}
        records = []
        for row in sorted(rows, key=lambda r: r['episode_index']):
            task = tasks[row['source_prefix']]
            totals[task['task']][0] += 1
            totals[task['task']][1] += row['length']
            records.append(dict(group=group, task=task['task'], target50_index=task['target50_index'], **row))
        for task, (episodes, frames) in totals.items():
            if (episodes, frames) != (counts[task]['episodes'], counts[task]['frames']):
                raise ValueError('Task membership or frame count differs: ' + task)
        raw = ''.join(json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n' for r in records).encode()
        if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('Regenerated index differs from frozen B300 index: ' + group)
        path = out / pin['file']
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('Existing index differs; retained: ' + str(path))
        else:
            with path.open('xb') as f:
                f.write(raw)
        print(json.dumps({'group': group, 'episodes': len(rows), 'sha256': pin['sha256'], 'passed': True}))


if __name__ == '__main__':
    main()
