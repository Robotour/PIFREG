#!/usr/bin/env python3
"""Verify before/after metrics used the same test set (fingerprint check)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.experiments.split_eval_manifest import (
    load_test_eval_manifest,
    verify_same_test_set,
)


def verify_run(run_dir: Path) -> int:
    run_dir = Path(run_dir)
    manifest = load_test_eval_manifest(run_dir)
    print(f'Run: {run_dir}')
    print(f'  seed                 : {manifest["seed"]}')
    print(f'  test sessions        : {manifest["num_test_sessions"]}')
    print(f'  test_set_fingerprint : {manifest["test_set_fingerprint"]}')

    unreg = run_dir / 'test_metrics_unregistered.json'
    after = run_dir / 'test_metrics.json'
    ok = True

    if unreg.is_file():
        unreg_payload = json.loads(unreg.read_text(encoding='utf-8'))
        check = verify_same_test_set(manifest, unreg_payload)
        status = 'OK' if check['ok'] else 'MISMATCH'
        print(f'  vs unregistered      : {status}')
        ok = ok and check['ok']
    else:
        print('  vs unregistered      : MISSING test_metrics_unregistered.json')
        ok = False

    if after.is_file():
        after_payload = json.loads(after.read_text(encoding='utf-8'))
        check = verify_same_test_set(manifest, after_payload)
        status = 'OK' if check['ok'] else 'MISMATCH'
        print(f'  vs after (test_metrics): {status}')
        if after_payload.get('same_test_set_as_unregistered') is True:
            print('  after file flag        : same_test_set_as_unregistered=true')
        ok = ok and check['ok']
    else:
        print('  vs after             : MISSING test_metrics.json')

    split_path = run_dir / 'split_manifest.json'
    if split_path.is_file():
        split_payload = json.loads(split_path.read_text(encoding='utf-8'))
        check = verify_same_test_set(manifest, split_payload)
        print(f'  vs split_manifest    : {"OK" if check["ok"] else "MISMATCH"}')
        ok = ok and check['ok']

    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description='Verify test set consistency in a run directory')
    p.add_argument('--run-dir', required=True, help='VoxelMorph or classical baseline run dir')
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    raise SystemExit(verify_run(run_dir))


if __name__ == '__main__':
    main()
