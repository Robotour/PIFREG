"""Fixed train/test split and verifiable test-set identity for before/after metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


def _normalize_session_paths(folders: Sequence) -> List[str]:
    return sorted(str(Path(p).resolve()) for p in folders)


def fingerprint_test_sessions(test_folders: Sequence) -> str:
    """Stable SHA256 over sorted absolute test session paths."""
    payload = '\n'.join(_normalize_session_paths(test_folders)).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def build_test_eval_manifest(
    seed: int,
    train_folders: Sequence,
    test_folders: Sequence,
    train_ratio: float,
    *,
    data_dir: Optional[str] = None,
    image_size=None,
    max_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    test_paths = _normalize_session_paths(test_folders)
    train_paths = _normalize_session_paths(train_folders)
    return {
        'seed': seed,
        'train_ratio': train_ratio,
        'data_dir': data_dir,
        'image_size': list(image_size) if image_size else None,
        'max_sessions': max_sessions,
        'num_train_sessions': len(train_paths),
        'num_test_sessions': len(test_paths),
        'train_sessions': train_paths,
        'test_sessions': test_paths,
        'test_set_fingerprint': fingerprint_test_sessions(test_folders),
        'metric_definition': (
            'Before/after metrics always use test_sessions above. '
            'Per session: mean over all C(N,2) band pairs, then mean over sessions.'
        ),
    }


def save_test_eval_manifest(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / 'test_eval_manifest.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def load_test_eval_manifest(source: Union[Path, str]) -> Dict[str, Any]:
    path = Path(source)
    if path.is_dir():
        path = path / 'test_eval_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing test eval manifest: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def test_folders_from_manifest(manifest: Dict[str, Any]) -> List[Path]:
    return [Path(p) for p in manifest['test_sessions']]


def verify_same_test_set(
    reference: Union[Dict[str, Any], Path, Sequence],
    candidate: Union[Dict[str, Any], Path, Sequence],
) -> Dict[str, Any]:
    """
    Compare two manifests, run dirs, or raw folder lists.
    Returns {ok, reference_fingerprint, candidate_fingerprint, ...}.
    """
    def _fp(obj) -> str:
        if isinstance(obj, (Path, str)) and Path(obj).is_dir():
            return load_test_eval_manifest(obj)['test_set_fingerprint']
        if isinstance(obj, dict):
            if 'test_set_fingerprint' in obj:
                return obj['test_set_fingerprint']
            if 'test_sessions' in obj:
                return fingerprint_test_sessions(obj['test_sessions'])
        return fingerprint_test_sessions(obj)

    ref_fp = _fp(reference)
    cand_fp = _fp(candidate)
    return {
        'ok': ref_fp == cand_fp,
        'reference_fingerprint': ref_fp,
        'candidate_fingerprint': cand_fp,
    }


def print_test_eval_banner(manifest: Dict[str, Any], run_dir: Path) -> None:
    print('\n' + '=' * 60, flush=True)
    print('Test set locked for before/after comparison', flush=True)
    print('=' * 60, flush=True)
    print(f'  run_dir              : {run_dir}', flush=True)
    print(f'  split_manifest       : {run_dir / "split_manifest.json"}', flush=True)
    print(f'  test_eval_manifest   : {run_dir / "test_eval_manifest.json"}', flush=True)
    print(f'  seed                 : {manifest["seed"]}', flush=True)
    print(
        f'  sessions             : {manifest["num_train_sessions"]} train / '
        f'{manifest["num_test_sessions"]} test',
        flush=True,
    )
    print(f'  test_set_fingerprint : {manifest["test_set_fingerprint"]}', flush=True)
    print(
        '  Before metrics file  : test_metrics_unregistered.json (written before training)',
        flush=True,
    )
    print(
        '  After metrics file   : test_metrics.json (written after eval)',
        flush=True,
    )
    print('=' * 60 + '\n', flush=True)


def assert_test_set_matches_run(
    run_dir: Path,
    test_folders: Sequence,
    *,
    context: str = 'evaluation',
) -> None:
    manifest_path = Path(run_dir) / 'test_eval_manifest.json'
    if not manifest_path.is_file():
        return
    manifest = load_test_eval_manifest(manifest_path)
    check = verify_same_test_set(manifest, test_folders)
    if not check['ok']:
        raise ValueError(
            f'{context}: test set fingerprint mismatch.\n'
            f'  run manifest: {check["reference_fingerprint"]}\n'
            f'  current:      {check["candidate_fingerprint"]}\n'
            f'Use the same seed/split_manifest or --split-from-run-dir.',
        )
