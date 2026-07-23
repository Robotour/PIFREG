"""VoxelMorph HSI experiment run directory layout and post-train pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .networks import VxmDense
from .training import (
    build_adjacent_band_pairs,
    discover_band_folders,
    evaluate_voxelmorph_pairs,
    evaluate_voxelmorph_sessions,
    save_split_manifest,
    split_folders_train_test,
    train_voxelmorph_baseline,
    train_voxelmorph_stack_spatial,
)


def create_run_dir(
    project_root: Path,
    method: str,
    exp_name: str,
    runs_root: Optional[Path] = None,
) -> Path:
    """
    outputs/voxelmorph_runs/{method}/{exp_name}_{timestamp}/
      config.json
      split_manifest.json
      train_history.json
      best_info.json
      test_metrics.json
      checkpoints/best.pt, final.pt, ...
      visualizations/
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = runs_root or (project_root / 'outputs' / 'voxelmorph_runs')
    run_dir = base / method / f'{exp_name}_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'visualizations').mkdir(exist_ok=True)
    return run_dir


def prepare_data_split(
    data_root: Path,
    run_dir: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
    image_size=(512, 512),
) -> Dict[str, Any]:
    folders = discover_band_folders([data_root])
    train_folders, test_folders = split_folders_train_test(
        folders, train_ratio=train_ratio, seed=seed,
    )
    save_split_manifest(
        run_dir / 'split_manifest.json',
        train_folders,
        test_folders,
        train_ratio,
        seed,
    )
    train_pairs = build_adjacent_band_pairs(train_folders, image_size=image_size)
    test_pairs = build_adjacent_band_pairs(test_folders, image_size=image_size)
    return {
        'folders': folders,
        'train_folders': train_folders,
        'test_folders': test_folders,
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
        'inshape': train_pairs[0][0].shape,
    }


def train_method(
    method: str,
    run_dir: Path,
    data_bundle: Dict[str, Any],
    train_kwargs: Dict[str, Any],
):
    run_dir = Path(run_dir)
    common = dict(
        pairs=data_bundle['train_pairs'],
        model_dir=run_dir,
        inshape=data_bundle['inshape'],
        load_model=train_kwargs.get('load_model'),
        device=train_kwargs.get('device', 'cuda'),
        epochs=train_kwargs.get('epochs', 300),
        steps_per_epoch=train_kwargs.get('steps_per_epoch', 80),
        lr=train_kwargs.get('lr', 1e-4),
        image_loss=train_kwargs.get('image_loss', 'ncc'),
        lamda=train_kwargs.get('lamda', 0.01),
        int_steps=train_kwargs.get('int_steps', 7),
        int_downsize=train_kwargs.get('int_downsize', 2),
        val_interval=train_kwargs.get('val_interval', 20),
    )

    if method == 'baseline':
        return train_voxelmorph_baseline(
            **common,
            val_pairs=data_bundle['test_pairs'],
            val_steps=train_kwargs.get('val_steps', 100),
        )
    if method == 'stack_spatial':
        return train_voxelmorph_stack_spatial(
            **common,
            train_folders=data_bundle['train_folders'],
            val_folders=data_bundle['test_folders'],
            subchain_len=train_kwargs.get('subchain_len', 6),
            center_floor=train_kwargs.get('center_floor', 0.35),
            edge_gain=train_kwargs.get('edge_gain', 1.0),
            chain_descending=train_kwargs.get('chain_descending', True),
            val_session_steps=train_kwargs.get('val_session_steps', 15),
            image_loss=train_kwargs.get('image_loss', 'mse'),
        )
    raise ValueError(f'Unknown method: {method}. Use baseline or stack_spatial.')


def evaluate_run(
    run_dir: Path,
    data_bundle: Dict[str, Any],
    checkpoint: Path,
    device: str = 'cuda',
    image_size=(512, 512),
    chain_descending: bool = True,
    smooth_flow_sigma: float = 1.5,
    method_name: str = 'voxelmorph',
) -> Dict[str, Any]:
    from src.python.experiments.stack_pairwise_metrics import evaluate_test_sessions_all_pairs
    from .training import register_stack_with_voxelmorph_chain

    model = VxmDense.load(str(checkpoint), device)

    def _register_fn(bands):
        registered, _ = register_stack_with_voxelmorph_chain(
            model,
            bands,
            device=device,
            descending=chain_descending,
            smooth_flow_sigma=smooth_flow_sigma,
        )
        return registered

    pair_result = evaluate_voxelmorph_pairs(
        model,
        data_bundle['test_pairs'],
        device=device,
        max_pairs=None,
        verbose=True,
    )
    stack_result = evaluate_voxelmorph_sessions(
        model,
        data_bundle['test_folders'],
        device=device,
        image_size=image_size,
        descending=chain_descending,
        smooth_flow_sigma=smooth_flow_sigma,
        max_sessions=None,
        verbose=True,
    )
    all_pairs_eval = evaluate_test_sessions_all_pairs(
        data_bundle['test_folders'],
        image_size=image_size,
        register_fn=_register_fn,
        max_sessions=None,
        verbose=True,
    )
    payload = {
        'checkpoint': str(checkpoint.resolve()),
        'pair_eval': pair_result,
        'stack_eval': stack_result,
        'all_pairs_eval': all_pairs_eval,
        'method_name': method_name,
    }
    with open(run_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return payload


def visualize_all_test_sessions(
    run_dir: Path,
    checkpoint: Path,
    data_bundle: Dict[str, Any],
    project_root: Path,
    device: str = 'cuda',
    image_size=(512, 512),
    spectral_path: Optional[Path] = None,
    chain_descending: bool = True,
    smooth_flow_sigma: float = 1.5,
) -> Path:
    from src.python.experiments.visualize_voxelmorph_test import visualize_test_sessions

    viz_dir = Path(run_dir) / 'visualizations'
    spectral_path = spectral_path or (project_root / 'HSI2RGB20240517.xlsx')
    visualize_test_sessions(
        run_dir=run_dir,
        output_dir=viz_dir,
        checkpoint=str(checkpoint),
        image_size=image_size,
        device=device,
        all_test_sessions=True,
        test_folders=data_bundle['test_folders'],
        spectral_path=str(spectral_path),
        descending=chain_descending,
        smooth_flow_sigma=smooth_flow_sigma,
    )
    return viz_dir


def run_full_experiment(
    project_root: Path,
    method: str,
    exp_name: str,
    data_dir: str = 'data/cut_images_all',
    train_ratio: float = 0.7,
    seed: int = 42,
    image_size=(512, 512),
    train_kwargs: Optional[Dict[str, Any]] = None,
    chain_descending: bool = True,
    smooth_flow_sigma: float = 1.5,
    skip_train: bool = False,
    run_dir: Optional[Path] = None,
    metrics_csv: Optional[Path] = None,
    write_metrics_csv: bool = True,
    overwrite_csv_row: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Train -> save best.pt -> eval all test -> visualize all test sessions."""
    train_kwargs = dict(train_kwargs or {})
    data_root = project_root / data_dir
    run_dir = run_dir or create_run_dir(project_root, method, exp_name)
    data_bundle = prepare_data_split(
        data_root, run_dir, train_ratio=train_ratio, seed=seed, image_size=image_size,
    )

    config = {
        'method': method,
        'exp_name': exp_name,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'data_dir': str(data_root.resolve()),
        'run_dir': str(run_dir.resolve()),
        'train_ratio': train_ratio,
        'seed': seed,
        'image_size': list(image_size),
        'chain_descending': chain_descending,
        'smooth_flow_sigma': smooth_flow_sigma,
        'num_sessions': len(data_bundle['folders']),
        'num_train_sessions': len(data_bundle['train_folders']),
        'num_test_sessions': len(data_bundle['test_folders']),
        'train_kwargs': train_kwargs,
    }
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    if skip_train:
        best_path = run_dir / 'checkpoints' / 'best.pt'
        if not best_path.is_file():
            raise FileNotFoundError(f'Missing checkpoint for eval-only: {best_path}')
        best_info = json.loads((run_dir / 'best_info.json').read_text(encoding='utf-8'))
    else:
        _, best_path, best_info = train_method(method, run_dir, data_bundle, train_kwargs)

    best_path = Path(best_path)
    csv_method_name = f'voxelmorph_{method}'
    metrics = evaluate_run(
        run_dir,
        data_bundle,
        best_path,
        device=train_kwargs.get('device', 'cuda'),
        image_size=image_size,
        chain_descending=chain_descending,
        smooth_flow_sigma=smooth_flow_sigma,
        method_name=csv_method_name,
    )
    if write_metrics_csv:
        from src.python.experiments.metrics_csv import (
            append_method_row,
            default_metrics_csv_path,
            ensure_unregistered_row,
        )
        csv_path = metrics_csv or default_metrics_csv_path(project_root, seed)
        ensure_unregistered_row(
            csv_path,
            seed,
            data_bundle['test_folders'],
            image_size=image_size,
            verbose=True,
        )
        all_pairs = metrics['all_pairs_eval']
        append_method_row(
            csv_path,
            method=csv_method_name,
            seed=seed,
            summary_after=all_pairs['summary_after'],
            test_folders=data_bundle['test_folders'],
            image_size=image_size,
            run_dir=run_dir,
            notes='all C(N,2) band pairs mean; VoxelMorph chain inference',
            num_bands_mean=all_pairs.get('num_bands_mean'),
            num_pairs_mean=all_pairs.get('num_pairs_mean'),
            overwrite=overwrite_csv_row,
            verbose=True,
        )
        config['metrics_csv'] = str(csv_path.resolve())
        with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    visualize_all_test_sessions(
        run_dir,
        best_path,
        data_bundle,
        project_root,
        device=train_kwargs.get('device', 'cuda'),
        image_size=image_size,
        chain_descending=chain_descending,
        smooth_flow_sigma=smooth_flow_sigma,
    )

    summary = {
        'run_dir': str(run_dir.resolve()),
        'best_checkpoint': str(best_path.resolve()),
        'best_info': best_info,
        'test_metrics': metrics,
    }
    with open(run_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    return run_dir, summary
