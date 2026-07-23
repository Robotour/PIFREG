"""Save per-session registered bands and displacement-field visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .experiment_data import sort_band_files
from .experiment_recorder import save_band_stacks, save_flow_fields


def _as_band_files(folder: Path, band_files: Optional[Sequence]) -> List[Path]:
    if band_files is not None:
        return [Path(p) for p in band_files]
    return sort_band_files(folder)


def chain_steps_to_flow_stack(chain_steps: Sequence[dict]) -> tuple[np.ndarray, List[int]]:
    flows = [np.asarray(step['flow'], dtype=np.float32) for step in chain_steps]
    moving_indices = [int(step['moving_idx']) for step in chain_steps]
    return np.stack(flows, axis=0), moving_indices


def elastix_fields_to_flow_stack(
    fields: Sequence,
    anchor_idx: int,
) -> tuple[np.ndarray, List[int]]:
    moving_indices = [i for i in range(len(fields)) if i != anchor_idx]
    flows = []
    for i in moving_indices:
        field_x, field_y = fields[i]
        flows.append(np.stack([np.asarray(field_x, dtype=np.float32),
                               np.asarray(field_y, dtype=np.float32)], axis=0))
    return np.stack(flows, axis=0), moving_indices


def save_session_registration_outputs(
    out_dir: Path,
    bands_raw_before: Sequence[np.ndarray],
    bands_raw_after: Sequence[np.ndarray],
    band_files: Sequence,
    *,
    chain_steps: Optional[Sequence[dict]] = None,
    elastix_fields: Optional[Sequence] = None,
    transform_meta: Optional[dict] = None,
    anchor_idx: Optional[int] = None,
    descending: bool = True,
    save_before_bands: bool = True,
) -> Dict[str, Any]:
    """
    Save per-session artifacts under out_dir:

      bands/before/*.jpeg
      bands/after/*.jpeg
      flows/flow_stack.npy
      flows/color/{wavelength}_flow.png
      flows/magnitude/{wavelength}_magnitude.png
      transforms.json  (optional, for StackReg/KEREN)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    band_files = [Path(p) for p in band_files]
    n = len(band_files)
    if anchor_idx is None:
        anchor_idx = n - 1 if descending else 0

    paths = save_band_stacks(
        out_dir,
        bands_raw_before,
        bands_raw_after,
        band_files,
        save_before=save_before_bands,
    )

    result: Dict[str, Any] = {
        'bands_before_dir': str(paths.get('before', '')),
        'bands_after_dir': str(paths.get('after', '')),
    }

    if chain_steps:
        flow_stack, moving_indices = chain_steps_to_flow_stack(chain_steps)
        flow_paths = save_flow_fields(out_dir, flow_stack, band_files, moving_indices, anchor_idx)
        result['flows'] = {k: str(v) for k, v in flow_paths.items()}
    elif elastix_fields is not None:
        flow_stack, moving_indices = elastix_fields_to_flow_stack(elastix_fields, anchor_idx)
        if flow_stack.size > 0:
            flow_paths = save_flow_fields(out_dir, flow_stack, band_files, moving_indices, anchor_idx)
            result['flows'] = {k: str(v) for k, v in flow_paths.items()}

    if transform_meta is not None:
        transform_path = out_dir / 'transforms.json'
        with open(transform_path, 'w', encoding='utf-8') as f:
            json.dump(transform_meta, f, indent=2)
        result['transforms'] = str(transform_path)

    meta = {
        'anchor_band_idx': anchor_idx,
        'anchor_wavelength': band_files[anchor_idx].stem if band_files else None,
        'num_bands': n,
        **{k: v for k, v in result.items()},
    }
    meta_path = out_dir / 'session_outputs.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    result['meta'] = str(meta_path)
    return result


def print_metrics_summary(title: str, summary: dict, metric_keys=('MI', 'NMI', 'NCC', 'NTG', 'MSE')) -> None:
    parts = [f'{k}={summary[k]:.6f}' for k in metric_keys if k in summary]
    print(f'{title}: ' + '  '.join(parts), flush=True)
