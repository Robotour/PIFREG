#!/usr/bin/env python3
"""
对比多次实验的指标汇总。

扫描 outputs/<method>/runs/*/manifest.json，生成对比表 CSV 与 Markdown。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
METRIC_KEYS = ("MI", "NMI", "NCC", "NTG")


def collect_runs(outputs_root: Path, method_filter: str | None = None):
    rows = []
    if not outputs_root.exists():
        return rows
    for method_dir in sorted(outputs_root.iterdir()):
        if not method_dir.is_dir():
            continue
        if method_filter and method_dir.name != method_filter:
            continue
        runs_dir = method_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            summary = manifest.get("metrics_summary") or {}
            timing_path = run_dir / "timing.json"
            elapsed = None
            if timing_path.is_file():
                with open(timing_path, encoding="utf-8") as f:
                    elapsed = json.load(f).get("elapsed_seconds")
            row = {
                "method": method_dir.name,
                "run_name": manifest.get("run_name", run_dir.name),
                "experiment": manifest.get("experiment", ""),
                "exp_name": manifest.get("exp_name", ""),
                "run_dir": str(run_dir),
                "elapsed_s": elapsed,
            }
            for k in METRIC_KEYS:
                if k in summary:
                    row[f"{k}_before"] = summary[k].get("before")
                    row[f"{k}_after"] = summary[k].get("after")
                    row[f"{k}_delta"] = summary[k].get("delta")
            rows.append(row)
    return rows


def write_csv(rows, path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path: Path):
    if not rows:
        path.write_text("# Experiment Comparison\n\nNo runs found.\n", encoding="utf-8")
        return
    lines = [
        "# Experiment Comparison",
        "",
        "| Method | Run | MI Δ | NMI Δ | NCC Δ | NTG Δ | Time (min) |",
        "|--------|-----|------|-------|-------|-------|------------|",
    ]
    for r in rows:
        t_min = f"{r['elapsed_s'] / 60:.1f}" if r.get("elapsed_s") else "—"
        lines.append(
            f"| {r['method']} | {r['run_name']} | "
            f"{r.get('MI_delta', 0):+.4f} | {r.get('NMI_delta', 0):+.4f} | "
            f"{r.get('NCC_delta', 0):+.4f} | {r.get('NTG_delta', 0):+.4f} | {t_min} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Compare registration experiment runs")
    p.add_argument("--outputs-root", type=str, default=str(DEFAULT_OUTPUTS_ROOT))
    p.add_argument("--method", type=str, default=None, help="只对比某一方法目录，如 pifreg_groupwise_stackflow")
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument("--out-md", type=str, default=None)
    args = p.parse_args()

    outputs_root = Path(args.outputs_root)
    rows = collect_runs(outputs_root, args.method)

    out_csv = Path(args.out_csv) if args.out_csv else outputs_root / "comparison.csv"
    out_md = Path(args.out_md) if args.out_md else outputs_root / "comparison.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_csv)
    write_markdown(rows, out_md)

    print(f"Found {len(rows)} runs under {outputs_root}")
    print(f"  CSV: {out_csv}")
    print(f"  MD : {out_md}")
    if rows:
        print("\nQuick view (NCC delta):")
        for r in sorted(rows, key=lambda x: x.get("NCC_delta") or 0, reverse=True):
            print(f"  {r['method']}/{r['run_name']}: NCC Δ={r.get('NCC_delta', 0):+.4f}")


if __name__ == "__main__":
    main()
