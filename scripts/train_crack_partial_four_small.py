"""
Train the four small ablation models with an external crack-only partial-label
dataset added as extra training data.

The external dataset should usually contain only img_dir/train and ann_dir/train.
Its non-crack pixels are expected to be 255 ignore, so validation remains driven
by the fully labeled main Tongji validation split.

Example:
    python scripts/train_crack_partial_four_small.py \
      --crack_data_root dataset/roboflow_tunnel_crack_partial_raw \
      --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODEL_RUNS = (
    {
        "name": "benchmark_small_crack_partial",
        "output_dir": "outputs/ablation_benchmark_small_crack_partial",
        "args": [],
        "rare_class_weights": None,
    },
    {
        "name": "tmds_full_small_crack_partial",
        "output_dir": "outputs/ablation_tmds_full_small_crack_partial",
        "args": ["--use_tmds", "--use_cmim", "--routing_loss_weight", "0.2"],
        "rare_class_weights": "1:3.0,3:2.0",
    },
    {
        "name": "tmds_no_cmim_small_crack_partial",
        "output_dir": "outputs/ablation_tmds_no_cmim_small_crack_partial",
        "args": ["--use_tmds", "--no-use_cmim", "--routing_loss_weight", "0.6"],
        "rare_class_weights": "1:3.0,3:2.0",
    },
    {
        "name": "tmds_no_routing_small_crack_partial",
        "output_dir": "outputs/ablation_tmds_no_routing_small_crack_partial",
        "args": ["--use_tmds", "--use_cmim", "--routing_loss_weight", "0.0"],
        "rare_class_weights": "1:3.0,3:2.0",
    },
)


def _build_command(run: dict, args: argparse.Namespace) -> list[str]:
    extra_roots = [str(args.crack_data_root)]
    if args.roboflow_tongji_root:
        extra_roots.insert(0, str(args.roboflow_tongji_root))

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--data_root",
        str(args.data_root),
        "--extra_data_roots",
        *extra_roots,
        "--output_dir",
        run["output_dir"],
        "--head_channels",
        str(args.head_channels),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    cmd.extend(run["args"])
    rare_class_weights = run["rare_class_weights"]
    if rare_class_weights:
        cmd.extend(["--rare_class_weights", rare_class_weights])
    if args.max_steps is not None:
        cmd.extend(["--max_steps", str(args.max_steps)])
    if args.val_interval is not None:
        cmd.extend(["--val_interval", str(args.val_interval)])
    if args.stage_epochs is not None and "--use_tmds" in run["args"]:
        cmd.extend(["--stage_epochs", args.stage_epochs])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train four small models with external crack partial labels.")
    parser.add_argument("--data_root", type=Path, default=Path("dataset/tongji_data_awesome"))
    parser.add_argument("--crack_data_root", type=Path, default=Path("dataset/roboflow_tunnel_crack_partial_raw"))
    parser.add_argument(
        "--roboflow_tongji_root",
        type=Path,
        default=None,
        help="Optional extra Roboflow Tongji raw data root to include before the crack-only data.",
    )
    parser.add_argument("--head_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--stage_epochs", default=None, help="Override TMDS/staged epochs, e.g. '1,0,0' for smoke tests.")
    parser.add_argument("--only", nargs="*", choices=[run["name"] for run in MODEL_RUNS], default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.data_root.exists() and not args.dry_run:
        raise FileNotFoundError(f"Main data_root not found: {args.data_root}")
    if not (args.crack_data_root / "img_dir" / "train").exists() and not args.dry_run:
        raise FileNotFoundError(
            f"Crack partial train split not found: {args.crack_data_root / 'img_dir' / 'train'}\n"
            "Run data_tools/prepare_roboflow_crack_partial_raw.py first."
        )
    if args.roboflow_tongji_root and not args.roboflow_tongji_root.exists() and not args.dry_run:
        raise FileNotFoundError(f"Roboflow Tongji root not found: {args.roboflow_tongji_root}")

    selected = [run for run in MODEL_RUNS if args.only is None or run["name"] in args.only]
    for idx, run in enumerate(selected, 1):
        cmd = _build_command(run, args)
        print("=" * 80)
        print(f"[{idx}/{len(selected)}] {run['name']}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
