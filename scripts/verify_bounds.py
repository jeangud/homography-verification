"""Verify the LARD network against precomputed PWL pixel bounds.

For each `*.pkl` file produced by `scripts/calculate_bounds.py`, build a
`PerturbationPWL` via `vnn.abcrown_adapter`, propagate CROWN bounds through
the network, and report SAFE / UNKNOWN for the spec "argmax stays at
the target class".

The target class is determined per-image from `--instances-csv` (which must
have an ``index`` and a ``label`` column). If no CSV is given, falls back to
a fixed `--target-class` for all images.

Usage (with the alpha-beta-CROWN conda env):

    PYTHONPATH=/path/to/alpha-beta-CROWN/auto_LiRPA:src \
        python scripts/verify_bounds.py \
            --bounds-dir   "<path-to-bounds-dir>" \
            --model        "<path-to-model.pt>"   \
            --instances-csv instances_full.csv      \
            --output-csv   results.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure both the consumer src and auto_LiRPA are importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from auto_LiRPA import BoundedModule, BoundedTensor  # noqa: E402

from vnn.abcrown_adapter import bounds_dict_to_perturbation  # noqa: E402


def _build_lard_model(num_classes: int = 10) -> nn.Module:
    """LARD CNN: 3x32x32 -> Conv(8,4,s2) -> ReLU -> Conv(16,4,s2) -> ReLU
    -> Flatten -> Linear(1024,100) -> ReLU -> Linear(100, num_classes).
    Mirrors the first architecture in `notebooks/train_lard.ipynb`.
    """
    return nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, num_classes),
    )


def _load_model(model_path: Path, device, num_classes: int = 10) -> nn.Module:
    """Load either a full `nn.Module` or a state_dict (LARD architecture)."""
    obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        # Assume state_dict for the LARD architecture.
        model = _build_lard_model(num_classes=num_classes)
        model.load_state_dict(obj)
    return model.to(device).eval()


def _sorted_pickles(bounds_dir: Path):
    """Return `*.pkl` files in `bounds_dir` sorted by int stem when possible."""
    pkls = list(bounds_dir.glob("*.pkl"))

    def _key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    return sorted(pkls, key=_key)


def _build_diff_matrix(num_classes: int, target: int, device, dtype):
    """C of shape (1, num_classes-1, num_classes) encoding y[target] - y[j]."""
    rows = []
    for j in range(num_classes):
        if j == target:
            continue
        row = torch.zeros(num_classes, dtype=dtype, device=device)
        row[target] = 1.0
        row[j] = -1.0
        rows.append(row)
    return torch.stack(rows, dim=0).unsqueeze(0)


def verify_one(
    pkl_path: Path,
    bounded_model: BoundedModule,
    raw_model: torch.nn.Module,
    target: int,
    num_classes: int,
    device,
    dtype,
    dataset=None,
):
    """Run incomplete CROWN verification on a single bounds pickle.

    Returns a dict with status, margin, and several diagnostic argmaxes:
      - midpoint_argmax: argmax on (x_L + x_U) / 2 (geometric center of the
        per-pixel PWL envelope; *not* an actual transformed image).
      - pickle_t0_argmax: argmax on the pickle sample at parameter t = 0
        (the true original, untransformed image).
      - dataset_argmax / dataset_label: argmax on the same image loaded
        directly from `dataset[idx_img]`, and its ground-truth label.
        Both are -1 when `dataset is None`.

    Raises ValueError if no sample at exactly t=0 exists in the pickle.
    """
    with Path(pkl_path).open("rb") as f:
        data = pickle.load(f)

    ptb, x_mid, _ = bounds_dict_to_perturbation(
        data["bounds"]["pwl"]["sound"],
        device=device,
        dtype=dtype,
    )
    bt = BoundedTensor(x_mid, ptb)

    C = _build_diff_matrix(num_classes, target, device, dtype)

    with torch.no_grad():
        mid_argmax = int(raw_model(x_mid).argmax(dim=-1).item())

        # t = 0 image from the pickle (samples[i] corresponds to params[i]).
        params = np.asarray(data["bounds"]["params"])
        t0_mask = params == 0.0
        if not t0_mask.any():
            raise ValueError(
                f"No sample at t=0 in {pkl_path.name}: "
                f"params range [{params.min()}, {params.max()}]"
            )
        t0_idx = int(np.argmax(t0_mask))  # first index where param == 0
        t0_img = torch.as_tensor(
            data["bounds"]["samples"][t0_idx],
            dtype=dtype,
            device=device,
        ).unsqueeze(0)
        pickle_t0_argmax = int(raw_model(t0_img).argmax(dim=-1).item())

        # Optional: same image fetched directly from the dataset.
        dataset_argmax = -1
        dataset_label = -1
        if dataset is not None:
            idx_img = int(data["idx_img"])
            ds_img, ds_label = dataset[idx_img]
            ds_img = ds_img.to(device=device, dtype=dtype).unsqueeze(0)
            dataset_argmax = int(raw_model(ds_img).argmax(dim=-1).item())
            dataset_label = int(ds_label)

    lb, _ = bounded_model.compute_bounds(x=(bt,), C=C, method="backward")
    min_margin = float(lb.min().item())
    status = "SAFE" if min_margin > 0.0 else "UNKNOWN"
    return {
        "status": status,
        "min_margin": min_margin,
        "midpoint_argmax": mid_argmax,
        "pickle_t0_argmax": pickle_t0_argmax,
        "dataset_argmax": dataset_argmax,
        "dataset_label": dataset_label,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--instances-csv",
        type=Path,
        default=None,
        help="CSV with 'index' and 'label' columns giving per-image target classes.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Fixed target class for all images (fallback "
        "when --instances-csv is not given).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only verify the first N pickles (smoke runs).",
    )
    parser.add_argument(
        "--dataset",
        default="LARD",
        help="Name of a class in `vnn.datasets` to cross-check t=0 argmax. "
        "Pass 'none' to skip dataset loading.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split passed to the dataset constructor.",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.float32

    # Load model (full nn.Module or LARD state_dict).
    raw_model = _load_model(args.model, device)

    # Per-image target class lookup ------------------------------------------------
    if args.instances_csv is not None:
        inst_df = pd.read_csv(args.instances_csv)
        if "index" not in inst_df.columns or "label" not in inst_df.columns:
            parser.error("--instances-csv must have 'index' and 'label' columns")
        target_map: dict[int, int] = dict(
            zip(inst_df["index"].astype(int), inst_df["label"].astype(int))
        )
        print(
            f"Loaded {len(target_map)} per-image target classes from {args.instances_csv}"
        )
    elif args.target_class is not None:
        target_map = None  # will use the fixed fallback
        print(f"Using fixed target class {args.target_class} for all images")
    else:
        parser.error("Must provide either --instances-csv or --target-class.")
        return 1  # unreachable, keeps type checkers happy

    # Optional dataset for the second t=0 cross-check.
    dataset = None
    if args.dataset.lower() != "none":
        try:
            import vnn.datasets as _ds_mod

            ds_cls = getattr(_ds_mod, args.dataset)
            dataset = ds_cls(split=args.split)
            print(
                f"Loaded dataset {args.dataset}(split={args.split!r}) with {len(dataset)} items"
            )
        except Exception as e:
            print(f"WARNING: could not load dataset {args.dataset!r}: {e!r}")
            dataset = None

    pkls = _sorted_pickles(args.bounds_dir)
    if args.limit is not None:
        pkls = pkls[: args.limit]
    if not pkls:
        print(f"No *.pkl files in {args.bounds_dir}", file=sys.stderr)
        return 1

    # Auto-detect input shape & output dim using the first pickle.
    with pkls[0].open("rb") as _f:
        _first_data = pickle.load(_f)
    first_ptb, first_x_mid, chw = bounds_dict_to_perturbation(
        _first_data["bounds"]["pwl"]["sound"],
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        y0 = raw_model(first_x_mid)
    num_classes = int(y0.shape[-1])
    print(f"Detected input CHW={chw}, num_classes={num_classes}, device={device}")

    # Wrap model ONCE. Use the efficient `patches` conv mode (the matrix mode
    # would OOM on CIFAR-scale CNNs).
    dummy = torch.zeros(1, *chw, dtype=dtype, device=device)
    bounded_model = BoundedModule(raw_model, dummy, bound_opts={"conv_mode": "patches"})

    rows = []
    counts = {"SAFE": 0, "UNKNOWN": 0, "ERROR": 0}
    for i, pkl in enumerate(pkls):
        # Resolve target class for this image.
        img_id = pkl.stem
        if target_map is not None:
            try:
                img_key = int(img_id)
            except ValueError:
                img_key = img_id  # type: ignore[assignment]
            if img_key not in target_map:
                print(f"WARNING: image {img_id} not found in instances CSV, skipping.")
                continue
            target = target_map[img_key]
        else:
            target = args.target_class

        t0 = time.perf_counter()
        try:
            out = verify_one(
                pkl,
                bounded_model,
                raw_model,
                target,
                num_classes,
                device,
                dtype,
                dataset=dataset,
            )
            err = ""
        except Exception as e:
            out = {
                "status": "ERROR",
                "min_margin": float("nan"),
                "midpoint_argmax": -1,
                "pickle_t0_argmax": -1,
                "dataset_argmax": -1,
                "dataset_label": -1,
            }
            err = repr(e)
        dt = time.perf_counter() - t0
        status = out["status"]
        counts[status] = counts.get(status, 0) + 1
        flag = (
            ""
            if out["midpoint_argmax"] == target or status == "ERROR"
            else " (mid_argmax != target!)"
        )
        print(
            f"[{i + 1}/{len(pkls)}] {img_id}: {status}  target={target}  "
            f"margin={out['min_margin']:+.4e}  "
            f"mid={out['midpoint_argmax']} t0_pkl={out['pickle_t0_argmax']} "
            f"t0_ds={out['dataset_argmax']} label={out['dataset_label']}  "
            f"t={dt:.2f}s{flag} {err}"
        )
        rows.append(
            {
                "image_id": img_id,
                "target_class": target,
                "status": status,
                "min_margin": out["min_margin"],
                "midpoint_argmax": out["midpoint_argmax"],
                "pickle_t0_argmax": out["pickle_t0_argmax"],
                "dataset_argmax": out["dataset_argmax"],
                "dataset_label": out["dataset_label"],
                "runtime_s": dt,
                "error": err,
            }
        )

    print("\nSummary:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.output_csv}")

    return 0


if __name__ == "__main__":
    # Suppress ResourceWarning from TemporaryDirectory objects created (and not
    # explicitly cleaned up) by PyTorch internals during JIT tracing.
    warnings.filterwarnings(
        "ignore", category=ResourceWarning, message=".*TemporaryDirectory.*"
    )
    sys.exit(main())
