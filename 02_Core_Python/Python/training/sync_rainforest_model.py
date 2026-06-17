"""
sync_rainforest_model.py

Consolidates a trained Rainforest model (model.pkl + meta.json) into a single
standalone pickle file at models/rainforest_SYMBOL.pkl, ready for inference.

Eliminates the fragile batch one-liners that were previously in SUPREME_GO.bat
by providing a clean, testable Python entry point.

Usage:
    python -m Python.training.sync_rainforest_model --symbol BTCUSDm
"""

import argparse
import json
import pathlib
import sys

import joblib as jb


def find_latest_model_dir(base_dir: str | pathlib.Path, symbol: str | None = None) -> pathlib.Path:
    """
    Return the most recently modified subdirectory under base_dir.

    If *symbol* is provided, only directories whose meta.json ``dataset_id``
    contains the symbol (case-insensitive) are considered.  This ensures that
    ``run --symbol BTCUSDm`` syncs the *latest BTCUSDm* model, not just the
    latest model of any symbol.
    """
    base = pathlib.Path(base_dir)
    candidates: list[pathlib.Path] = []

    for p in base.iterdir():
        if not p.is_dir():
            continue

        # If a symbol filter is active, read meta.json and check dataset_id
        if symbol is not None:
            meta_path = p / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_bytes())
                ds_id = meta.get("dataset_id", "")
                if symbol.lower() not in ds_id.lower():
                    continue  # not a model for this symbol
            except (json.JSONDecodeError, OSError):
                continue  # can't read meta, skip

        candidates.append(p)

    if not candidates:
        msg = (
            f"ERROR: No model directories found under {base}"
            if symbol is None
            else f"ERROR: No model directories for symbol '{symbol}' under {base}"
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    return max(candidates, key=lambda p: p.stat().st_mtime)


def discover_symbols(models_root: str | pathlib.Path) -> list[str]:
    """
    Scan rainforest model directories and return the sorted unique symbols
    found in their ``dataset_id`` meta.json fields.

    Only directories that have a readable meta.json with a non-empty
    dataset_id are considered.  Symbols are extracted by stripping the
    common ``mt5_`` (or similar) prefix so that, e.g., ``mt5_BTCUSDm``
    yields ``BTCUSDm``.
    """
    rf_dir = pathlib.Path(models_root) / "rainforest"
    if not rf_dir.is_dir():
        return []

    symbols: set[str] = set()
    for p in rf_dir.iterdir():
        if not p.is_dir():
            continue
        meta_path = p / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_bytes())
            ds_id = meta.get("dataset_id", "")
            # Strip common prefix like "mt5_" to get the bare symbol
            if "_" in ds_id:
                sym = ds_id.split("_", 1)[1]
            else:
                sym = ds_id
            if sym:
                symbols.add(sym)
        except (json.JSONDecodeError, OSError):
            continue

    return sorted(symbols)


def sync_model(symbol: str) -> None:
    """
    Locate the latest trained model for *symbol*, read its meta.json and
    model.pkl, and write a consolidated pickle to models/rainforest_SYMBOL.pkl.
    """
    models_root = pathlib.Path("models")
    rf_dir = models_root / "rainforest"
    if not rf_dir.is_dir():
        print(f"ERROR: Rainforest directory not found: {rf_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    latest = find_latest_model_dir(rf_dir, symbol=symbol)

    meta_path = latest / "meta.json"
    model_path = latest / "model.pkl"

    if not meta_path.is_file():
        print(f"ERROR: meta.json not found in {latest}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_file():
        print(f"ERROR: model.pkl not found in {latest}", file=sys.stderr)
        sys.exit(1)

    # Load artifacts
    meta = json.loads(meta_path.read_bytes())
    p = jb.load(model_path)
    model_data = p["model"]  # RandomForestClassifier instance

    # Build consolidated output
    output = {
        "model": model_data,
        "classes": list(model_data.classes_),
        "feature_importances": {
            f"f_{i}": round(float(v), 6)
            for i, v in enumerate(model_data.feature_importances_)
        },
        "oob_score": meta.get("oob_score"),
        "trained_at": meta.get("trained_at"),
        "n_estimators": meta.get("n_estimators"),
        "max_depth": meta.get("max_depth"),
    }

    out_path = models_root / f"rainforest_{symbol}.pkl"
    jb.dump(output, str(out_path))

    oob = meta.get("oob_score", "N/A")
    classes_count = len(model_data.classes_)
    oob_str = f"{oob:.4f}" if isinstance(oob, (int, float)) else str(oob)
    print(f"      {symbol} trained | OOB: {oob_str} | {classes_count} classes")
    print(f"      Synced to: {out_path}")


def sync_all() -> None:
    """
    Discover every symbol with a trained model under ``models/rainforest/``
    and sync the most recent one for each symbol.
    """
    models_root = pathlib.Path("models")
    symbols = discover_symbols(models_root)
    if not symbols:
        print("ERROR: No symbols found -- no trained models detected.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(symbols)} symbol(s): {', '.join(symbols)}")
    print()
    for sym in symbols:
        sync_model(sym)
    print()
    print("All symbols synced.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate latest Rainforest model into a standalone pickle."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--symbol",
        help="Trading symbol (e.g. BTCUSDm, XAUUSDm).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Sync *all* symbols that have trained models under models/rainforest/.",
    )
    args = parser.parse_args()

    if args.all:
        sync_all()
    else:
        sync_model(args.symbol)


if __name__ == "__main__":
    main()
