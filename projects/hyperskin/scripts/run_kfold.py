#!/usr/bin/env python3
import argparse
import os
import random
import string
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def random_run_id(length: int = 8) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def run_fold(
    fold: int,
    run_id: str,
    model_config: str,
    data_config: str,
    synthetic_base_dir: str | None,
    synth_ratio: float,
) -> int:
    run_name = f"{run_id}_kfold{fold}"
    cmd = [
        "python",
        "src/main.py",
        "fit",
        "-c",
        data_config,
        "-c",
        model_config,
        f"--data.init_args.current_fold={fold}",
        f"--trainer.logger.init_args.id={run_name}",
    ]

    if synthetic_base_dir:
        fold_suffix = f"_fold{fold}"
        synth_dir = None
        for item in os.listdir(synthetic_base_dir):
            path = os.path.join(synthetic_base_dir, item)
            if os.path.isdir(path) and item.endswith(fold_suffix):
                synth_dir = path
                break

        if synth_dir:
            cmd.append(f"--data.init_args.synthetic_data_dir={synth_dir}")
            cmd.append(f"--data.init_args.synth_ratio={synth_ratio}")
            print(f"[fold {fold}] Using synthetic data from '{synth_dir}'")
        else:
            print(f"[fold {fold}] WARNING: No matching synthetic data directory found, skipping synthetic data")

    print(f"[fold {fold}] Starting ({run_name})")
    print(f"[fold {fold}] Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"[fold {fold}] Finished successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[fold {fold}] FAILED with exit code {e.returncode}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run k-fold training with optional synthetic data directories and adjustable parallelism."
    )

    parser.add_argument("model_config", help="Path to model config YAML.")
    parser.add_argument("data_config", help="Path to data config YAML.")
    parser.add_argument(
        "--num_folds", type=int, default=5, help="Number of folds (default: 5)."
    )
    parser.add_argument(
        "--synthetic_base_dir",
        type=str,
        default=None,
        help="Base directory containing synthetic data subfolders like *_fold0, *_fold1, ...",
    )
    parser.add_argument(
        "--synth_ratio", type=float, default=1.0, help="Synthetic data ratio (default: 1.0)."
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=1,
        help="How many folds to run simultaneously (default: 1).",
    )

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.model_config):
        sys.exit(f"Model config not found: {args.model_config}")
    if not os.path.exists(args.data_config):
        sys.exit(f"Data config not found: {args.data_config}")
    if args.parallel < 1:
        sys.exit("--parallel must be 1 or greater")
    if args.synthetic_base_dir and not os.path.isdir(args.synthetic_base_dir):
        sys.exit(f"Synthetic base dir '{args.synthetic_base_dir}' not found.")

    run_id = random_run_id()

    print()
    print("=== K-FOLD TRAINING LAUNCHER ===")
    print(f"Model config:       {args.model_config}")
    print(f"Data config:        {args.data_config}")
    print(f"Num folds:          {args.num_folds}")
    print(f"Parallel jobs:      {args.parallel}")
    if args.synthetic_base_dir:
        print(f"Synthetic base dir: {args.synthetic_base_dir}")
        print(f"Synth ratio:        {args.synth_ratio}")
    else:
        print("Synthetic data:     None (no augmentation dirs passed)")
    print(f"Run ID prefix:      {run_id}")
    print()

    folds = list(range(args.num_folds))
    results = {}

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        future_map = {
            executor.submit(
                run_fold,
                fold,
                run_id,
                args.model_config,
                args.data_config,
                args.synthetic_base_dir,
                args.synth_ratio,
            ): fold
            for fold in folds
        }

        for future in as_completed(future_map):
            fold = future_map[future]
            try:
                code = future.result()
                results[fold] = code
            except Exception as e:
                print(f"[fold {fold}] CRASHED: {e}")
                results[fold] = 1

    print("\n=== SUMMARY ===")
    for fold, code in sorted(results.items()):
        status = "OK" if code == 0 else "FAILED"
        print(f"Fold {fold}: {status} (exit {code})")

    if all(code == 0 for code in results.values()):
        print("\nAll folds completed successfully!")
    else:
        print("\nSome folds failed.")


if __name__ == "__main__":
    main()