#!/usr/bin/env python3
import os
import glob
import argparse
import subprocess
import sys


def find_checkpoint(run_path: str):
    ckpt_dir = os.path.join(run_path, "checkpoints")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Get all .ckpt files except "last.ckpt"
    ckpt_files = [f for f in glob.glob(os.path.join(ckpt_dir, "*.ckpt")) if not f.endswith("last.ckpt")]

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No valid checkpoint file found in {ckpt_dir}")
    elif len(ckpt_files) > 1:
        raise RuntimeError(
            f"Multiple .ckpt files found in {ckpt_dir} (not including last.ckpt):\n"
            + "\n".join(ckpt_files)
        )

    return ckpt_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run predictions for k-fold W&B runs using saved checkpoints."
    )
    parser.add_argument("--run_ids", nargs="+", required=True, help="List of W&B run IDs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of k-folds")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where predictions will be stored",
    )
    args = parser.parse_args()

    base_log_dir = "logs/hypersynth"
    output_dir = args.output_dir
    run_ids = args.run_ids

    # If only one wandb run is given, construct the list by appending _fold{i}
    if len(run_ids) == 1:
        base_run = run_ids[0]
        run_ids = [f"{base_run}_fold{i}" for i in range(args.k_folds)]

    # Validate the number of folds
    if len(run_ids) != args.k_folds:
        print(
            f"Error: number of run IDs ({len(run_ids)}) does not match the number of k-folds ({args.k_folds})."
        )
        sys.exit(1)

    missing_folders = []
    existing_runs = []

    for run_id in run_ids:
        run_path = os.path.join(base_log_dir, run_id)
        if not os.path.isdir(run_path):
            missing_folders.append(run_path)
        else:
            existing_runs.append(run_id)

    if missing_folders:
        print("Error: the following run directories do not exist:")
        for missing in missing_folders:
            print(f" - {missing}")
        sys.exit(1)

    # Process each run
    for run_id in existing_runs:
        run_path = os.path.join(base_log_dir, run_id)
        config_path = os.path.join(run_path, "config.yaml")

        if not os.path.exists(config_path):
            print(f"Warning: config.yaml missing for run {run_id}, skipping.")
            continue

        try:
            ckpt_path = find_checkpoint(run_path)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error for run {run_id}:\n{e}\n")
            continue

        # Construct command
        cmd = [
            "python",
            "src/main.py",
            "predict",
            "-c",
            config_path,
            f'--ckpt_path={ckpt_path}',
            "--trainer.logger=false",
            f"--model.init_args.pred_output_dir={os.path.join(output_dir, run_id)}",
        ]

        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"

        print(f"Running prediction for {run_id}...")
        subprocess.run(cmd, env=env, check=True)
        print(f"Finished prediction for {run_id}\n")

    print("All folds processed successfully!")


if __name__ == "__main__":
    main()
