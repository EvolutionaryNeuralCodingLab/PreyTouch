#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_one(run_model_py: Path, folder: str, calib_dir: str, model: str = "deeplabcut", yes: bool = True, model_path_override=None):
    cmd = [
        sys.executable, str(run_model_py),
        "-m", model,
        "-p", folder,
        "--calib_dir", calib_dir,
    ]
    # if model_path_override:
    #     cmd.append(f"--model_path_override '{model_path_override}'")
    # else:
    #     cmd.append(f"-m '{model}'")
    if model_path_override:
        cmd.extend(["--model_path_override", str(model_path_override)])
    
    if yes:
        cmd.append("-y")

    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_csv", required=True,
                    help="Path to deeplabcut_jobs.csv created by MATLAB")
    ap.add_argument("--model", default="deeplabcut",
                    help="Model name for -m (default: deeplabcut)")
    ap.add_argument("--no_yes", action="store_true",
                    help="Do not pass -y")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Continue even if a job fails")
    ap.add_argument("--hpc",default='1', help="check true for HPC run")
    ap.add_argument('--model_path_override', default=None, help='optional model path to override config') # folder optional prefix path

    args = ap.parse_args()
    hpc=True if args.hpc=='1' else False
    print("It's me!",hpc)

    jobs_csv = Path(args.jobs_csv)

    # --- resolve run_model.py in the SAME folder as this script ---
    script_dir = Path(__file__).resolve().parent
    run_model_py = script_dir / "run_model.py"

    if not jobs_csv.exists():
        raise FileNotFoundError(f"jobs_csv not found: {jobs_csv}")
    if not run_model_py.exists():
        raise FileNotFoundError(f"run_model.py not found next to batch script: {run_model_py}")

    failures = 0

    with jobs_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"folder", "calib_dir"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain columns {required_cols}, got {reader.fieldnames}"
            )

        for i, row in enumerate(reader, start=1):
            folder = row["folder"].strip()
            calib_dir = row["calib_dir"].strip()

            if not folder or not calib_dir:
                print(f"[SKIP] row {i}: missing folder/calib_dir")
                continue
            if hpc: 
                folder = row["folder_hpc"].strip()
                calib_dir = row["calib_dir_hpc"].strip()

            try:
                run_one(
                    run_model_py=run_model_py,
                    folder=folder,
                    calib_dir=calib_dir,
                    model=args.model,
                    yes=not args.no_yes,
                    model_path_override=args.model_path_override

                )
            except subprocess.CalledProcessError as e:
                failures += 1
                print(f"[FAIL] row {i}: {e}")
                if not args.continue_on_error:
                    raise

    print(f"\nDone. Failures: {failures}")


if __name__ == "__main__":
    main()
