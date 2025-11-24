import csv
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import signal
import sys

def handler(signum, frame):
    print(f"Received signal {signum}, saving checkpoint...")
    # Save checkpoint or partial results here
    sys.exit(1)

signal.signal(signal.SIGTERM, handler)

def load_sweep_config(csv_path, idx):
    with open(csv_path, newline='') as f:
        row = list(csv.DictReader(f))[int(idx)-1]

    for k, v in row.items():
        row[k] = float(v) if '.' in v else int(v)

    return row

def append_to_csv(args, params, multipleGpus=True, results=None):
    config_dict = {"epochs": int(params.get("epochs", 100)),
                    "batch": int(params.get("batch", 16)),
                    "lr": float(params.get("lr", 1e-3)),
                    "imgsz": int(params.get("imgsz", 640))}

    if multipleGpus:
        with open(args.workdir + "/run_" + args.idx + "/results.csv", "r", newline="", encoding="utf-8") as f:
            results_dict = list(csv.DictReader(f))[-1]
    else:
        results_dict = results.results_dict

    run_results_dict = {**config_dict, **results_dict}

    if int(os.environ.get("RANK")) == 0:
        file_exists = os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=run_results_dict.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(run_results_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--idx")
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--workdir")
    parser.add_argument("--gpuCount")
    parser.add_argument("--results_csv")
    args = parser.parse_args()

    run_params = load_sweep_config(args.csv, args.idx)
    print("Config loaded.")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=run_params["epochs"],
        batch=128, # int(run_params["batch"]*args.gpuCount),
        lr0=run_params["lr"],
        imgsz=run_params["imgsz"],
        project=args.workdir,
        name=f"run_{args.idx}",
        workers=16,
        device=list(range(int(args.gpuCount))) if args.gpuCount != "-1" else "cpu",
        optimizer="AdamW"
    )
    print("Training completed.")
    
    append_to_csv(args, run_params)
    print("CSV saved.")

if __name__ == "__main__":
    main()