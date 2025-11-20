import itertools
import csv
from pathlib import Path
import random

OUT = Path("sweep_configs.csv")
space = {
    "lr": [1e-3, 5e-4, 1e-4],
    "batch": [4, 8, 16, 24],
    "epochs": [50, 100],
    "imgsz": [1024],
}

DO_RANDOM = False
N_RANDOM = 20

def expand_grid(space):
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def random_sample(space, n):
    keys = list(space.keys())
    for _ in range(n):
        yield {k: random.choice(space[k]) for k in keys}

configs = list(random_sample(space, N_RANDOM)) if DO_RANDOM else list(expand_grid(space))

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(configs[0].keys()))
    writer.writeheader()
    for row in configs:
        writer.writerow(row)

print(f"Wrote {len(configs)} configs to {OUT}")
