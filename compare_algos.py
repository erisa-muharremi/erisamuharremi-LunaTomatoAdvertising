import csv
import os
import sys
import time
import numpy as np
import solve_lta as s
from lta_udp import udp

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
MIN = 2 * scale
os.makedirs("results", exist_ok=True)
rows = []


def evaluate(best):
    x = s.build_x(best)
    obj, num_sel, mse = udp.fitness(x, postprocess=True)
    return int(num_sel), round(float(mse), 4)


def run_greedy(seed):
    s.rng = np.random.default_rng(seed)
    crafts, counts, total = s.greedy()
    mse = s.mse_of(total)
    return dict(algoritem="greedy vetem", seed=seed, anije=len(crafts),
                mse=round(mse, 4), e_vlefshme="po" if mse <= 0.05 else "jo")


def run_anneal(seed, name, t0, t1):
    s.rng = np.random.default_rng(seed)
    crafts, counts, total = s.greedy()
    n, best = s.anneal(crafts, counts, total, MIN * 60, temp0=t0, temp1=t1)
    if best is None:
        return dict(algoritem=name, seed=seed, anije="", mse="", e_vlefshme="jo")
    n, mse = evaluate(best)
    return dict(algoritem=name, seed=seed, anije=n, mse=mse, e_vlefshme="po")


def run_random(seed, k=40):
    s.rng = np.random.default_rng(seed)
    best_mse, t0 = 1.0, time.time()
    while time.time() - t0 < MIN * 60 / 2:
        counts = np.zeros(s.N, dtype=int)
        total = np.zeros_like(s.TARGET)
        for _ in range(k):
            fam, i = s.random_orbit(counts)
            counts[s.FAM_OFF[fam] + i] += 1
            total = total + s.craft_signal(fam, i, s.rng.uniform(0, s.TWO_PI))
        best_mse = min(best_mse, s.mse_of(total))
    return dict(algoritem="random search (40 anije)", seed=seed, anije=k,
                mse=round(best_mse, 4), e_vlefshme="po" if best_mse <= 0.05 else "jo")


for seed in [1, 2, 3]:
    for r in (run_greedy(seed),
              run_random(seed),
              run_anneal(seed, "hill climbing", 0.001, 0.001),
              run_anneal(seed, "simulated annealing", 1.0, 0.03)):
        print(r)
        rows.append(r)

with open("results/algoritmet.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["algoritem", "seed", "anije", "mse", "e_vlefshme"])
    w.writeheader()
    w.writerows(rows)
print("Gati! Shiko results/algoritmet.csv")