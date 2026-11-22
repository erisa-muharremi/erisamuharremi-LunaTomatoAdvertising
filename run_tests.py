import os
import sys
import csv
import numpy as np
import solve_lta as s
from lta_udp import udp

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
os.makedirs("results", exist_ok=True)

# (seed, minuta)
configs = [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (1, 1), (1, 5)]
fields = ["seed", "minuta", "greedy_anije", "greedy_mse",
          "final_anije", "final_mse", "e_vlefshme"]

with open("results/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for seed, minutes in configs:
        s.rng = np.random.default_rng(seed)
        print(f"=== seed={seed}, minuta={minutes * scale} ===")
        crafts, counts, total = s.greedy()
        row = {"seed": seed, "minuta": minutes * scale,
               "greedy_anije": len(crafts),
               "greedy_mse": round(s.mse_of(total), 4),
               "final_anije": "", "final_mse": "", "e_vlefshme": "jo"}
        n, best = s.anneal(crafts, counts, total, minutes * scale * 60)
        if best is not None:
            x = s.build_x(best)
            obj, num_sel, mse = udp.fitness(x, postprocess=True)
            row.update(final_anije=int(num_sel), final_mse=round(float(mse), 4),
                       e_vlefshme="po")
        w.writerow(row)
        f.flush()
        print(row)

print("Gati! Shiko results/results.csv")