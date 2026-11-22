import sys
import numpy as np
import solve_lta as s
from lta_udp import udp

minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 60
seeds = [int(a) for a in sys.argv[2:]] or [1]
best_n = 10**9

for seed in seeds:
    s.rng = np.random.default_rng(seed)
    crafts, counts, total = s.greedy()
    n, best = s.anneal(crafts, counts, total, minutes * 60)
    if best is None:
        print(f"seed {seed}: nuk u gjet zgjidhje e vlefshme")
        continue
    x = s.build_x(best)
    obj, num_sel, mse = udp.fitness(x, postprocess=True)
    num_sel = int(num_sel)
    print(f"SEED {seed}: {num_sel} anije, MSE={float(mse):.4f}")
    np.save(f"results/x_seed{seed}_{num_sel}anije.npy", x)
    if num_sel < best_n:
        best_n = num_sel
        np.save("results/best_overall.npy", x)

print("Gati! Me i miri:", best_n, "anije")