import sys
import glob
import numpy as np
import solve_lta as s
from lta_udp import udp

files = sys.argv[1:] or sorted(glob.glob("results/x_seed*_3[34]anije.npy"))
N, MAXK = s.N, s.MAXK
LIMIT = 0.0499

for path in files:
    x = np.load(path)
    counts = x[:N].astype(int)
    phs = x[N:]
    crafts = []
    for g in range(N):
        fam = 0 if g < s.FAM_OFF[1] else (1 if g < s.FAM_OFF[2] else 2)
        i = g - s.FAM_OFF[fam]
        for k in range(counts[g]):
            crafts.append((fam, i, phs[g * MAXK + k]))
    sigs = [s.craft_signal(f, i, ph) for f, i, ph in crafts]
    total = np.sum(sigs, axis=0)
    start_n = len(crafts)

    changed = True
    while changed:
        changed = False
        for k in range(len(crafts)):
            new_total = total - sigs[k]
            if s.mse_of(new_total) <= LIMIT:
                total = new_total
                del crafts[k]
                del sigs[k]
                changed = True
                break

    x2 = s.build_x(crafts)
    obj, num_sel, mse = udp.fitness(x2, postprocess=True)
    print(f"{path}: {start_n} -> {int(num_sel)} anije, MSE={float(mse):.4f}")
    if int(num_sel) < start_n and float(mse) < 0.05:
        np.save(f"results/x_pruned_{int(num_sel)}anije.npy", x2)
        print("  U ruajt zgjidhja e re!")