import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lta_udp import udp

x = np.load("results/x_seed131_32anije.npy")
obj, num_sel, mse = udp.fitness(x, postprocess=True)
print(f"Anije: {int(num_sel)}, MSE: {float(mse):.4f}")

udp.plot_signal(x)
plt.gcf().savefig("results/best_32anije.png", dpi=150)
print("U ruajt results/best_32anije.png")