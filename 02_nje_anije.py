import numpy as np
import matplotlib.pyplot as plt
from lta_udp import udp

# Nje anije ne orbiten DRO numer 0, me faze 1.0 radian
i = 0
ph = 1.0
f = udp.frequencies_dro[i]
center = (2 * np.pi - ph) / (2 * np.pi) / f

# Sinjali i kesaj anijeje (pulse te perseritura)
signal = udp._repeating_box_window(udp.t, center, f)

# Gabimi MSE ndaj sinjalit synues
mse = np.mean((np.clip(signal, 0, 1) - udp.target_signal) ** 2)
print("Pika te ndezura nga kjo anije:", int(signal.sum()))
print("MSE me vetem kete anije:", round(mse, 4))

plt.plot(udp.t, udp.target_signal, label="synuar")
plt.plot(udp.t, signal, label="anija")
plt.xlabel("Dite")
plt.legend()
plt.show()