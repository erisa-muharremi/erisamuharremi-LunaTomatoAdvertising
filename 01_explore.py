import matplotlib.pyplot as plt
from lta_udp import udp

print("Mesazhi:", udp.message)
print("Morse:", udp.message_to_morse_string(udp.message))
print("Orbita DRO:", udp.num_dro)
print("Orbita Lyapunov:", udp.num_lyap)
print("Orbita Axial:", udp.num_axial)
print("Maksimumi i anijeve per orbite:", udp.max_per_orbit)
print("Pika kohore te sinjalit:", len(udp.t))
print("Gjatesia e nje pulsi (dite):", round(udp.width, 4))
print("Pika 'ndezur' te sinjalit synues:", int(udp.target_signal.sum()))

udp.plot_target()
plt.show()