import sys
import time
import numpy as np
from lta_udp import udp

rng = np.random.default_rng(0)
TWO_PI = 2 * np.pi

N = udp.num_orbits
MAXK = udp.max_per_orbit
FAM_SIZE = [udp.num_dro, udp.num_lyap, udp.num_axial]
FAM_OFF = [0, udp.num_dro, udp.num_dro + udp.num_lyap]
FREQ = [udp.frequencies_dro, udp.frequencies_lyap, udp.frequencies_axial]
TARGET = udp.target_signal
T = udp.t

SEC = 86400.0
WIDTH_ND = udp.width * SEC / udp.TU      # gjeresia e nje pulsi
DT_ND = (T[1] - T[0]) * SEC / udp.TU     # hapi i kohes
T_ND = T * SEC / udp.TU

M_SAFE = 0.047     # synimi yne (pak nen pragun 0.05)
FEAS = 0.049       # kur e ruajme si zgjidhje te vlefshme
W = 500.0          # sa "kushton" gabimi ne krahasim me nje anije
FAM_P = np.array(FAM_SIZE) / sum(FAM_SIZE)


# ---------- veglat bazë ----------
def window(center, f):
    return udp._repeating_box_window(T, center, f)


def craft_signal(fam, i, ph):
    """Sinjali qe jep nje anije ne orbiten i te familjes fam me fazen ph."""
    if fam == 0:
        f = FREQ[0][i]
        return window((TWO_PI - ph) / TWO_PI / f, f)
    n = FAM_SIZE[fam]
    f1, f2 = FREQ[fam][i], FREQ[fam][n + i]
    return window((TWO_PI - ph) / TWO_PI / f1, f1) + window((ph + np.pi) / f2, f2)


def anchored_phase(fam, i, sample):
    """Zgjedh fazen ashtu qe pulsi i pare te bjeri aty ku ndodhet 'sample'."""
    f = FREQ[fam][i]
    period = 1.0 / f
    center = T_ND[sample] + rng.uniform(-0.4, 0.4) * WIDTH_ND
    return (TWO_PI * (1.0 - (center % period) / period)) % TWO_PI


def mse_of(total):
    return float(np.mean((np.clip(total, 0, 1) - TARGET) ** 2))


def energy(n, mse):
    return n + W * max(0.0, mse - M_SAFE)


def random_orbit(counts):
    while True:
        fam = int(rng.choice(3, p=FAM_P))
        i = int(rng.integers(FAM_SIZE[fam]))
        if counts[FAM_OFF[fam] + i] < MAXK:
            return fam, i


def new_craft(counts, total):
    """Anije e re, e ankoruar te nje pike e padukur e sinjalit."""
    fam, i = random_orbit(counts)
    miss = np.flatnonzero((TARGET > 0) & (total < 1))
    pool = miss if len(miss) > 0 else np.flatnonzero(TARGET > 0)
    ph = anchored_phase(fam, i, int(rng.choice(pool)))
    return [fam, i, ph, craft_signal(fam, i, ph)]


# ---------- prova e pjesës 1 ----------
if __name__ == "__main__":
    counts = np.zeros(N, dtype=int)
    total = np.zeros_like(TARGET)
    c = new_craft(counts, total)
    print("MSE pa asnje anije:", round(mse_of(total), 4))
    print("MSE me nje anije:", round(mse_of(total + c[3]), 4))
