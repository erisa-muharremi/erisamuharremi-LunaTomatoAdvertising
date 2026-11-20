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


# ---------- pjesa 2: greedy ----------
def greedy(max_crafts=600, cands=300):
    """Shton anije nje nga nje; ne cdo hap mban ate qe ul me shume gabimin."""
    crafts, counts = [], np.zeros(N, dtype=int)
    total = np.zeros_like(TARGET)
    mse = mse_of(total)
    while mse > M_SAFE and len(crafts) < max_crafts:
        best, best_mse = None, mse
        for _ in range(cands):
            c = new_craft(counts, total)
            m = mse_of(total + c[3])
            if m < best_mse:
                best, best_mse = c, m
        if best is None:
            break
        crafts.append(best)
        counts[FAM_OFF[best[0]] + best[1]] += 1
        total = total + best[3]
        mse = best_mse
        print(f"  greedy: {len(crafts)} anije, MSE={mse:.4f}")
    return crafts, counts, total


# ---------- pjesa 3: simulated annealing ----------
def anneal(crafts, counts, total, seconds, temp0=1.0, temp1=0.03):
    """Provon ndryshime te vogla dhe mban ato qe permiresojne zgjidhjen."""
    mse = mse_of(total)
    cur = energy(len(crafts), mse)
    best_n, best = (len(crafts), [c[:3] for c in crafts]) if mse <= FEAS else (10**9, None)
    t_start, it = time.time(), 0

    while True:
        el = time.time() - t_start
        if el > seconds:
            break
        temp = temp0 * (temp1 / temp0) ** (el / seconds)
        it += 1
        r = rng.random()
        removed, added = None, None   # removed = indeksi, added = anije e re

        if r < 0.45 and crafts:            # leviz pak fazen
            k = int(rng.integers(len(crafts)))
            fam, i, ph, _ = crafts[k]
            period = 1.0 / FREQ[fam][i]
            ph2 = (ph + TWO_PI * rng.normal(0, 3) * DT_ND / period) % TWO_PI
            removed, added = k, [fam, i, ph2, craft_signal(fam, i, ph2)]
        elif r < 0.75 and crafts:          # zhvendose ne nje orbite tjeter
            k = int(rng.integers(len(crafts)))
            c = crafts[k]
            counts[FAM_OFF[c[0]] + c[1]] -= 1
            added = new_craft(counts, total - c[3])
            counts[FAM_OFF[c[0]] + c[1]] += 1
            removed = k
        elif r < 0.90 and crafts:          # hiq nje anije
            removed = int(rng.integers(len(crafts)))
        else:                              # shto nje anije
            added = new_craft(counts, total)

        new_total = total.copy()
        if removed is not None:
            new_total -= crafts[removed][3]
        if added is not None:
            new_total += added[3]
        n_new = len(crafts) - (removed is not None) + (added is not None)
        new_mse = mse_of(new_total)
        new_e = energy(n_new, new_mse)

        if new_e <= cur or rng.random() < np.exp(-(new_e - cur) / temp):
            if removed is not None:
                c = crafts[removed]
                counts[FAM_OFF[c[0]] + c[1]] -= 1
                crafts[removed] = crafts[-1]
                crafts.pop()
            if added is not None:
                if counts[FAM_OFF[added[0]] + added[1]] >= MAXK:
                    continue
                crafts.append(added)
                counts[FAM_OFF[added[0]] + added[1]] += 1
            total, mse, cur = new_total, new_mse, new_e
            if mse <= FEAS and len(crafts) < best_n:
                best_n, best = len(crafts), [c[:3] for c in crafts]
                print(f"  [{el:6.0f}s] rekord i ri: {best_n} anije, MSE={mse:.4f}")
        if it % 20000 == 0:
            print(f"  [{el:6.0f}s] it={it}  anije={len(crafts)}  MSE={mse:.4f}  T={temp:.3f}")
    return best_n, best


# ---------- ndertimi i vektorit x per UDP ----------
def build_x(best):
    """Kthen listen e anijeve ne formatin qe pret udp.fitness."""
    phs = np.zeros(N * MAXK)
    slot = np.zeros(N, dtype=int)
    for fam, i, ph in best:
        g = FAM_OFF[fam] + i
        phs[g * MAXK + slot[g]] = ph
        slot[g] += 1
    return np.concatenate([slot.astype(float), phs])


if __name__ == "__main__":
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    print("Hapi 1: greedy")
    crafts, counts, total = greedy()
    print(f"Hapi 2: annealing per {minutes} min")
    n, best = anneal(crafts, counts, total, minutes * 60)
    if best is None:
        print("Nuk u gjet zgjidhje e vlefshme. Provo me me shume kohe.")
        sys.exit(1)
    x = build_x(best)
    np.save("best_x.npy", x)
    obj, num_sel, mse = udp.fitness(x, postprocess=True)
    print(f"Kontrolli final nga UDP: anije={num_sel}, MSE={mse:.4f}, obj={obj[0]:.1f}")
    udp.plot_signal(x)
    