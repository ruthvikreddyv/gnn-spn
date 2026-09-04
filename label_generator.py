"""
label_generator.py  -  BER-Constrained Label Generation (NaN-safe)
"""
import numpy as np
from scipy.special import erfc

POWER_DBM   = np.array([23, 10, 5], dtype=np.float64)
POWER_LIN   = 10 ** (POWER_DBM / 10)
N_RB        = 20
N_POWER     = 3
N_ACTIONS   = N_RB * N_POWER    # 60
NOISE_DBM   = -114
NOISE_LIN   = float(10 ** (NOISE_DBM / 10))
MOD_ORDERS  = np.array([2, 4, 16], dtype=np.float64)
CODING_RATE = 0.5
SPECTRAL_EFF = CODING_RATE * np.log2(MOD_ORDERS)
BER_TARGET  = 1e-3

_RB_IDX = np.tile(np.arange(N_RB), N_POWER)
_PW_IDX = np.repeat(np.arange(N_POWER), N_RB)
_PW_LIN = POWER_LIN[_PW_IDX]


def _Q(x):
    return 0.5 * erfc(np.asarray(x, dtype=np.float64) / np.sqrt(2))


def _ber_all_mods(sinr):
    """sinr: (n_links, N_ACTIONS) -> ber: (n_links, N_ACTIONS, 3)"""
    s = sinr[:, :, np.newaxis]
    M = MOD_ORDERS[np.newaxis, np.newaxis, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        coeff = np.where(M == 2, 1.0, 4*(1 - M**(-0.5)) / np.log2(M))
        arg   = np.where(M == 2,
                         np.sqrt(np.maximum(2*CODING_RATE*s, 0)),
                         np.sqrt(np.maximum(3*CODING_RATE*s/(M-1), 0)))
        ber = coeff * _Q(arg)
    return np.nan_to_num(ber, nan=1.0, posinf=1.0)


def generate_labels(v2v_ch_ff, v2v_interf_all, vehicles,
                    action_all, n_veh, n_rb=N_RB, label_temp=0.0):
    """
    Returns labels: (n_veh*3, N_ACTIONS) float32
    Hard one-hot when label_temp=0, soft otherwise.
    """
    n_links = n_veh * 3

    # Desired channel gain (n_links, n_rb)
    desired_gain = np.zeros((n_links, n_rb), dtype=np.float64)
    for i in range(n_veh):
        for j in range(3):
            dest = vehicles[i].destinations[j]
            desired_gain[3*i+j] = 10 ** (v2v_ch_ff[i, dest, :] / 10)

    # Signal (n_links, N_ACTIONS)
    signal = _PW_LIN[np.newaxis, :] * desired_gain[:, _RB_IDX]

    # V2V co-tier interference (n_links, n_rb)
    interf_v2v = np.zeros((n_links, n_rb), dtype=np.float64)
    for vk in range(n_veh):
        for lm in range(3):
            rb_k = int(action_all[vk, lm, 0])
            if rb_k < 0 or rb_k >= n_rb:
                continue
            pw_k = float(POWER_LIN[int(action_all[vk, lm, 1])])
            for i in range(n_veh):
                for j in range(3):
                    if vk == i and lm == j:
                        continue
                    dest = vehicles[i].destinations[j]
                    ch   = 10 ** (v2v_ch_ff[vk, dest, rb_k] / 10)
                    interf_v2v[3*i+j, rb_k] += pw_k * ch

    # V2I interference (n_links, n_rb)
    interf_v2i = np.zeros((n_links, n_rb), dtype=np.float64)
    for i in range(n_veh):
        for j in range(3):
            interf_v2i[3*i+j] = 10 ** (v2v_interf_all[i, j, :] / 10)

    total_interf = NOISE_LIN + interf_v2v + interf_v2i
    sinr = signal / np.maximum(total_interf[:, _RB_IDX], 1e-12)

    ber_all  = _ber_all_mods(sinr)             # (L, A, 3)
    feasible = ber_all <= BER_TARGET            # (L, A, 3)
    any_feas = np.any(feasible, axis=2)         # (L, A)

    eff_mat  = np.where(feasible,
                        SPECTRAL_EFF[np.newaxis, np.newaxis, :],
                        -np.inf)
    best_eff = np.max(eff_mat, axis=2)          # (L, A)

    labels = np.zeros((n_links, N_ACTIONS), dtype=np.float32)
    for li in range(n_links):
        if np.any(any_feas[li]):
            eff_i = np.where(any_feas[li], best_eff[li], -np.inf)
        else:
            eff_i = sinr[li]                    # fallback: best SINR

        if label_temp > 0.0 and np.any(np.isfinite(eff_i)):
            # Soft labels: temperature-scaled softmax over feasible set
            eff_safe = np.where(np.isfinite(eff_i), eff_i, -1e9)
            eff_safe = eff_safe - eff_safe.max()
            weights  = np.exp(eff_safe / label_temp)
            weights  = np.where(np.isfinite(eff_i), weights, 0.0)
            total    = weights.sum()
            if total > 0:
                labels[li] = (weights / total).astype(np.float32)
            else:
                labels[li, int(np.argmax(sinr[li]))] = 1.0
        else:
            best_a = int(np.argmax(eff_i))
            labels[li, best_a] = 1.0

    return labels