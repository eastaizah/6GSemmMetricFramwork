"""Low-altitude (UAV air-to-ground) channel with a two-state LoS/blocked Markov process.

Model (every parameter is documented in RESULTS.md, Section "Low-altitude channel"):

* LoS probability vs elevation angle theta (degrees), Al-Hourani, Kandeepan and
  Lardner, IEEE Wireless Commun. Lett. 3(6):569-572, 2014, urban environment:
      P_LoS(theta) = 1 / (1 + a exp(-b (theta - a))),  a = 9.61, b = 0.16.
* Mean excess path loss over free space (same reference, urban): eta_LoS = 1 dB,
  eta_NLoS = 20 dB.  The nominal SNR is the free-space SNR; the LoS state therefore
  receives SNR_nom - 1 dB and the blocked state SNR_nom - 20 dB - shadowing.
* LoS small-scale fading: Rician with elevation-dependent K factor
      K(theta) = K0 exp((2/pi) ln(K90/K0) theta_rad)   (exponential form used by
  Azari et al., IEEE Trans. Commun. 66(1):330-344, 2018), with the end-point values
  K0 = 0 dB at theta = 0 and K90 = 15 dB at theta = 90 deg (assumed values of this study).
* Blocked state: Rayleigh fading plus log-normal shadowing, sigma_SH = 6 dB (the NLoS
  shadow-fading std. of the 3GPP TR 36.777 aerial-UE models), drawn once per
  blockage episode (constant while the episode lasts).
* Within a session of T messages the state follows a two-state Markov chain with
  p(LoS->blocked) = lam (1 - P_LoS), p(blocked->LoS) = lam P_LoS, whose stationary
  LoS probability is P_LoS(theta) for every lam; lam in (0, 1] is the blockage
  transition rate (lam = 1: i.i.d. states, lam -> 0: one state per session).
  The initial state is drawn from the stationary distribution.
* Fast fading is i.i.d. across complex channel uses (as for the terrestrial channels).
"""
from __future__ import annotations

import math

import torch

from .common import _cn

A_URBAN, B_URBAN = 9.61, 0.16
ETA_LOS_DB, ETA_NLOS_DB = 1.0, 20.0
K0_DB, K90_DB = 0.0, 15.0
SIGMA_SH_DB = 6.0


def p_los(theta_deg: float) -> float:
    return 1.0 / (1.0 + A_URBAN * math.exp(-B_URBAN * (theta_deg - A_URBAN)))


def k_factor(theta_deg: float) -> float:
    k0, k90 = 10 ** (K0_DB / 10), 10 ** (K90_DB / 10)
    return k0 * math.exp((2 / math.pi) * math.log(k90 / k0) * math.radians(theta_deg))


def markov_states(n_sess: int, T: int, pl: float, lam: float, g: torch.Generator) -> torch.Tensor:
    """Boolean (n_sess, T): True = LoS."""
    st = torch.empty(n_sess, T, dtype=torch.bool)
    st[:, 0] = torch.rand(n_sess, generator=g) < pl
    p_lb, p_bl = lam * (1 - pl), lam * pl
    for t in range(1, T):
        u = torch.rand(n_sess, generator=g)
        prev = st[:, t - 1]
        st[:, t] = torch.where(prev, u >= p_lb, u < p_bl)
    return st


def uav_gains(n_sess: int, T: int, n_use: int, theta: float, lam: float, g: torch.Generator):
    """Channel coefficients (n_sess*T, n_use) including excess loss/shadowing, and states."""
    pl = p_los(theta)
    st = markov_states(n_sess, T, pl, lam, g)
    K = k_factor(theta)
    los = (math.sqrt(K / (K + 1)) + math.sqrt(1 / (K + 1)) * _cn((n_sess, T, n_use), g))
    los = los * 10 ** (-ETA_LOS_DB / 20)
    # shadowing per blockage episode: new draw whenever a blocked episode starts
    start = (~st) & torch.cat([torch.ones(n_sess, 1, dtype=torch.bool), st[:, :-1]], 1)
    ep_id = torch.cumsum(start.long(), 1)
    sh_draw = torch.randn(n_sess, T + 1, generator=g) * SIGMA_SH_DB
    sh_db = torch.gather(sh_draw, 1, ep_id)
    nlos_amp = 10 ** (-(ETA_NLOS_DB + sh_db) / 20)
    nlos = _cn((n_sess, T, n_use), g) * nlos_amp.unsqueeze(-1)
    h = torch.where(st.unsqueeze(-1), los, nlos)
    return h.reshape(n_sess * T, n_use).to(torch.complex64), st.reshape(-1)
