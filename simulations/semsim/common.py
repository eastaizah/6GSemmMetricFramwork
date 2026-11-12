"""Shared building blocks of the round-1 revision campaign.

Data, networks, training and the link-level channel models.  The architecture
and training recipe are those of the authors' original script
(``simulations/original/simulate_semantic_metrics.py``):

* encoder d -> 128 -> 64 -> k, decoder k -> 64 -> 128 -> d (ReLU MLPs,
  orthogonal init), oracle/task classifier d -> 128 -> 64 -> 10;
* joint loss  L = (1 - lambda_t) MSE(x, x_hat) + lambda_t CE(oracle(x_hat), y),
  lambda_t = 0.02, oracle frozen; AWGN training noise at 18 dB;
* synthetic proxy: 10 class-conditional Gaussians in d = 512,
  mu_c = 2.0 * v_c (v_c random unit vectors), Sigma = I_d.

Differences w.r.t. the original script (all deliberate, see RESULTS.md):

* a held-out test set is used (the original trained and evaluated on the same
  1,000 samples, so the oracle "100 % accuracy" was in-sample memorisation);
* the same trained models feed every table (no re-seeded retraining);
* every random draw (data, initialisation, minibatches, channel noise) is
  derived from the run seed, and each evaluation point has its own generator;
* the k real latents are carried by k/2 complex channel uses (I/Q pairing) so
  that the analog scheme and the digital baselines share one channel model and
  one channel-use count; the receiver has perfect CSI (coherent detection).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_INPUT = 512
NUM_CLASSES = 10
SEPARATION = 2.0
TASK_WEIGHT = 0.02
TRAIN_SNR_DB = 18.0
K_VALUES = [8, 16, 32, 64, 128]
SNR_GRID = [float(s) for s in np.arange(-5.0, 25.01, 2.5)]
CHANNELS = ["AWGN", "Rayleigh", "Rician_K5", "Rician_K10", "TDL-A"]
CHANNEL_LABEL = {"AWGN": "AWGN", "Rayleigh": "Rayleigh", "Rician_K5": "Rician K=5 dB",
                 "Rician_K10": "Rician K=10 dB", "TDL-A": "TDL-A (simplified, 3 taps)"}
TDLA_TAP_POWERS = [0.60, 0.24, 0.16]
EQUALIZER = "mmse"  # per-symbol LMMSE with perfect CSI (better than ZF under fading, identical on AWGN up to 0.01)


def sub_seed(*parts) -> int:
    """Deterministic 31-bit seed from any tuple of identifiers."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def gen(*parts) -> torch.Generator:
    return torch.Generator().manual_seed(sub_seed(*parts))


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    info: dict = field(default_factory=dict)

    @property
    def d(self):
        return self.x_train.shape[1]


def gaussian_dataset(seed: int, n_train: int, n_test: int) -> Dataset:
    g = gen("data", seed)
    cent = F.normalize(torch.randn(NUM_CLASSES, D_INPUT, generator=g), dim=1) * SEPARATION
    ytr = torch.randint(0, NUM_CLASSES, (n_train,), generator=g)
    yte = torch.randint(0, NUM_CLASSES, (n_test,), generator=g)
    xtr = cent[ytr] + torch.randn(n_train, D_INPUT, generator=g)
    xte = cent[yte] + torch.randn(n_test, D_INPUT, generator=g)
    # Bayes-optimal (nearest true centroid) accuracy on the test set: the ceiling of any TSR.
    bayes = (torch.cdist(xte, cent).argmin(1) == yte).float().mean().item()
    return Dataset("gauss", xtr, ytr, xte, yte,
                   {"bayes_acc_test": bayes, "separation": SEPARATION, "n_train": n_train,
                    "n_test": n_test})


def clip_dataset(path: str, n_test: int | None = None) -> Dataset:
    z = np.load(path, allow_pickle=False)
    xtr = torch.tensor(z["x_train"], dtype=torch.float32)
    xte = torch.tensor(z["x_test"], dtype=torch.float32)
    ytr = torch.tensor(z["y_train"], dtype=torch.long)
    yte = torch.tensor(z["y_test"], dtype=torch.long)
    # per-dimension z-scoring fitted on the training split only (so the same network
    # sizes, learning rate and SNR definition apply as for the Gaussian proxy)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    if n_test is not None:
        xte, yte = xte[:n_test], yte[:n_test]
    raw_norm = float(np.linalg.norm(z["x_test"], axis=1).mean())
    return Dataset("clip", xtr, ytr, xte, yte,
                   {"n_train": int(xtr.shape[0]), "n_test": int(xte.shape[0]),
                    "model": str(z["model"]), "raw_mean_norm": raw_norm})


# ----------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------
def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _orth_init(mods, g: torch.Generator):
    for m in mods:
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                w = torch.empty_like(mod.weight)
                # orthogonal init with an explicit generator (reproducible)
                a = torch.randn(max(w.shape), min(w.shape), generator=g)
                q, r = torch.linalg.qr(a)
                q = q * torch.sign(torch.diagonal(r))
                mod.weight.data.copy_(q if w.shape[0] >= w.shape[1] else q.T)
                nn.init.zeros_(mod.bias)


class AE(nn.Module):
    """Semantic encoder/decoder pair with a fixed average-power reference P."""

    def __init__(self, d: int, k: int):
        super().__init__()
        self.k = k
        self.enc = _mlp([d, 128, 64, k])
        self.dec = _mlp([k, 64, 128, d])
        self.register_buffer("P", torch.tensor(1.0))  # mean z^2 per real dim (train set)

    def forward_clean(self, x):
        return self.dec(self.enc(x))


def oracle_net(d: int) -> nn.Module:
    return _mlp([d, 128, 64, NUM_CLASSES])


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def flops_forward(m: nn.Module) -> int:
    """2 x multiply-accumulates of all Linear layers (bias adds ignored)."""
    return sum(2 * mod.in_features * mod.out_features for mod in m.modules()
               if isinstance(mod, nn.Linear))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_oracle(ds: Dataset, seed: int, iters: int = 3000, batch: int = 256,
                 wd: float = 1e-4, val_frac: float = 0.1) -> tuple[nn.Module, dict]:
    torch.manual_seed(sub_seed("oracle-torch", seed, ds.name))
    g = gen("oracle", seed, ds.name)
    n = ds.x_train.shape[0]
    perm = torch.randperm(n, generator=g)
    nv = int(val_frac * n)
    iv, it = perm[:nv], perm[nv:]
    clf = oracle_net(ds.d)
    _orth_init([clf], g)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=wd)
    best, best_state = -1.0, None
    for i in range(iters):
        idx = it[torch.randint(0, it.numel(), (batch,), generator=g)]
        loss = F.cross_entropy(clf(ds.x_train[idx]), ds.y_train[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 100 == 0:
            with torch.no_grad():
                acc = (clf(ds.x_train[iv]).argmax(1) == ds.y_train[iv]).float().mean().item()
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(best_state)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        tr = (clf(ds.x_train[it]).argmax(1) == ds.y_train[it]).float().mean().item()
        te = (clf(ds.x_test).argmax(1) == ds.y_test).float().mean().item()
    return clf, {"oracle_val_acc": best, "oracle_train_acc": tr, "oracle_test_acc": te}


def _awgn_train_noise(z, snr_db, g):
    sp = z.detach().pow(2).mean() + 1e-8
    std = (sp / 10 ** (snr_db / 10)).sqrt()
    return z + std * torch.randn(z.shape, generator=g)


def train_ae(ds: Dataset, k: int, oracle: nn.Module, seed: int, lam: float = TASK_WEIGHT,
             iters: int = 3000, batch: int = 256, adv: dict | None = None,
             tag: str = "prop") -> AE:
    """Joint-loss training (lam = 0 gives the DeepJSCC-type MSE-only baseline B1).

    ``adv`` (optional): {"eps", "steps", "sigma"} -> PGD adversarial training with
    Gaussian input-noise augmentation (SmoothAdv-style, Salman et al. 2019): each
    minibatch is x_n = x + sigma*xi, delta = PGD_linf(x_n, eps), loss on
    x_n + delta with the clean x as reconstruction target; half of every minibatch
    stays clean.
    """
    torch.manual_seed(sub_seed("ae-torch", seed, ds.name, k, tag))
    g = gen("ae", seed, ds.name, k, tag)
    ae = AE(ds.d, k)
    _orth_init([ae.enc, ae.dec], g)
    opt = torch.optim.Adam(list(ae.enc.parameters()) + list(ae.dec.parameters()), lr=1e-3)
    n = ds.x_train.shape[0]
    for i in range(iters):
        idx = torch.randint(0, n, (batch,), generator=g)
        x, y = ds.x_train[idx], ds.y_train[idx]
        xin = x
        if adv is not None:
            h = batch // 2
            xn = x[h:] + adv["sigma"] * torch.randn(x[h:].shape, generator=g)
            from .adversarial import pgd  # local import (no cycle at module load)
            ae.eval()
            d = pgd(ae, oracle, xn, y[h:], adv["eps"], steps=adv["steps"], gen_=g, rand_init=True)
            ae.train()
            xin = torch.cat([x[:h], xn + d], 0)
        z = ae.enc(xin)
        xh = ae.dec(_awgn_train_noise(z, TRAIN_SNR_DB, g))
        loss = (1 - lam) * F.mse_loss(xh, x)
        if lam > 0:
            loss = loss + lam * F.cross_entropy(oracle(xh), y)
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        ae.P.fill_(ae.enc(ds.x_train).pow(2).mean().item())
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


# ----------------------------------------------------------------------------
# Channels (complex baseband, one complex channel use = two real latent dims)
# ----------------------------------------------------------------------------
def _cn(shape, g):
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g)) / math.sqrt(2)


TDLA_EGC_NORM = math.sqrt(1 + (math.pi / 4) * sum(
    math.sqrt(a * b) for i, a in enumerate(TDLA_TAP_POWERS) for j, b in enumerate(TDLA_TAP_POWERS) if i != j))


def channel_gains(ch: str, shape, g: torch.Generator) -> torch.Tensor:
    """Per-complex-symbol channel coefficient h with E|h|^2 = 1.

    AWGN: h = 1.  Rayleigh: h ~ CN(0,1).  Rician K: h = sqrt(K/(K+1)) + sqrt(1/(K+1)) CN(0,1).
    TDL-A (simplified, as in the original script): three resolvable taps with powers
    [0.60, 0.24, 0.16], each Rayleigh, combined coherently with equal gain; the
    effective real gain sum_i sqrt(p_i)|h_i| is normalised to unit mean power (the
    original script did not normalise, which silently added ~2.6 dB of array gain).
    It is a flat-fading diversity abstraction of the TR 38.901 TDL-A power-delay
    profile, not the full 23-tap frequency-selective model.
    All fading is i.i.d. across channel uses (fast fading / ideal interleaving).
    """
    if ch == "AWGN":
        return torch.ones(shape, dtype=torch.complex64)
    if ch == "Rayleigh":
        return _cn(shape, g)
    if ch.startswith("Rician_K"):
        K = 10 ** (float(ch[len("Rician_K"):]) / 10)
        return (math.sqrt(K / (K + 1)) + math.sqrt(1 / (K + 1)) * _cn(shape, g)).to(torch.complex64)
    if ch == "TDL-A":
        gsum = sum(math.sqrt(p) * _cn(shape, g).abs() for p in TDLA_TAP_POWERS)
        return (gsum / TDLA_EGC_NORM).to(torch.complex64)
    raise ValueError(ch)


def analog_link(z: torch.Tensor, P: float, snr_db: float, h: torch.Tensor, g: torch.Generator,
                equalizer: str = EQUALIZER) -> torch.Tensor:
    """Carry k real latents over k/2 complex channel uses; returns the equalised latents.

    Transmit power normalisation uses the fixed training-set average power P (per
    real dim), so E|s|^2 = 1 per complex use; SNR = E|s|^2 / N0 (per channel use).
    """
    k = z.shape[1]
    h2 = k // 2
    s = torch.complex(z[:, :h2], z[:, h2:]) / math.sqrt(2 * P)
    snr = 10 ** (snr_db / 10)
    n = _cn(s.shape, g) / math.sqrt(snr)
    y = h * s + n
    if equalizer == "zf":
        s_hat = y / h
    else:  # LMMSE
        s_hat = torch.conj(h) * y / (h.abs() ** 2 + 1 / snr)
    s_hat = s_hat * math.sqrt(2 * P)
    return torch.cat([s_hat.real, s_hat.imag], 1).float()


@torch.no_grad()
def run_analog(ae: AE, x: torch.Tensor, ch: str, snr_db: float, g: torch.Generator,
               h: torch.Tensor | None = None) -> torch.Tensor:
    z = ae.enc(x)
    if h is None:
        h = channel_gains(ch, (x.shape[0], ae.k // 2), g)
    zh = analog_link(z, float(ae.P), snr_db, h, g)
    return ae.dec(zh)
