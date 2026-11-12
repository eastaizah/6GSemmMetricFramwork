"""Digital baselines with 5G NR LDPC + QAM (Sionna 2.2, PyTorch back-end).

* B2  separate source-channel coding: PCA of the 512-dim embedding (fitted on
      the training split) + per-coefficient uniform scalar quantisation
      (b bits, Max (1960) optimum step for a Gaussian source, scaled by the
      coefficient std) -> 5G NR LDPC (rate-matched, BG chosen by Sionna) ->
      Gray QPSK / 16-QAM -> channel -> coherent APP demapping with perfect CSI ->
      LDPC BP decoding (20 iterations, boxplus-phi) -> dequantise the decoded
      information bits as they are (no CRC: a failed codeword is reconstructed
      from whatever bits the decoder outputs) -> PCA synthesis -> task model.
* B3  bit-exact reference: every one of the 512 embedding dims quantised with
      b = 8 bits (4,096 payload bits), same LDPC/QAM chain.
* DP  digital variant of the proposed encoder: the k latents of the trained
      semantic encoder quantised with b in {4, 8} bits, same chain; the trained
      semantic decoder is applied to the de-quantised latents.

Sionna 2.x LLR convention: ln P(b=1)/P(b=0); the Sionna demapper produces and the
Sionna decoder expects that convention, so no sign flip is needed (checked in
``self_test``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .common import channel_gains, _cn

_sn = None


def _sionna():
    global _sn
    if _sn is None:
        import sionna  # noqa: F401
        from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
        from sionna.phy.mapping import Demapper, Mapper
        _sn = dict(Enc=LDPC5GEncoder, Dec=LDPC5GDecoder, Map=Mapper, Demap=Demapper,
                   version=sionna.__version__)
    return _sn


# Max (1960) optimum uniform quantiser step (in units of sigma) for a Gaussian source
MAX_STEP = {1: 1.596, 2: 0.9957, 3: 0.5860, 4: 0.3352, 5: 0.1881, 6: 0.1041, 7: 0.0569, 8: 0.0308}


@dataclass
class UniformQuantizer:
    sigma: torch.Tensor   # per-dimension std
    mean: torch.Tensor
    b: int

    def encode(self, v: torch.Tensor) -> torch.Tensor:
        step = MAX_STEP[self.b] * self.sigma
        L = 2 ** self.b
        q = torch.floor((v - self.mean) / step).clamp(-L // 2, L // 2 - 1) + L // 2
        return q.long()

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        step = MAX_STEP[self.b] * self.sigma
        L = 2 ** self.b
        return (q.float() - L // 2 + 0.5) * step + self.mean


def ints_to_bits(q: torch.Tensor, b: int) -> torch.Tensor:
    sh = torch.arange(b - 1, -1, -1)
    return ((q.unsqueeze(-1) >> sh) & 1).reshape(q.shape[0], -1).float()


def bits_to_ints(bits: torch.Tensor, b: int) -> torch.Tensor:
    w = (2 ** torch.arange(b - 1, -1, -1)).float()
    return (bits.reshape(bits.shape[0], -1, b) * w).sum(-1).long()


class PCACoder:
    def __init__(self, x_train: torch.Tensor):
        self.mu = x_train.mean(0)
        xc = x_train - self.mu
        _, S, Vh = torch.linalg.svd(xc, full_matrices=False)
        self.V = Vh.T                                  # d x d
        self.std = S / math.sqrt(x_train.shape[0] - 1)  # coefficient stds
        self.x_train = x_train

    def quantizer(self, m: int, b: int) -> UniformQuantizer:
        return UniformQuantizer(self.std[:m], torch.zeros(m), b)

    def analysis(self, x, m):
        return (x - self.mu) @ self.V[:, :m]

    def synthesis(self, c, m):
        return c @ self.V[:, :m].T + self.mu


# ----------------------------------------------------------------------------- link
class LDPCLink:
    """K info bits -> LDPC(K, n = n_sym*bps) -> QAM(2^bps) over n_sym complex uses."""

    _cache: dict = {}

    def __init__(self, K: int, n_sym: int, bps: int, num_iter: int = 20):
        sn = _sionna()
        self.K, self.n_sym, self.bps = K, n_sym, bps
        self.n = n_sym * bps
        key = (K, self.n, num_iter)
        if key not in LDPCLink._cache:
            enc = sn["Enc"](K, self.n)
            dec = sn["Dec"](enc, num_iter=num_iter, hard_out=True, return_infobits=True)
            LDPCLink._cache[key] = (enc, dec)
        self.enc, self.dec = LDPCLink._cache[key]
        self.mapper = sn["Map"]("qam", bps)
        self.demapper = sn["Demap"]("app", "qam", bps)

    @property
    def rate(self):
        return self.K / self.n

    @torch.no_grad()
    def transmit(self, bits: torch.Tensor, ch: str, snr_db: float, g: torch.Generator,
                 chunk: int = 2000) -> torch.Tensor:
        out = []
        snr = 10 ** (snr_db / 10)
        for i in range(0, bits.shape[0], chunk):
            b = bits[i:i + chunk]
            c = self.enc(b)
            x = self.mapper(c)                                     # (N, n_sym) complex, E|x|^2 = 1
            h = channel_gains(ch, x.shape, g).to(x.dtype)
            y = h * x + (_cn(x.shape, g) / math.sqrt(snr)).to(x.dtype)
            y_eq = y / h
            no = (1.0 / snr) / (h.abs() ** 2).clamp_min(1e-12)
            llr = self.demapper(y_eq, no.float())
            out.append(self.dec(llr))
        return torch.cat(out).float()


def self_test(snr_db=6.0):
    """Sanity: BER at 6 dB with QPSK R=1/2 must be ~0; sign convention checked."""
    link = LDPCLink(256, 256, 2)
    g = torch.Generator().manual_seed(0)
    b = torch.randint(0, 2, (200, 256), generator=g).float()
    bh = link.transmit(b, "AWGN", snr_db, g)
    llr_ok = None
    c = link.enc(b[:10]); x = link.mapper(c)
    llr = link.demapper(x, torch.tensor(0.1))
    llr_ok = bool(((llr > 0).float() == c).float().mean() > 0.99)
    return {"ber_qpsk_r05_6dB": float((bh != b).float().mean()), "llr_sign_is_ln_p1_over_p0": llr_ok,
            "sionna_version": _sionna()["version"]}


# --------------------------------------------------------------------- schemes
MCS_LIST = [("QPSK", 2, 1 / 3), ("QPSK", 2, 1 / 2), ("QPSK", 2, 2 / 3),
            ("16QAM", 4, 1 / 2), ("16QAM", 4, 3 / 4)]


def k_info(n_sym, bps, rate):
    return max(12, int(round(n_sym * bps * rate)))  # 5G NR LDPC needs K >= 12 (Sionna)


@torch.no_grad()
def tsr_of(clf, xh, y):
    return float((clf(xh).argmax(1) == y).float().mean())


def source_curve(pca: PCACoder, clf, x_val, y_val, K_max: int, b_list=(1, 2, 3, 4, 5, 6)):
    """Noiseless TSR of the PCA+uniform-quantiser source coder for every payload K
    (best b per K, m = floor(K/b)).  Returns {K: (tsr, m, b)}."""
    cache = {}
    res = {}
    for K in range(1, K_max + 1):
        best = (tsr_of(clf, pca.synthesis(torch.zeros(x_val.shape[0], 0), 0), y_val), 0, 0)
        for b in b_list:
            m = K // b
            if m < 1 or m > pca.V.shape[1]:
                continue
            if (m, b) not in cache:
                q = pca.quantizer(m, b)
                c = pca.analysis(x_val, m)
                cache[(m, b)] = tsr_of(clf, pca.synthesis(q.decode(q.encode(c)), m), y_val)
            if cache[(m, b)] > best[0]:
                best = (cache[(m, b)], m, b)
        res[K] = best
    return res


class SeparateScheme:
    """B2: PCA(m) + b-bit quantiser + LDPC/QAM over n_sym complex uses."""

    def __init__(self, pca: PCACoder, n_sym: int, mcs, m: int, b: int):
        self.pca, self.m, self.b = pca, m, b
        name, bps, rate = mcs
        K = k_info(n_sym, bps, rate)
        self.payload = m * b
        assert self.payload <= K
        self.link = LDPCLink(K, n_sym, bps)
        self.q = pca.quantizer(m, b)
        self.desc = dict(scheme="B2", n_sym=n_sym, real_dims=2 * n_sym, mod=name, rate=K / (n_sym * bps),
                         K=K, m=m, b=b, payload_bits=self.payload)

    def run(self, x, ch, snr, g):
        q = self.q.encode(self.pca.analysis(x, self.m))
        bits = ints_to_bits(q, self.b)
        pad = self.link.K - bits.shape[1]
        if pad:
            bits = torch.cat([bits, torch.zeros(bits.shape[0], pad)], 1)
        bh = self.link.transmit(bits, ch, snr, g)[:, :self.payload]
        return self.pca.synthesis(self.q.decode(bits_to_ints(bh, self.b)), self.m)


class BitExactScheme:
    """B3: 8-bit quantisation of all d dims (per-dim Max step), LDPC/QAM."""

    def __init__(self, x_train, mcs, b=8):
        name, bps, rate = mcs
        d = x_train.shape[1]
        self.b = b
        self.q = UniformQuantizer(x_train.std(0), x_train.mean(0), b)
        self.payload = d * b
        n = int(round(self.payload / rate))
        n_sym = n // bps
        self.link = LDPCLink(self.payload, n_sym, bps)
        self.desc = dict(scheme="B3", n_sym=n_sym, real_dims=2 * n_sym, mod=name,
                         rate=self.payload / (n_sym * bps), K=self.payload, m=d, b=b,
                         payload_bits=self.payload)

    def run(self, x, ch, snr, g):
        bits = ints_to_bits(self.q.encode(x), self.b)
        bh = self.link.transmit(bits, ch, snr, g)
        return self.q.decode(bits_to_ints(bh, self.b))


class DigitalProposed:
    """DP: b-bit uniform quantisation of the k semantic latents + LDPC/QAM."""

    def __init__(self, ae, x_train, mcs, b):
        name, bps, rate = mcs
        with torch.no_grad():
            z = ae.enc(x_train)
        self.ae, self.b = ae, b
        self.q = UniformQuantizer(z.std(0), z.mean(0), b)
        self.payload = ae.k * b
        n = int(round(self.payload / rate))
        n_sym = n // bps
        self.link = LDPCLink(self.payload, n_sym, bps)
        self.desc = dict(scheme=f"DP-q{b}", n_sym=n_sym, real_dims=2 * n_sym, mod=name,
                         rate=self.payload / (n_sym * bps), K=self.payload, m=ae.k, b=b,
                         payload_bits=self.payload)

    @torch.no_grad()
    def run(self, x, ch, snr, g):
        bits = ints_to_bits(self.q.encode(self.ae.enc(x)), self.b)
        bh = self.link.transmit(bits, ch, snr, g)
        return self.ae.dec(self.q.decode(bits_to_ints(bh, self.b)))

    @torch.no_grad()
    def run_noiseless(self, x):
        return self.ae.dec(self.q.decode(self.q.encode(self.ae.enc(x))))


# ------------------------------------------------------------- analytical bound
def capacity_bits_per_use(ch: str, snr_db: float, n_mc: int = 200000) -> float:
    snr = 10 ** (snr_db / 10)
    if ch == "AWGN":
        return math.log2(1 + snr)
    g = torch.Generator().manual_seed(7)
    h = channel_gains(ch, (n_mc,), g)
    return float(torch.log2(1 + snr * h.abs() ** 2).mean())
