"""Metric estimators used by the revision campaign.

Normative (estimator-independent range) metrics
* TSR with Wilson 95 % CI.
* Discrete RSE / NSMI on the task-relevant semantic variable S_T (true task
  label) and its receiver-side estimate S_hat_T (task-model decision on the
  reconstruction): plug-in entropies from the |S|x|S| confusion matrix with the
  Miller-Madow correction, in bits, percentile-bootstrap 95 % CI.

Informative (estimator-dependent) scores
* k-NN RSE/NSMI: Kraskov MI (KSG-1) over Kozachenko-Leonenko differential
  entropy, fixed d_proj = 16 (fixed random orthonormal projection, seed 1234),
  N = 500 sub-sampled points, k = 3.
* S3I (embedding-domain SSIM, as in the original script), SWD (Sinkhorn OT,
  N = 500, eps = 0.1 x median cost, 200 iterations), ID (classifier-based KL,
  nats), ICC, AP, SU, CE, PF, SCI (total-variation distance between task-model
  posteriors).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import digamma, gammaln
from scipy.spatial import cKDTree
from scipy.stats import beta as _beta

LN2 = math.log(2.0)


# ---------------------------------------------------------------------------- TSR
def wilson_ci(p: float, n: int, z: float = 1.959964):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(c - h, 0.0), min(c + h, 1.0)


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided (1 - alpha) lower confidence bound on a binomial proportion."""
    return 0.0 if k == 0 else float(_beta.ppf(alpha, k, n - k + 1))


# --------------------------------------------------------------- discrete RSE / NSMI
def _H_mm(counts: np.ndarray, n: int) -> float:
    """Miller-Madow corrected plug-in entropy (nats)."""
    c = counts[counts > 0]
    p = c / n
    return float(-(p * np.log(p)).sum() + (c.size - 1) / (2 * n))


def _disc_from_confusion(C: np.ndarray):
    n = int(C.sum())
    hs = _H_mm(C.sum(1), n)
    hsh = _H_mm(C.sum(0), n)
    hj = _H_mm(C.ravel(), n)
    mi_raw = hs + hsh - hj
    # Miller-Madow can push the estimate marginally outside [0, min(H)] when I ~ 0;
    # the normative estimator truncates to the feasible range (counted in report).
    mi = min(max(mi_raw, 0.0), hs, hsh)
    rse = mi / hs if hs > 0 else 0.0
    nsmi = mi / math.sqrt(hs * hsh) if hs > 0 and hsh > 0 else 0.0
    return mi / LN2, hs / LN2, hsh / LN2, rse, nsmi, mi_raw / LN2


def confusion(s: np.ndarray, sh: np.ndarray, m: int) -> np.ndarray:
    return np.bincount(s * m + sh, minlength=m * m).reshape(m, m)


def rse_nsmi_discrete(s: np.ndarray, sh: np.ndarray, m: int, n_boot: int = 0,
                      rng: np.random.Generator | None = None) -> dict:
    """Discrete RSE = I(S;S^)/H(S), NSMI = I/sqrt(H(S)H(S^)); bits; bootstrap CI."""
    C = confusion(s, sh, m)
    mi, hs, hsh, rse, nsmi, mi_raw = _disc_from_confusion(C)
    # uncorrected plug-in (exact MI of the empirical joint pmf; always within [0, 1])
    n = C.sum(); pj = C / n; ps_, ph_ = pj.sum(1), pj.sum(0)
    nz = pj > 0
    mi_pi = float((pj[nz] * np.log(pj[nz] / (ps_[:, None] * ph_[None, :])[nz])).sum())
    hs_pi = float(-(ps_[ps_ > 0] * np.log(ps_[ps_ > 0])).sum())
    hh_pi = float(-(ph_[ph_ > 0] * np.log(ph_[ph_ > 0])).sum())
    out = {"I_bits": mi, "H_S_bits": hs, "H_Shat_bits": hsh, "RSE": rse, "NSMI": nsmi,
           "I_bits_MM_untruncated": mi_raw, "RSE_plugin": mi_pi / hs_pi if hs_pi > 0 else 0.0,
           "NSMI_plugin": mi_pi / np.sqrt(hs_pi * hh_pi) if hs_pi > 0 and hh_pi > 0 else 0.0}
    if n_boot:
        rng = rng or np.random.default_rng(0)
        n = s.size
        flat = s * m + sh
        rs, ns = np.empty(n_boot), np.empty(n_boot)
        for b in range(n_boot):
            Cb = np.bincount(flat[rng.integers(0, n, n)], minlength=m * m).reshape(m, m)
            _, _, _, rs[b], ns[b], _ = _disc_from_confusion(Cb)
        out.update(RSE_ci=(float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))),
                   NSMI_ci=(float(np.percentile(ns, 2.5)), float(np.percentile(ns, 97.5))),
                   RSE_boot_min=float(rs.min()), RSE_boot_max=float(rs.max()),
                   NSMI_boot_min=float(ns.min()), NSMI_boot_max=float(ns.max()))
    return out


# ------------------------------------------------------------ k-NN (informative) RSE
KNN_DPROJ, KNN_N, KNN_K, KNN_PROJ_SEED = 16, 500, 3, 1234


def knn_projection(d: int, dproj: int = KNN_DPROJ, seed: int = KNN_PROJ_SEED) -> np.ndarray:
    a = np.random.default_rng(seed).standard_normal((d, dproj))
    q, _ = np.linalg.qr(a)
    return q


def ksg_mi(X: np.ndarray, Y: np.ndarray, k: int = KNN_K) -> float:
    """Kraskov-Stoegbauer-Grassberger estimator (algorithm 1), nats (not clipped)."""
    N = X.shape[0]
    XY = np.hstack([X, Y])
    eps = cKDTree(XY).query(XY, k=k + 1, p=np.inf)[0][:, -1]
    r = np.nextafter(eps, 0)
    nx = cKDTree(X).query_ball_point(X, r, p=np.inf, return_length=True) - 1
    ny = cKDTree(Y).query_ball_point(Y, r, p=np.inf, return_length=True) - 1
    return float(digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def kl_entropy(X: np.ndarray, k: int = KNN_K) -> float:
    """Kozachenko-Leonenko differential entropy (nats, Euclidean balls)."""
    N, d = X.shape
    r = np.maximum(cKDTree(X).query(X, k=k + 1, p=2)[0][:, k], 1e-12)
    log_cd = (d / 2) * math.log(math.pi) - gammaln(d / 2 + 1)
    return float(d * np.mean(np.log(r)) + log_cd - digamma(k) + digamma(N))


def rse_knn(x: np.ndarray, xh: np.ndarray, Q: np.ndarray, rng: np.random.Generator,
            n: int = KNN_N) -> dict:
    idx = rng.choice(x.shape[0], n, replace=False)
    X = x[idx] @ Q
    Y = xh[idx] @ Q
    X = X / (X.std(0) + 1e-8)
    Y = Y / (Y.std(0) + 1e-8)
    mi = ksg_mi(X, Y)
    hx, hy = kl_entropy(X), kl_entropy(Y)
    return {"MI_nats": mi, "Hx_nats": hx, "Hy_nats": hy,
            "RSE_knn": mi / hx,
            "NSMI_knn": mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else float("nan")}


# ---------------------------------------------------------------- S3I, SWD (original)
def s3i(x: np.ndarray, xh: np.ndarray) -> float:
    mu_s = np.linalg.norm(x, axis=1); mu_h = np.linalg.norm(xh, axis=1)
    var_s = np.var(x, axis=1); var_h = np.var(xh, axis=1)
    std_s = np.sqrt(var_s + 1e-12); std_h = np.sqrt(var_h + 1e-12)
    cross = np.mean(x * xh, axis=1)  # uncentred, exactly as the original script
    L = np.sqrt(np.mean(mu_s ** 2)) + 1e-8
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    c3 = c2 / 2
    lum = (2 * mu_s * mu_h + c1) / (mu_s ** 2 + mu_h ** 2 + c1)
    con = (2 * std_s * std_h + c2) / (var_s + var_h + c2)
    stru = (cross + c3) / (std_s * std_h + c3)
    return float(np.clip(np.mean(lum * con * stru), 0, 1))


SWD_N, SWD_EPS, SWD_ITERS = 500, 0.1, 200


def swd(x: np.ndarray, xh: np.ndarray, rng: np.random.Generator) -> float:
    """Entropic OT cost (Sinkhorn, log domain) between empirical tx/rx embedding sets.

    Cost = Euclidean distance; eps = 0.1 x median cost; 200 iterations; N = 500
    points each, uniform weights.  Returned in embedding units.
    """
    idx = rng.choice(x.shape[0], SWD_N, replace=False)
    jdx = rng.choice(x.shape[0], SWD_N, replace=False)
    C = np.sqrt(np.maximum(((x[idx] ** 2).sum(1)[:, None] + (xh[jdx] ** 2).sum(1)[None]
                            - 2 * x[idx] @ xh[jdx].T), 0))
    med = np.median(C)
    eps = SWD_EPS * med
    n = SWD_N
    loga = logb = np.full(n, -math.log(n))
    f = np.zeros(n); gg = np.zeros(n)
    from scipy.special import logsumexp
    for _ in range(SWD_ITERS):
        f = eps * (loga - logsumexp((gg[None, :] - C) / eps, axis=1))
        gg = eps * (logb - logsumexp((f[:, None] - C) / eps, axis=0))
    P = np.exp((f[:, None] + gg[None, :] - C) / eps)
    return float((P * C).sum())


# ---------------------------------------------------------------- intent / task
def posteriors(clf, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return F.softmax(clf(x), dim=1)


def id_clf(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> dict:
    """ID = KL(I_T || I_R) in nats with I = task-model posterior.

    'per_sample': mean over messages of KL(p_i || q_i)  (primary, per-message intent);
    'marginal'  : KL of the batch-averaged posteriors (the V1 definition)."""
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    ps = (p * (p.log() - q.log())).sum(1).mean().item()
    pm, qm = p.mean(0), q.mean(0)
    mg = (pm * (pm.log() - qm.log())).sum().item()
    return {"ID": max(ps, 0.0), "ID_marginal": max(mg, 0.0)}


def tv(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p - q).abs().sum(-1)


def sci_tv(posts: list[torch.Tensor]) -> torch.Tensor:
    """Per-trial SCI = 1 - (1/N^2) sum_{i,j} TV(I_i, I_j)."""
    N = len(posts)
    tot = torch.zeros(posts[0].shape[0])
    for i in range(N):
        for j in range(N):
            if i != j:
                tot += tv(posts[i], posts[j])
    return 1 - tot / N ** 2


def icc(x: np.ndarray, xh: np.ndarray) -> float:
    cs = np.sum(x * xh, 1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(xh, axis=1) + 1e-12)
    return float(np.tanh(np.mean(cs)))


def ap(x: np.ndarray, xh: np.ndarray) -> float:
    diff = np.linalg.norm(x - xh, axis=1)
    dmax = np.linalg.norm(x, axis=1).max() * 2 + 1e-8
    return float(np.clip(np.mean(1 - diff / dmax), 0, 1))


def su(x: np.ndarray, xh: np.ndarray) -> float:
    nx, nh = np.linalg.norm(x, axis=1), np.linalg.norm(xh, axis=1)
    cs = np.sum(x * xh, 1) / (nx * nh + 1e-12)
    delta = np.linalg.norm(x - xh, axis=1) / (nx + 1e-12)
    return float(np.clip(np.mean(cs * np.exp(-2 * delta)), 0, 1))
