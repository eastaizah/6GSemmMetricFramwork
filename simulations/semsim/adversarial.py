"""Adversarial-resilience metrics: ARR, SASR, MSD_norm, randomized smoothing (CertCost).

Threat model (as in the original script): white-box l_inf PGD on the transmitted
embedding x (before the semantic encoder), against the chain
task_model(decoder(encoder(x + delta))) with a noiseless channel, so that the
attack and the metrics are deterministic for a given seed.  The SASR target is the
clean decision S(x) (definition of Eq. (44): S(x + delta) != S(x)); MSD uses the
cross-entropy w.r.t. the true task label, normalised by ln|S| (Eq. MSD_norm).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

from .common import NUM_CLASSES, flops_forward
from .metrics import clopper_pearson_lower


def _chain(ae, clf, x):
    return clf(ae.dec(ae.enc(x)))


def pgd(ae, clf, x, y, eps, steps=20, alpha=None, gen_=None, rand_init=False,
        smooth_sigma: float = 0.0, eot: int = 1):
    """l_inf PGD maximising CE(chain(x+delta), y).  With smooth_sigma > 0 the loss is
    the CE of the Monte-Carlo averaged softmax over ``eot`` Gaussian draws (EOT attack
    on the smoothed classifier)."""
    if eps <= 0:
        return torch.zeros_like(x)
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    if rand_init:
        delta = (torch.rand(x.shape, generator=gen_) * 2 - 1) * eps
    else:
        delta = torch.zeros_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        with torch.enable_grad():
            if smooth_sigma > 0:
                xx = (x + delta).unsqueeze(0).expand(eot, *x.shape)
                nz = torch.randn(xx.shape, generator=gen_) * smooth_sigma
                pr = F.softmax(_chain(ae, clf, (xx + nz).reshape(-1, x.shape[1])), 1)
                pr = pr.reshape(eot, x.shape[0], -1).mean(0)
                loss = F.nll_loss(torch.log(pr + 1e-12), y)
            else:
                loss = F.cross_entropy(_chain(ae, clf, x + delta), y)
            grad, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * grad.sign()).clamp(-eps, eps)
    return delta.detach()


@torch.no_grad()
def decisions(ae, clf, x):
    return _chain(ae, clf, x).argmax(1)


def sasr(ae, clf, x, eps, steps=20, gen_=None):
    c = decisions(ae, clf, x)
    d = pgd(ae, clf, x, c, eps, steps=steps, gen_=gen_)
    a = decisions(ae, clf, x + d)
    return float((a != c).float().mean())


def arr(ae, clf, x, n_search=10, hi=0.5, steps=20):
    """ARR = eps_threshold = inf{eps : SASR(eps) >= 0.5}, binary search on [0, hi]."""
    lo = 0.0
    c = decisions(ae, clf, x)
    for _ in range(n_search):
        mid = (lo + hi) / 2
        d = pgd(ae, clf, x, c, mid, steps=steps)
        rate = float((decisions(ae, clf, x + d) != c).float().mean())
        if rate >= 0.5:
            hi = mid
        else:
            lo = mid
    return hi


def msd_norm(ae, clf, x, y, eps, steps=20):
    with torch.no_grad():
        lc = F.cross_entropy(_chain(ae, clf, x), y).item()
    d = pgd(ae, clf, x, y, eps, steps=steps)
    with torch.no_grad():
        la = F.cross_entropy(_chain(ae, clf, x + d), y).item()
    lmax = math.log(NUM_CLASSES)
    return float(np.clip((la - lc) / (lmax - lc + 1e-8), 0, 1)), lc, la


# --------------------------------------------------------------- randomized smoothing
@torch.no_grad()
def _smooth_counts(ae, clf, x1, n, sigma, g, chunk=1000):
    cnt = torch.zeros(NUM_CLASSES, dtype=torch.long)
    done = 0
    while done < n:
        m = min(chunk, n - done)
        xx = x1.expand(m, -1) + sigma * torch.randn((m, x1.shape[1]), generator=g)
        cnt += torch.bincount(_chain(ae, clf, xx).argmax(1), minlength=NUM_CLASSES)
        done += m
    return cnt


def certify(ae, clf, x, y, sigma=0.25, n0=100, n=1000, alpha=0.001, gen_=None):
    """Cohen et al. (2019) CERTIFY.  Returns per-sample radius (0 if abstain/wrong)
    plus timing/FLOPs."""
    flops_pass = flops_forward(ae.enc) + flops_forward(ae.dec) + flops_forward(clf)
    radii, correct, abstain = [], [], 0
    t0 = time.perf_counter()
    for i in range(x.shape[0]):
        x1 = x[i:i + 1]
        c0 = _smooth_counts(ae, clf, x1, n0, sigma, gen_)
        cA = int(c0.argmax())
        nA = int(_smooth_counts(ae, clf, x1, n, sigma, gen_)[cA])
        pA = clopper_pearson_lower(nA, n, alpha)
        if pA <= 0.5:
            abstain += 1
            radii.append(0.0); correct.append(False)
        else:
            r = sigma * norm.ppf(pA)
            ok = cA == int(y[i])
            radii.append(r if ok else 0.0); correct.append(ok)
    wall = time.perf_counter() - t0
    radii = np.asarray(radii)
    return {
        "ACR": float(radii.mean()),                     # average certified radius (0 if wrong/abstain)
        "cert_acc_r0": float(np.mean(correct)),
        "cert_acc_r0.25": float(np.mean(radii >= 0.25)),
        "cert_acc_r0.5": float(np.mean(radii >= 0.5)),
        "mean_radius_certified_correct": float(radii[radii > 0].mean()) if (radii > 0).any() else 0.0,
        "abstain_rate": abstain / x.shape[0],
        "wall_s_per_sample": wall / x.shape[0],
        "flops_per_sample": float((n0 + n) * flops_pass),
        "flops_per_pass": float(flops_pass),
        "n_cert": int(x.shape[0]), "sigma": sigma, "n0": n0, "n": n, "alpha": alpha,
    }


@torch.no_grad()
def smoothed_predict(ae, clf, x, sigma, n, g, channel_fn=None):
    """Majority vote of the base chain under input noise (and, optionally, the channel)."""
    votes = torch.zeros(x.shape[0], NUM_CLASSES)
    for _ in range(n):
        xn = x + sigma * torch.randn(x.shape, generator=g)
        if channel_fn is None:
            p = _chain(ae, clf, xn).argmax(1)
        else:
            p = clf(channel_fn(xn)).argmax(1)
        votes[torch.arange(x.shape[0]), p] += 1
    return votes.argmax(1)


def sasr_smoothed(ae, clf, x, eps, sigma=0.25, eot=8, n_vote=100, steps=20, gen_=None,
                  seed_vote=0):
    gv = torch.Generator().manual_seed(seed_vote)
    c = smoothed_predict(ae, clf, x, sigma, n_vote, gv)
    d = pgd(ae, clf, x, c, eps, steps=steps, gen_=gen_, smooth_sigma=sigma, eot=eot)
    gv = torch.Generator().manual_seed(seed_vote)
    a = smoothed_predict(ae, clf, x + d, sigma, n_vote, gv)
    return float((a != c).float().mean())
