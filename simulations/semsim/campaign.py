"""One seed of the campaign: trains every model once and evaluates every part.

All tables of the revised Section 11 are produced from the models trained here
(one oracle, five proposed encoders/decoders k = 8..128, five DeepJSCC-type
B1 models, one PGD/noise-augmented proposed model per seed and per dataset).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from . import adversarial as A
from . import metrics as M
from .common import (CHANNELS, K_VALUES, NUM_CLASSES, SNR_GRID, AE, analog_link, channel_gains,
                     clip_dataset, gaussian_dataset, gen, oracle_net, run_analog, sub_seed,
                     train_ae, train_oracle)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "cache")
CLIP_PATH = os.path.join(CACHE, "clip_vitb32_cifar10.npz")

FULL = dict(n_train=50000, n_test=10000, ae_iters=3000, oracle_iters=3000, adv_iters=3000,
            n_boot=1000, n_knn_rep=20, n_adv=500, n_cert=500, n_b3=500, n_dp=2000,
            snr_grid=SNR_GRID, k_values=K_VALUES, channels=CHANNELS,
            uav_sessions=100, uav_T=100, prop1_trials=10000)
QUICK = dict(n_train=5000, n_test=2000, ae_iters=300, oracle_iters=500, adv_iters=100,
             n_boot=100, n_knn_rep=3, n_adv=100, n_cert=30, n_b3=100, n_dp=300,
             snr_grid=[0.0, 10.0, 20.0], k_values=[16, 32], channels=["AWGN", "Rayleigh"],
             uav_sessions=10, uav_T=20, prop1_trials=500)

ADV_EPS = 8 / 255
EPS_LIST = [0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 12 / 255, 16 / 255, 24 / 255, 32 / 255]
ADV_TRAIN = {"eps": 8 / 255, "steps": 7, "sigma": 0.25}
SIGMA_SMOOTH = 0.25


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------- model cache
def _mdir(ds, seed, quick):
    d = os.path.join(CACHE, "models", f"{'quick_' if quick else ''}{ds.name}_seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def get_oracle(ds, seed, cfg, quick):
    p = os.path.join(_mdir(ds, seed, quick), "oracle.pt")
    if os.path.exists(p):
        st = torch.load(p)
        clf = oracle_net(ds.d); clf.load_state_dict(st["sd"]); clf.eval()
        for q in clf.parameters():
            q.requires_grad_(False)
        return clf, st["info"]
    clf, info = train_oracle(ds, seed, iters=cfg["oracle_iters"], wd=1e-2)
    torch.save({"sd": clf.state_dict(), "info": info}, p)
    return clf, info


def get_ae(ds, seed, k, clf, cfg, quick, tag, lam=0.02, adv=None):
    p = os.path.join(_mdir(ds, seed, quick), f"{tag}_k{k}.pt")
    if os.path.exists(p):
        ae = AE(ds.d, k); ae.load_state_dict(torch.load(p)); ae.eval()
        for q in ae.parameters():
            q.requires_grad_(False)
        return ae, 0.0
    t0 = time.time()
    it = cfg["adv_iters"] if adv else cfg["ae_iters"]
    ae = train_ae(ds, k, clf, seed, lam=lam, iters=it, adv=adv, tag=tag)
    torch.save(ae.state_dict(), p)
    return ae, time.time() - t0


# --------------------------------------------------------------------- evaluation
def eval_point(clf, x, y, xh, rich: bool, n_boot: int, rng, Q=None, knn=False, swd=False):
    pt = M.posteriors(clf, x)
    ph = M.posteriors(clf, xh)
    yhat = ph.argmax(1).numpy()
    yn = y.numpy()
    n = yn.size
    tsr = float((yhat == yn).mean())
    lo, hi = M.wilson_ci(tsr, n)
    r = {"TSR": tsr, "TSR_lo": lo, "TSR_hi": hi, "N": n,
         "agree": float((yhat == pt.argmax(1).numpy()).mean())}
    r.update(M.id_clf(pt, ph))
    r["TV_mean"] = float(M.tv(pt, ph).mean())
    d = M.rse_nsmi_discrete(yn, yhat, NUM_CLASSES, n_boot=n_boot if rich else 0, rng=rng)
    r.update({f"disc_{k}": v for k, v in d.items()})
    xn, xhn = x.numpy(), xh.numpy()
    r["S3I"] = M.s3i(xn, xhn)
    if swd:
        r["SWD"] = M.swd(xn, xhn, rng)
    if knn:
        kk = M.rse_knn(xn, xhn, Q, rng)
        r.update({f"knn_{k}": v for k, v in kk.items()})
    if rich:
        ic = M.icc(xn, xhn)
        r.update(AP=M.ap(xn, xhn), SU=M.su(xn, xhn), ICC=ic, PF=float(np.clip(tsr * max(ic, 0), 0, 1)))
    return r


def part_main(ds, seed, clf, models, cfg, out, tagset=("prop", "b1", "adv")):
    """SNR x k x channel sweep with the SAME models (fidelity / TSR / multi-channel tables)."""
    Q = M.knn_projection(ds.d)
    x, y = ds.x_test, ds.y_test
    rows = []
    for tag in tagset:
        for k in cfg["k_values"]:
            if (tag, k) not in models:
                continue
            ae = models[(tag, k)]
            for ch in cfg["channels"]:
                for snr in cfg["snr_grid"]:
                    g = gen("eval", seed, ds.name, tag, k, ch, snr)
                    xh = run_analog(ae, x, ch, snr, g)
                    key = snr in (0.0, 10.0, 20.0)
                    rng = np.random.default_rng(sub_seed("np", seed, ds.name, tag, k, ch, snr))
                    r = eval_point(clf, x, y, xh, rich=key and tag == "prop", n_boot=cfg["n_boot"], rng=rng,
                                   Q=Q, knn=key and tag == "prop", swd=key and tag == "prop")
                    r.update(dataset=ds.name, seed=seed, scheme=tag, k=k, channel=ch, snr=snr,
                             real_dims=k, n_sym=k // 2)
                    if key and tag == "prop" and k == 32:
                        # SCI over 5 receivers at the same SNR (TV distance), as in Table 10 context
                        posts = [M.posteriors(clf, run_analog(ae, x, ch, snr, gen("sci", seed, ch, snr, i)))
                                 for i in range(5)]
                        r["SCI5"] = float(M.sci_tv(posts).mean())
                        r["CE"] = r["TSR"] / (k / ds.d)
                    rows.append(r)
        log(ds.name, "seed", seed, "main done", tag)
    # clean (no-channel) reference for every model
    clean = []
    with torch.no_grad():
        for (tag, k), ae in models.items():
            xh = ae.forward_clean(x)
            clean.append({"dataset": ds.name, "seed": seed, "scheme": tag, "k": k,
                          "TSR_noiseless": float((clf(xh).argmax(1) == y).float().mean())})
    out["main"] = rows
    out["clean"] = clean


def part_estimators(ds, seed, clf, models, cfg, out):
    """Estimator variance (R2.2) and the empirical I(Z;Z^) of SELF-4 at k=32/AWGN/10 dB."""
    ae = models[("prop", 32)]
    x = ds.x_test
    Q = M.knn_projection(ds.d)
    g = gen("est", seed)
    xh = run_analog(ae, x, "AWGN", 10.0, g)
    rng = np.random.default_rng(sub_seed("knnrep", seed))
    reps = [M.rse_knn(x.numpy(), xh.numpy(), Q, rng) for _ in range(cfg["n_knn_rep"])]
    with torch.no_grad():
        z = ae.enc(x)
        zh = analog_link(z, float(ae.P), 10.0, torch.ones(z.shape[0], 16, dtype=torch.complex64), g)
    zrep = []
    for _ in range(cfg["n_knn_rep"]):
        idx = rng.choice(z.shape[0], M.KNN_N, replace=False)
        zrep.append(M.ksg_mi(z[idx].numpy(), zh[idx].numpy()))
    # proj-seed sensitivity: same data, different projection seeds
    proj = []
    for ps in range(10):
        Qp = M.knn_projection(ds.d, seed=ps)
        proj.append(M.rse_knn(x.numpy(), xh.numpy(), Qp, np.random.default_rng(0))["RSE_knn"])
    # dproj / N / k sensitivity (shows why the parameters must be fixed)
    sens = []
    for dp in (8, 16, 32):
        for nn_ in (250, 500, 1000):
            Qd = M.knn_projection(ds.d, dproj=dp)
            sens.append({"dproj": dp, "N": nn_, "RSE_knn": M.rse_knn(x.numpy(), xh.numpy(), Qd,
                                                                    np.random.default_rng(1), n=nn_)["RSE_knn"]})
    out["estimators"] = {"knn_within_seed": reps, "I_Z_Zhat_nats": zrep, "proj_seed_RSE": proj,
                         "sensitivity": sens,
                         "awgn_capacity_nats_k32_10dB": 16 * math.log(1 + 10.0)}


def part_digital(ds, seed, clf, models, cfg, out):
    from . import digital as Dg
    x, y = ds.x_test, ds.y_test
    rows = []
    pca = Dg.PCACoder(ds.x_train)
    nv = min(5000, ds.x_train.shape[0])
    xv, yv = ds.x_train[:nv], ds.y_train[:nv]
    t0 = time.time()
    curve_val = Dg.source_curve(pca, clf, xv, yv, 300)
    # test-set TSR of the val-selected (m, b) for every payload K (for the analytical bound)
    curve = {}
    for K, (_, m, b) in curve_val.items():
        if m == 0:
            curve[K] = (float((clf(pca.synthesis(torch.zeros(x.shape[0], 0), 0)).argmax(1) == y).float().mean()), 0, 0)
            continue
        q = pca.quantizer(m, b)
        curve[K] = (Dg.tsr_of(clf, pca.synthesis(q.decode(q.encode(pca.analysis(x, m))), m), y), m, b)
    out["source_curve"] = {str(K): v for K, v in curve.items()}
    log(ds.name, "seed", seed, "source curve", round(time.time() - t0))

    def run_scheme(sch, xs, ys, chs, snrs, extra):
        for ch in chs:
            for snr in snrs:
                g = gen("dig", seed, ds.name, sch.desc["scheme"], sch.desc["n_sym"], sch.desc["mod"], sch.desc["K"], ch, snr)
                xh = sch.run(xs, ch, snr, g)
                with torch.no_grad():
                    pred = clf(xh).argmax(1)
                tsr = float((pred == ys).float().mean())
                lo, hi = M.wilson_ci(tsr, ys.numel())
                pt, ph = M.posteriors(clf, xs), M.posteriors(clf, xh)
                d = M.rse_nsmi_discrete(ys.numpy(), pred.numpy(), NUM_CLASSES)
                r = dict(sch.desc, dataset=ds.name, seed=seed, channel=ch, snr=snr, TSR=tsr, TSR_lo=lo,
                         TSR_hi=hi, N=ys.numel(), disc_RSE=d["RSE"], disc_NSMI=d["NSMI"], **M.id_clf(pt, ph))
                r.update(extra)
                rows.append(r)

    chs = [c for c in ("AWGN", "Rayleigh") if c in cfg["channels"]]
    # B2 at matched channel uses (k = 32 and 64 real dims) for every MCS, full SNR grid
    for n_sym in (16, 32):
        for mcs in Dg.MCS_LIST:
            K = Dg.k_info(n_sym, mcs[1], mcs[2])
            _, m, b = curve_val[min(K, 300)]
            if m == 0:
                continue
            sch = Dg.SeparateScheme(pca, n_sym, mcs, m, b)
            run_scheme(sch, x, y, chs, cfg["snr_grid"], {"noiseless_TSR": curve[m * b][0]})
    log(ds.name, "seed", seed, "B2 done", round(time.time() - t0))
    # B2 with larger channel budgets, 10 dB only (for the matched-TSR channel-use search)
    if ds.name == "gauss":
        for n_sym in (64, 128):
            for mcs in Dg.MCS_LIST:
                K = Dg.k_info(n_sym, mcs[1], mcs[2])
                _, m, b = curve_val[min(K, 300)]
                sch = Dg.SeparateScheme(pca, n_sym, mcs, m, b)
                run_scheme(sch, x, y, chs, [10.0], {"noiseless_TSR": curve[min(m * b, 300)][0]})
    # B3 bit-exact (8-bit full embedding)
    if ds.name == "gauss":
        nb = cfg["n_b3"]
        for mcs in (("QPSK", 2, 1 / 2), ("16QAM", 4, 1 / 2)):
            sch = Dg.BitExactScheme(ds.x_train, mcs)
            with torch.no_grad():
                nl = Dg.tsr_of(clf, sch.q.decode(sch.q.encode(x[:nb])), y[:nb])
            run_scheme(sch, x[:nb], y[:nb], chs, cfg["snr_grid"], {"noiseless_TSR": nl})
        log(ds.name, "seed", seed, "B3 done", round(time.time() - t0))
    # digital variant of the proposed encoder
    nd = cfg["n_dp"]
    for b in (4, 8):
        for mcs in (("QPSK", 2, 1 / 2), ("16QAM", 4, 1 / 2)):
            if ds.name == "clip" and (b, mcs[0]) != (4, "QPSK"):
                continue
            sch = Dg.DigitalProposed(models[("prop", 32)], ds.x_train, mcs, b)
            nl = Dg.tsr_of(clf, sch.run_noiseless(x[:nd]), y[:nd])
            run_scheme(sch, x[:nd], y[:nd], chs, cfg["snr_grid"], {"noiseless_TSR": nl})
    log(ds.name, "seed", seed, "DP done", round(time.time() - t0))
    out["digital"] = rows
    # analytical bound: K = floor(n_sym * C) bits through an ideal code + same source coder
    bound = []
    for n_sym in (16, 32):
        for ch in chs:
            for snr in cfg["snr_grid"]:
                C = Dg.capacity_bits_per_use(ch, snr)
                K = int(math.floor(n_sym * C))
                bound.append({"dataset": ds.name, "seed": seed, "n_sym": n_sym, "channel": ch, "snr": snr,
                              "C_bits_per_use": C, "K_bits": K, "TSR": curve[min(max(K, 1), 300)][0]})
    out["bound"] = bound


def part_adv(ds, seed, clf, models, cfg, out):
    x, y = ds.x_test[:cfg["n_adv"]], ds.y_test[:cfg["n_adv"]]
    res = {}
    for tag in ("prop", "b1", "adv"):
        ae = models[(tag, 32)]
        g = gen("pgd", seed, tag)
        curve = [A.sasr(ae, clf, x, e, gen_=g) for e in EPS_LIST]
        with torch.no_grad():
            err = float((A.decisions(ae, clf, x) != y).float().mean())
        msd, lc, la = A.msd_norm(ae, clf, x, y, ADV_EPS)
        res[tag] = {"eps": EPS_LIST, "SASR": curve, "ARR": A.arr(ae, clf, x), "MSD_norm": msd,
                    "CE_clean": lc, "CE_adv": la, "clean_error": err,
                    "SASR_8_255": curve[EPS_LIST.index(8 / 255)]}
        log(ds.name, "seed", seed, "adv", tag, res[tag]["ARR"], res[tag]["SASR_8_255"])
    # randomized smoothing (Cohen et al.) of the standard and the adversarially trained chain
    xc, yc = ds.x_test[:cfg["n_cert"]], ds.y_test[:cfg["n_cert"]]
    for tag in ("prop", "adv"):
        res[tag]["cert"] = A.certify(models[(tag, 32)], clf, xc, yc, sigma=SIGMA_SMOOTH,
                                     gen_=gen("cert", seed, tag))
    # SASR of the smoothed adversarially trained chain under an adaptive EOT-PGD attack
    xs = x[: min(200, x.shape[0])]
    res["adv"]["SASR_smoothed"] = [A.sasr_smoothed(models[("adv", 32)], clf, xs, e, sigma=SIGMA_SMOOTH,
                                                   gen_=gen("eot", seed, e), seed_vote=sub_seed("vote", seed))
                                   for e in EPS_LIST]
    # TSR of the smoothed chain over AWGN at 10 dB (channel inside the vote; n = 100 votes)
    ae = models[("adv", 32)]
    g = gen("smtsr", seed)
    ch_fn = lambda xn: ae.dec(analog_link(ae.enc(xn), float(ae.P), 10.0,
                                          torch.ones(xn.shape[0], 16, dtype=torch.complex64), g))
    xt, yt = ds.x_test[:2000], ds.y_test[:2000]
    pred = A.smoothed_predict(ae, clf, xt, SIGMA_SMOOTH, 100, g, channel_fn=ch_fn)
    res["adv"]["TSR_smoothed_awgn10"] = float((pred == yt).float().mean())
    res["adv"]["TSR_smoothed_N"] = int(yt.numel())
    out["adv"] = res


def part_uav(ds, seed, clf, models, cfg, out):
    from . import uav as U
    ae = models[("prop", 32)]
    S, T = cfg["uav_sessions"], cfg["uav_T"]
    n = S * T
    x, y = ds.x_test[:n], ds.y_test[:n]
    with torch.no_grad():
        clean_acc = float((clf(ae.forward_clean(x)).argmax(1) == y).float().mean())
    pt = M.posteriors(clf, x)
    rows = []
    thetas = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    lams = [0.01, 0.03, 0.1, 0.3, 1.0]
    grid = [(th, 0.1) for th in thetas] + [(th, l) for th in (30, 60) for l in lams if l != 0.1]
    for snr in (10.0, 20.0):
        for th, lam in grid:
            g = gen("uav", seed, snr, th, lam)
            h, st = U.uav_gains(S, T, 16, th, lam, g)
            xh = run_analog(ae, x, "UAV", snr, g, h=h)
            ph = M.posteriors(clf, xh)
            ok = (ph.argmax(1) == y).float().reshape(S, T)
            sess_tsr = ok.mean(1)
            idm = []
            for s_ in range(S):
                sl = slice(s_ * T, (s_ + 1) * T)
                idm.append(M.id_clf(pt[sl], ph[sl])["ID_marginal"])
            r = {"dataset": ds.name, "seed": seed, "snr_nom": snr, "theta": th, "lam": lam,
                 "P_LoS": U.p_los(th), "K_dB": 10 * math.log10(U.k_factor(th)),
                 "frac_LoS": float(st.float().mean()), "TSR": float(ok.mean()),
                 "TSR_LoS": float(ok.reshape(-1)[st].mean()) if st.any() else float("nan"),
                 "TSR_blocked": float(ok.reshape(-1)[~st].mean()) if (~st).any() else float("nan"),
                 "sess_TSR_p5": float(np.percentile(sess_tsr.numpy(), 5)),
                 "sess_TSR_std": float(sess_tsr.std()),
                 "sess_outage": float((sess_tsr < 0.9 * clean_acc).float().mean()),
                 "ID": M.id_clf(pt, ph)["ID"], "ID_session_marginal_mean": float(np.mean(idm)),
                 "clean_acc": clean_acc, "sessions": S, "T": T}
            rows.append(r)
    out["uav"] = rows


def part_prop1(ds, seed, clf, models, cfg, out):
    ae = models[("prop", 32)]
    n = cfg["prop1_trials"]
    x = ds.x_test[:n]
    pt = M.posteriors(clf, x)
    res = []
    for ch in ("AWGN", "Rayleigh"):
        for N in (2, 4, 8):
            g = gen("prop1", seed, ch, N)
            snrs = torch.rand(N, generator=g) * 30 - 5   # one SNR per receiver in [-5, 25] dB
            posts, deltas = [], []
            for i in range(N):
                ph = M.posteriors(clf, run_analog(ae, x, ch, float(snrs[i]), g))
                posts.append(ph)
                deltas.append(M.tv(ph, pt))
            sci = M.sci_tv(posts)
            dbar = torch.stack(deltas).mean(0)
            bound = 1 - 2 * (N - 1) / N * dbar
            slack = sci - bound
            res.append({"dataset": ds.name, "seed": seed, "channel": ch, "N": N,
                        "snrs": [round(float(s), 2) for s in snrs], "trials": n,
                        "violations": int((slack < -1e-6).sum()), "min_slack": float(slack.min()),
                        "mean_slack": float(slack.mean()), "mean_SCI": float(sci.mean()),
                        "mean_bound": float(bound.mean()), "mean_delta": float(dbar.mean()),
                        "mean_Dhat": float((1 - sci).mean()),
                        "ratio_Dhat_over_boundgap": float(((1 - sci).sum() / (1 - bound).sum()))})
    out["prop1"] = res


# --------------------------------------------------------------------- driver
def run_seed(seed: int, quick: bool, parts: list[str], out_dir: str, datasets=("gauss", "clip")):
    cfg = QUICK if quick else FULL
    torch.set_num_threads(int(os.environ.get("SIM_THREADS", "4")))
    for dname in datasets:
        if dname == "clip" and "clip" not in parts:
            continue
        if dname == "gauss" and not any(p in parts for p in ("main", "digital", "adv", "uav", "prop1", "est")):
            continue
        t_seed = time.time()
        if dname == "gauss":
            ds = gaussian_dataset(seed, cfg["n_train"], cfg["n_test"])
        else:
            if not os.path.exists(CLIP_PATH):
                log("CLIP cache missing -> skipped (run simulations/clip_embed.py)")
                continue
            ds = clip_dataset(CLIP_PATH, n_test=None if not quick else cfg["n_test"])
        out = {"dataset": ds.name, "seed": seed, "quick": quick, "info": dict(ds.info), "timing": {}}
        clf, oinfo = get_oracle(ds, seed, cfg, quick)
        out["info"].update(oinfo)
        log(ds.name, "seed", seed, "oracle", oinfo)
        models, ttrain = {}, {}
        ks = cfg["k_values"] if dname == "gauss" else cfg["k_values"]
        for k in ks:
            models[("prop", k)], ttrain[f"prop_k{k}"] = get_ae(ds, seed, k, clf, cfg, quick, "prop")
            if dname == "gauss" or k == 32:
                models[("b1", k)], ttrain[f"b1_k{k}"] = get_ae(ds, seed, k, clf, cfg, quick, "b1", lam=0.0)
        if dname == "gauss" and ("adv" in parts or "main" in parts):
            models[("adv", 32)], ttrain["adv_k32"] = get_ae(ds, seed, 32, clf, cfg, quick, "adv", adv=ADV_TRAIN)
        out["timing"]["train_s"] = ttrain
        log(ds.name, "seed", seed, "models ready", {k: round(v) for k, v in ttrain.items()})
        todo = ["main", "est", "digital", "adv", "uav", "prop1"] if dname == "gauss" else ["main", "digital"]
        fns = {"main": part_main, "est": part_estimators, "digital": part_digital, "adv": part_adv,
               "uav": part_uav, "prop1": part_prop1}
        for p in todo:
            if p not in parts and not (dname == "clip" and "clip" in parts):
                continue
            t0 = time.time()
            fns[p](ds, seed, clf, models, cfg, out)
            out["timing"][p] = time.time() - t0
            log(ds.name, "seed", seed, "part", p, "done in", round(out["timing"][p]), "s")
        out["timing"]["total_s"] = time.time() - t_seed
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.join(out_dir, f"{ds.name}_seed{seed}.json")
        with open(fn, "w") as f:
            json.dump(out, f, default=float)
        log("wrote", fn)
