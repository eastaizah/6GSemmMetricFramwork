"""Aggregation of the per-seed raw results: CSV tables, markdown tables, summary.json, figures.

Conventions: "mean +/- std" = mean and sample standard deviation (ddof = 1) across
the R seeds; Wilson 95 % CIs are those of the reference seed (seed 0).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(os.path.dirname(HERE), "work", "figs")
REF_SEED = 0

# validated categorical palette (dataviz reference instance, light mode), fixed order
C = {"prop": "#2a78d6", "b1": "#eb6834", "b2": "#1baf7a", "b3": "#eda100", "dp": "#e87ba4",
     "dp8": "#008300", "adv": "#6250d6", "bound": "#52514e", "smooth": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CH_COL = {"AWGN": "#2a78d6", "Rayleigh": "#eb6834", "Rician_K5": "#1baf7a", "Rician_K10": "#eda100",
          "TDL-A": "#6250d6"}
CH_MK = {"AWGN": "o", "Rayleigh": "s", "Rician_K5": "^", "Rician_K10": "v", "TDL-A": "D"}
CH_NAME = {"AWGN": "AWGN", "Rayleigh": "Rayleigh", "Rician_K5": "Rician $K$=5 dB",
           "Rician_K10": "Rician $K$=10 dB", "TDL-A": "TDL-A (simplified)"}


# ----------------------------------------------------------------------------- utils
def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    a = np.asarray(v, float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, a.size


def f3(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def pm(v, d=3):
    m, s, n = ms(v)
    return f"{m:.{d}f} ± {s:.{d}f}" if n else "n/a"


def load(res_dir):
    raw = defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(res_dir, "raw", "*_seed*.json"))):
        d = json.load(open(fn))
        raw[d["dataset"]][d["seed"]] = d
    return raw


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------------- figure style
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.2,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False,
        "savefig.dpi": 300, "font.family": "DejaVu Sans"})
    return plt


def _save(fig, name):
    d = os.path.join(HERE, "results_quick", "figs") if name.endswith("_quick.png") else FIG_DIR
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------ build
def build_all(res_dir, quick=False):
    raw = load(res_dir)
    G = raw.get("gauss", {})
    CL = raw.get("clip", {})
    seeds = sorted(G)
    main = [r for s in seeds for r in G[s]["main"]]
    dig = [r for s in seeds for r in G[s].get("digital", [])]
    bnd = [r for s in seeds for r in G[s].get("bound", [])]
    cmain = [r for s in sorted(CL) for r in CL[s]["main"]]
    cdig = [r for s in sorted(CL) for r in CL[s].get("digital", [])]
    cbnd = [r for s in sorted(CL) for r in CL[s].get("bound", [])]
    uav = [r for s in seeds for r in G[s].get("uav", [])]
    p1 = [r for s in seeds for r in G[s].get("prop1", [])]
    snrs = sorted({r["snr"] for r in main})
    ks = sorted({r["k"] for r in main if r["scheme"] == "prop"})
    chans = [c for c in ["AWGN", "Rayleigh", "Rician_K5", "Rician_K10", "TDL-A"] if any(r["channel"] == c for r in main)]
    T = {}           # markdown tables
    S = {}           # summary
    n_test = G[seeds[0]]["info"]["n_test"]

    for name, rows in [("main_gauss", main), ("digital_gauss", dig), ("bound_gauss", bnd), ("main_clip", cmain),
                       ("digital_clip", cdig), ("bound_clip", cbnd), ("uav", uav), ("prop1", p1)]:
        write_csv(os.path.join(res_dir, f"{name}.csv"), rows)

    def M(rows, key, **kw):
        return [r.get(key) for r in sel(rows, **kw)]

    def ref(rows, **kw):
        rr = sel(rows, seed=REF_SEED, **kw)
        return rr[0] if rr else None

    info = {s: G[s]["info"] for s in seeds}
    T["setup"] = md_table(
        ["Quantity", "Value (mean ± std over seeds)"],
        [["Seeds R", len(seeds)], ["N_train / N_test (Gaussian proxy)", f"{info[seeds[0]]['n_train']} / {n_test}"],
         ["Bayes-optimal accuracy (test)", pm([i["bayes_acc_test"] for i in info.values()])],
         ["Task model (oracle) clean test accuracy", pm([i["oracle_test_acc"] for i in info.values()])],
         ["Task model clean train accuracy", pm([i["oracle_train_acc"] for i in info.values()])]] +
        ([["CLIP: N_train / N_test", f"{CL[min(CL)]['info']['n_train']} / {CL[min(CL)]['info']['n_test']}"],
          ["CLIP task model clean test accuracy", pm([CL[s]['info']['oracle_test_acc'] for s in CL])]] if CL else []))
    S["oracle_clean_acc"] = round(ms([i["oracle_test_acc"] for i in info.values()])[0], 3)
    S["bayes_acc"] = round(ms([i["bayes_acc_test"] for i in info.values()])[0], 3)

    # ---- Table 7 replacement: fidelity vs k (AWGN, 10 dB)
    rows7 = []
    for k in ks:
        kw = dict(scheme="prop", k=k, channel="AWGN", snr=10.0)
        rows7.append([k, f"{100 * k / 512:.2f}", pm(M(main, "TSR", **kw)), pm(M(main, "disc_RSE", **kw)),
                      pm(M(main, "disc_NSMI", **kw)), pm(M(main, "knn_RSE_knn", **kw), 4),
                      pm(M(main, "S3I", **kw)), pm(M(main, "SWD", **kw), 2), pm(M(main, "ID", **kw))])
    T["t7"] = md_table(["k", "ρ (%)", "TSR", "RSE (discrete, bits/bits)", "NSMI (discrete)",
                        "RSE_kNN (informative)", "S³I", "SWD", "ID (nats)"], rows7)

    # ---- Table 8 replacement: TSR vs k across channels, 10 dB
    rows8 = []
    for ch in chans:
        rows8.append([CH_NAME[ch].replace("$", "")] + [pm(M(main, "TSR", scheme="prop", k=k, channel=ch, snr=10.0)) for k in ks])
    T["t8"] = md_table(["Channel"] + [f"TSR (k={k})" for k in ks], rows8)
    rows8ci = []
    for ch in chans:
        cells = []
        for k in ks:
            r = ref(main, scheme="prop", k=k, channel=ch, snr=10.0)
            cells.append(f"{r['TSR']:.3f} [{r['TSR_lo']:.3f}, {r['TSR_hi']:.3f}]" if r else "n/a")
        rows8ci.append([CH_NAME[ch].replace("$", "")] + cells)
    T["t8_refci"] = md_table(["Channel (seed 0, Wilson 95% CI)"] + [f"k={k}" for k in ks], rows8ci)

    # ---- B2 envelope helper (best MCS per SNR, chosen on the seed-mean TSR)
    def b2_best(drows, n_sym, ch, snr):
        cands = defaultdict(list)
        for r in sel(drows, scheme="B2", n_sym=n_sym, channel=ch, snr=snr):
            cands[(r["mod"], round(r["rate"], 3), r["m"], r["b"])].append(r)
        if not cands:
            return None, []
        best = max(cands, key=lambda c: np.mean([r["TSR"] for r in cands[c]]))
        return best, cands[best]

    def dsel(drows, scheme, mod, ch, snr, n_sym=None):
        kw = dict(scheme=scheme, mod=mod, channel=ch, snr=snr)
        if n_sym:
            kw["n_sym"] = n_sym
        return sel(drows, **kw)

    # ---- Table 9 replacement: TSR vs SNR (AWGN and Rayleigh)
    for ch in ["AWGN", "Rayleigh"]:
        rows9 = []
        for snr in snrs:
            best16, r16 = b2_best(dig, 16, ch, snr)
            b3 = dsel(dig, "B3", "QPSK", ch, snr)
            dp = dsel(dig, "DP-q4", "QPSK", ch, snr)
            bd = sel(bnd, n_sym=16, channel=ch, snr=snr)
            rows9.append([f"{snr:+.1f}", pm(M(main, "TSR", scheme="prop", k=32, channel=ch, snr=snr)),
                          pm(M(main, "TSR", scheme="b1", k=32, channel=ch, snr=snr)),
                          pm(M(main, "TSR", scheme="adv", k=32, channel=ch, snr=snr)),
                          (pm([r["TSR"] for r in r16]) + f" ({best16[0]} R={best16[1]:.2f})") if best16 else "n/a",
                          pm([r["TSR"] for r in dp]), pm([r["TSR"] for r in b3]), pm([r["TSR"] for r in bd])])
        T[f"t9_{ch}"] = md_table(["SNR (dB)", "Proposed k=32 (analog)", "B1 DeepJSCC-type k=32",
                                  "Proposed, PGD-trained k=32", "B2 separate, 16 c.u. (best MCS)",
                                  "DP-q4 (128 bits, QPSK R=1/2)", "B3 bit-exact (4096 bits, QPSK R=1/2)",
                                  "Analytical bound, 16 c.u."], rows9)

    # ---- Table 10 replacement: multi-channel at k=32, 10 and 20 dB
    rows10 = []
    for snr in (10.0, 20.0):
        base = ms(M(main, "TSR", scheme="prop", k=32, channel="AWGN", snr=snr))[0]
        for ch in chans:
            kw = dict(scheme="prop", k=32, channel=ch, snr=snr)
            r = ref(main, **kw)
            m_ = ms(M(main, "TSR", **kw))[0]
            rows10.append([CH_NAME[ch].replace("$", ""), int(snr), pm(M(main, "TSR", **kw)),
                           f"[{r['TSR_lo']:.3f}, {r['TSR_hi']:.3f}]" if r else "n/a",
                           pm(M(main, "disc_RSE", **kw)), pm(M(main, "S3I", **kw)), pm(M(main, "SWD", **kw), 2),
                           pm(M(main, "ID", **kw)),
                           "— (reference)" if ch == "AWGN" else f"{100 * (m_ - base) / base:+.1f}%"])
    T["t10"] = md_table(["Channel", "SNR (dB)", "TSR", "Seed-0 Wilson 95% CI", "RSE (discrete)", "S³I", "SWD",
                         "ID (nats)", "TSR Δ vs AWGN"], rows10)

    # ---- all 16 metrics at the reference point
    kw = dict(scheme="prop", k=32, channel="AWGN", snr=10.0)
    T["all16"] = md_table(["Metric", "Value (k=32, AWGN, 10 dB)"], [
        ["RSE (discrete)", pm(M(main, "disc_RSE", **kw))], ["NSMI (discrete)", pm(M(main, "disc_NSMI", **kw))],
        ["RSE_kNN (informative)", pm(M(main, "knn_RSE_knn", **kw), 4)], ["S³I", pm(M(main, "S3I", **kw))],
        ["SWD", pm(M(main, "SWD", **kw), 2)], ["TSR", pm(M(main, "TSR", **kw))],
        ["AP", pm(M(main, "AP", **kw))], ["SU", pm(M(main, "SU", **kw))],
        ["CE = TSR/ρ", pm(M(main, "CE", **kw), 2)], ["ID (per-message KL, nats)", pm(M(main, "ID", **kw))],
        ["ID (V1 marginal definition, nats)", pm(M(main, "ID_marginal", **kw), 4)],
        ["ICC", pm(M(main, "ICC", **kw))], ["SCI (5 receivers, TV)", pm(M(main, "SCI5", **kw))],
        ["PF", pm(M(main, "PF", **kw))]])

    # ---- adversarial
    adv = {s: G[s].get("adv") for s in seeds if G[s].get("adv")}
    eps = adv[seeds[0]]["prop"]["eps"] if adv else []

    def advv(tag, key):
        return [adv[s][tag][key] for s in adv]

    def thr_from_curve(e, c):
        for i in range(1, len(e)):
            if c[i] >= 0.5:
                e0, e1, c0, c1 = e[i - 1], e[i], c[i - 1], c[i]
                return e0 + (0.5 - c0) * (e1 - e0) / (c1 - c0) if c1 > c0 else e1
        return float("nan")

    if adv:
        sm_thr = [thr_from_curve(eps, adv[s]["adv"]["SASR_smoothed"]) for s in adv]
        rows_adv = []
        for tag, lab in [("prop", "Proposed k=32, standard training"), ("b1", "B1 DeepJSCC-type k=32"),
                         ("adv", "Proposed k=32, PGD + noise-augmented training")]:
            rows_adv.append([lab, pm(advv(tag, "clean_error")), pm(advv(tag, "ARR"), 4),
                             pm(advv(tag, "SASR_8_255")), pm(advv(tag, "MSD_norm")),
                             pm([adv[s][tag]["cert"]["ACR"] for s in adv]) if "cert" in adv[seeds[0]][tag] else "n/a",
                             pm([adv[s][tag]["cert"]["cert_acc_r0.25"] for s in adv]) if "cert" in adv[seeds[0]][tag] else "n/a"])
        rows_adv.append(["Proposed k=32, PGD-trained + randomized smoothing (σ=0.25)", "—",
                         pm(sm_thr, 4) + " (EOT-PGD threshold)",
                         pm([adv[s]["adv"]["SASR_smoothed"][eps.index(8 / 255)] for s in adv]), "—",
                         "(see row above)", "—"])
        T["adv"] = md_table(["Model", "Clean decision error (noiseless)", "ARR = ε_threshold (l∞)",
                             "SASR(ε=8/255)", "MSD_norm(8/255)", "ACR (certified l2, σ=0.25)",
                             "Certified acc. at r=0.25"], rows_adv)
        cert = adv[seeds[0]]["adv"]["cert"]
        T["certcost"] = md_table(["Quantity", "Value"], [
            ["n0 / n / α", f"{cert['n0']} / {cert['n']} / {cert['alpha']}"],
            ["Forward FLOPs per pass (enc+dec+task model)", f"{cert['flops_per_pass']:.3e}"],
            ["CertCost FLOPs per certified sample", f"{cert['flops_per_sample']:.3e}"],
            ["CertCost wall-clock per sample (this CPU, batched)",
             pm([adv[s][t]['cert']['wall_s_per_sample'] * 1e3 for s in adv for t in ('prop', 'adv')], 2) + " ms"],
            ["Abstain rate (std / PGD-trained)", f"{pm([adv[s]['prop']['cert']['abstain_rate'] for s in adv])} / "
                                                 f"{pm([adv[s]['adv']['cert']['abstain_rate'] for s in adv])}"],
            ["Mean radius over certified-correct samples (PGD-trained)",
             pm([adv[s]['adv']['cert']['mean_radius_certified_correct'] for s in adv])],
            ["TSR of smoothed PGD-trained chain, AWGN 10 dB (100 votes)",
             pm([adv[s]['adv']['TSR_smoothed_awgn10'] for s in adv])]])
        S.update(arr_std_k32=round(ms(advv("prop", "ARR"))[0], 3), arr_adv_k32=round(ms(advv("adv", "ARR"))[0], 3),
                 arr_deepjscc_k32=round(ms(advv("b1", "ARR"))[0], 3),
                 eps_thr_std_k32=round(ms(advv("prop", "ARR"))[0], 3),
                 eps_thr_adv_k32=round(ms(advv("adv", "ARR"))[0], 3),
                 eps_thr_deepjscc_k32=round(ms(advv("b1", "ARR"))[0], 3),
                 eps_thr_adv_smoothed_k32=round(ms(sm_thr)[0], 4),
                 sasr_std_k32_8_255=round(ms(advv("prop", "SASR_8_255"))[0], 3),
                 sasr_adv_k32_8_255=round(ms(advv("adv", "SASR_8_255"))[0], 3),
                 sasr_deepjscc_k32_8_255=round(ms(advv("b1", "SASR_8_255"))[0], 3),
                 sasr_adv_smoothed_k32_8_255=round(ms([adv[s]["adv"]["SASR_smoothed"][eps.index(8 / 255)] for s in adv])[0], 3),
                 msd_std_k32=round(ms(advv("prop", "MSD_norm"))[0], 3),
                 msd_adv_k32=round(ms(advv("adv", "MSD_norm"))[0], 3),
                 cert_radius_smooth_k32=round(ms([adv[s]["adv"]["cert"]["ACR"] for s in adv])[0], 3),
                 cert_radius_std_k32=round(ms([adv[s]["prop"]["cert"]["ACR"] for s in adv])[0], 3),
                 certcost_flops_per_sample=float(f"{cert['flops_per_sample']:.3g}"),
                 certcost_ms_per_sample=round(ms([adv[s]['adv']['cert']['wall_s_per_sample'] * 1e3 for s in adv])[0], 2),
                 tsr_adv_smoothed_awgn_k32_10db=round(ms([adv[s]['adv']['TSR_smoothed_awgn10'] for s in adv])[0], 3))

    # ---- comparison / bit accounting table (Table 11 replacement)
    def tsr_at(rows, snr, **kw):
        return ms([r["TSR"] for r in sel(rows, snr=snr, **kw)])

    acc_rows = []
    prop_t = {s_: ms(M(main, "TSR", scheme="prop", k=32, channel="AWGN", snr=s_))[0] for s_ in (0.0, 10.0, 20.0)}

    def line(name, cu_real, bpd, payload, mod, rate, tsr3, prov, arr="—", sasr="—"):
        acc_rows.append([name, f"{cu_real} / {cu_real // 2}", bpd, payload, mod, rate] +
                        [f3(t) for t in tsr3] + [arr, sasr, prov])

    def cell_adv(tag, key, d=3):
        return pm(advv(tag, key), d) if adv else "—"

    line("Proposed analog, k=32 (standard)", 32, "unquantized", "n/a (analog)", "analog (I/Q)", "—",
         [prop_t[s_] for s_ in (0.0, 10.0, 20.0)], "Simulated", cell_adv("prop", "ARR", 4), cell_adv("prop", "SASR_8_255"))
    line("Proposed analog, k=32, PGD-trained", 32, "unquantized", "n/a (analog)", "analog (I/Q)", "—",
         [ms(M(main, "TSR", scheme="adv", k=32, channel="AWGN", snr=s_))[0] for s_ in (0.0, 10.0, 20.0)],
         "Simulated", cell_adv("adv", "ARR", 4), cell_adv("adv", "SASR_8_255"))
    line("B1 DeepJSCC-type analog JSCC, k=32 (λ_t=0)", 32, "unquantized", "n/a (analog)", "analog (I/Q)", "—",
         [ms(M(main, "TSR", scheme="b1", k=32, channel="AWGN", snr=s_))[0] for s_ in (0.0, 10.0, 20.0)],
         "Simulated", cell_adv("b1", "ARR", 4), cell_adv("b1", "SASR_8_255"))
    line("B1 DeepJSCC-type, k=64", 64, "unquantized", "n/a (analog)", "analog (I/Q)", "—",
         [ms(M(main, "TSR", scheme="b1", k=64, channel="AWGN", snr=s_))[0] for s_ in (0.0, 10.0, 20.0)], "Simulated")
    line("Proposed analog, k=64", 64, "unquantized", "n/a (analog)", "analog (I/Q)", "—",
         [ms(M(main, "TSR", scheme="prop", k=64, channel="AWGN", snr=s_))[0] for s_ in (0.0, 10.0, 20.0)], "Simulated")
    seen = set()
    for r in sorted(sel(dig, seed=seeds[0], channel="AWGN", snr=10.0), key=lambda r: (r["scheme"], r["n_sym"], r["rate"])):
        key = (r["scheme"], r["n_sym"], r["mod"], round(r["rate"], 3))
        if key in seen or r["n_sym"] > 32 and r["scheme"] == "B2":
            continue
        seen.add(key)
        rr = lambda s_: tsr_at(dig, s_, scheme=r["scheme"], n_sym=r["n_sym"], mod=r["mod"], channel="AWGN")[0] \
            if sel(dig, scheme=r["scheme"], n_sym=r["n_sym"], mod=r["mod"], channel="AWGN", snr=s_) else float("nan")
        same = [x for x in sel(dig, scheme=r["scheme"], n_sym=r["n_sym"], mod=r["mod"], channel="AWGN", snr=0.0)
                if abs(x["rate"] - r["rate"]) < 1e-6]
        t0 = ms([x["TSR"] for x in same])[0]
        t10 = ms([x["TSR"] for x in sel(dig, scheme=r["scheme"], n_sym=r["n_sym"], mod=r["mod"], channel="AWGN", snr=10.0) if abs(x["rate"] - r["rate"]) < 1e-6])[0]
        t20 = ms([x["TSR"] for x in sel(dig, scheme=r["scheme"], n_sym=r["n_sym"], mod=r["mod"], channel="AWGN", snr=20.0) if abs(x["rate"] - r["rate"]) < 1e-6])[0]
        nm = {"B2": f"B2 separate (PCA m={r['m']}, b={r['b']})", "B3": "B3 bit-exact (512 dims × 8 bits)",
              "DP-q4": "Digital proposed (k=32 latents × 4 bits)", "DP-q8": "Digital proposed (k=32 latents × 8 bits)"}[r["scheme"]]
        line(nm, 2 * r["n_sym"], r["b"], r["payload_bits"], r["mod"], f"{r['rate']:.3f}", [t0, t10, t20],
             "Simulated (Sionna 5G NR LDPC)")
    for n_sym in (16, 32):
        bb = lambda s_: ms([r["TSR"] for r in sel(bnd, n_sym=n_sym, channel="AWGN", snr=s_)])[0]
        Ks = [r["K_bits"] for r in sel(bnd, n_sym=n_sym, channel="AWGN", snr=10.0, seed=seeds[0])]
        line(f"Analytical bound: capacity-achieving code + PCA quantizer, {n_sym} c.u.", 2 * n_sym, "best b",
             f"⌊{n_sym}·log2(1+SNR)⌋ ({Ks[0] if Ks else '?'} at 10 dB)", "Gaussian (ideal)", "C",
             [bb(0.0), bb(10.0), bb(20.0)], "Analytical (source part simulated)")
    T["t11"] = md_table(["Scheme", "Channel uses / sample (real dims / complex symbols)", "Quant. bits per dim",
                         "Payload bits / sample", "Modulation", "Code rate", "TSR @0 dB", "TSR @10 dB", "TSR @20 dB",
                         "ARR (l∞)", "SASR(8/255)", "Provenance"], acc_rows)

    # ---- matched-TSR savings at AWGN 10 dB
    target = prop_t[10.0]
    sav = []
    # B2: smallest channel-use budget whose best MCS reaches the target (seed-mean)
    b2_ok = []
    for n_sym in sorted({r["n_sym"] for r in sel(dig, scheme="B2")}):
        best, rr = b2_best(dig, n_sym, "AWGN", 10.0)
        if best:
            t = ms([r["TSR"] for r in rr])[0]
            b2_ok.append((n_sym, best, t, rr[0]["payload_bits"]))
    reach = [x for x in b2_ok if x[2] >= target]
    curve = {int(k): v for k, v in G[seeds[0]].get("source_curve", {}).items()}
    curves_all = [{int(k): v for k, v in G[s]["source_curve"].items()} for s in seeds if "source_curve" in G[s]]
    Kmin = None
    for K in sorted(curve):
        if ms([c[K][0] for c in curves_all])[0] >= target:
            Kmin = K
            break
    best16, r16 = b2_best(dig, 16, "AWGN", 10.0)
    def verdict(t, cu, pay):
        if not (t >= target):
            return "does not reach target", "—"
        cu_txt = (f"{100 * (1 - 16 / cu):.1f}% fewer c.u. ({cu / 16:.0f}× fewer)" if cu > 16
                  else "none: baseline reaches the target with the same 16 c.u.")
        bit_txt = (f"DP-q4 uses {100 * (1 - 128 / pay):.1f}% fewer bits" if pay > 128
                   else f"baseline needs {128 / pay:.1f}× fewer bits than DP-q4")
        return cu_txt, bit_txt

    sav_rows = [["Target: proposed analog k=32 at AWGN 10 dB", f3(target), "32 / 16", "n/a (analog)", "—", "—"]]
    for n_sym, best, t, pay in b2_ok:
        sav_rows.append([f"B2 separate, best MCS at {n_sym} c.u. ({best[0]} R={best[1]:.2f}, PCA m={best[2]}, b={best[3]})",
                         f3(t), f"{2 * n_sym} / {n_sym}", pay, *verdict(t, n_sym, pay)])
    for mod in ("QPSK", "16QAM"):
        rr = sel(dig, scheme="B3", mod=mod, channel="AWGN", snr=10.0)
        if rr:
            t = ms([r["TSR"] for r in rr])[0]
            sav_rows.append([f"B3 bit-exact, {mod} R=1/2", f3(t), f"{2 * rr[0]['n_sym']} / {rr[0]['n_sym']}", 4096,
                             *verdict(t, rr[0]["n_sym"], 4096)])
    for b in (4, 8):
        for mod in ("QPSK", "16QAM"):
            rr = sel(dig, scheme=f"DP-q{b}", mod=mod, channel="AWGN", snr=10.0)
            if rr:
                t = ms([r["TSR"] for r in rr])[0]
                sav_rows.append([f"Digital proposed q{b}, {mod} R=1/2", f3(t), f"{2 * rr[0]['n_sym']} / {rr[0]['n_sym']}",
                                 rr[0]["payload_bits"], verdict(t, rr[0]["n_sym"], rr[0]["payload_bits"])[0], "—"])
    sav_rows.append(["Source coder alone (PCA + uniform quantizer, noiseless): smallest payload reaching the target",
                     f3(ms([c[Kmin][0] for c in curves_all])[0]) if Kmin else "not reached (K ≤ 300)",
                     "—", Kmin if Kmin else "> 300", "—", "—"])
    T["savings"] = md_table(["Scheme (AWGN, 10 dB)", "TSR", "Channel uses (real / complex)", "Payload bits",
                             "Channel-use saving of the proposed analog k=32 at matched TSR",
                             "Bit comparison with digital proposed q4 (128 bits)"], sav_rows)

    # ---- summary keys from the contract
    kw = dict(scheme="prop", k=32, channel="AWGN", snr=10.0)
    r0 = ref(main, **kw)
    S.update(
        tsr_awgn_k32_10db=round(ms(M(main, "TSR", **kw))[0], 3),
        tsr_awgn_k32_10db_std=round(ms(M(main, "TSR", **kw))[1], 3),
        tsr_awgn_k32_10db_ci_lo=round(r0["TSR_lo"], 3), tsr_awgn_k32_10db_ci_hi=round(r0["TSR_hi"], 3),
        tsr_awgn_k32_10db_seed0=round(r0["TSR"], 3),
        tsr_awgn_k32_0db=round(ms(M(main, "TSR", scheme="prop", k=32, channel="AWGN", snr=0.0))[0], 3),
        tsr_awgn_k32_20db=round(ms(M(main, "TSR", scheme="prop", k=32, channel="AWGN", snr=20.0))[0], 3),
        tsr_rayleigh_k32_10db=round(ms(M(main, "TSR", scheme="prop", k=32, channel="Rayleigh", snr=10.0))[0], 3),
        rse_disc_awgn_k32_10db=round(ms(M(main, "disc_RSE", **kw))[0], 3),
        nsmi_disc_awgn_k32_10db=round(ms(M(main, "disc_NSMI", **kw))[0], 3),
        rse_disc_awgn_k32_10db_ci_lo=round(r0["disc_RSE_ci"][0], 3), rse_disc_awgn_k32_10db_ci_hi=round(r0["disc_RSE_ci"][1], 3),
        rse_knn_awgn_k32_10db_mean=round(ms(M(main, "knn_RSE_knn", **kw))[0], 4),
        rse_knn_awgn_k32_10db_std=round(ms(M(main, "knn_RSE_knn", **kw))[1], 4),
        id_clf_awgn_k32_10db=round(ms(M(main, "ID", **kw))[0], 3),
        id_clf_marginal_awgn_k32_10db=round(ms(M(main, "ID_marginal", **kw))[0], 4),
        s3i_awgn_k32_10db=round(ms(M(main, "S3I", **kw))[0], 3),
        swd_awgn_k32_10db=round(ms(M(main, "SWD", **kw))[0], 2),
        tsr_deepjscc_awgn_k32_10db=round(ms(M(main, "TSR", scheme="b1", k=32, channel="AWGN", snr=10.0))[0], 3),
        tsr_sep_awgn_10db=round(ms([r["TSR"] for r in r16])[0], 3) if r16 else None,
        bits_sep=int(r16[0]["payload_bits"]) if r16 else None,
        sep_code_rate=round(best16[1], 3) if best16 else None, sep_pca_dims=best16[2] if best16 else None,
        sep_bits_per_coeff=best16[3] if best16 else None, sep_bits_per_qam_symbol=4 if best16 and best16[0] == "16QAM" else 2,
        bits_proposed_q8_k32=256, bits_proposed_q4_k32=128, bits_bitexact=4096,
        tsr_bitexact_awgn_10db=round(ms([r["TSR"] for r in sel(dig, scheme="B3", mod="QPSK", channel="AWGN", snr=10.0)])[0], 3),
        n_seeds=len(seeds), n_test=n_test,
        n_test_b3=G[seeds[0]]["digital"] and sel(dig, scheme="B3")[0]["N"],
        n_test_dp=G[seeds[0]]["digital"] and sel(dig, scheme="DP-q4")[0]["N"],
    )
    for b in (4, 8):
        rr = sel(dig, scheme=f"DP-q{b}", mod="QPSK", channel="AWGN", snr=10.0)
        if rr:
            S[f"tsr_dp_q{b}_qpsk_awgn_k32_10db"] = round(ms([r["TSR"] for r in rr])[0], 3)
    if Kmin:
        S["src_bits_to_match_proposed_10db"] = Kmin
    est = [G[s]["estimators"] for s in seeds if "estimators" in G[s]]
    if est:
        within = [np.std([r["RSE_knn"] for r in e["knn_within_seed"]], ddof=1) for e in est]
        izz = [np.mean(e["I_Z_Zhat_nats"]) for e in est]
        S.update(rse_knn_within_seed_std=round(float(np.mean(within)), 4),
                 i_z_zhat_knn_nats_mean=round(ms(izz)[0], 2), i_z_zhat_knn_nats_std=round(ms(izz)[1], 2),
                 awgn_capacity_nats_k32_10db=round(est[0]["awgn_capacity_nats_k32_10dB"], 1))
        sens = est[0]["sensitivity"]
        T["est"] = md_table(["Quantity", "Value"], [
            ["RSE_kNN, across seeds (mean ± std)", pm(M(main, "knn_RSE_knn", **kw), 4)],
            ["RSE_kNN, within seed: std over 20 sub-samples (mean over seeds)", f"{np.mean(within):.4f}"],
            ["RSE_kNN, std over 10 projection seeds (seed 0)", f"{np.std(est[0]['proj_seed_RSE'], ddof=1):.4f}"],
            ["k-NN MI Î(X;X̂) on the 16-dim projection (nats)", pm(M(main, "knn_MI_nats", **kw), 3)],
            ["KL entropy Ĥ(X) on the 16-dim projection (nats)", pm(M(main, "knn_Hx_nats", **kw), 2)],
            ["Kraskov Î(Z;Ẑ), k=32 latents, N=500, k=3 (nats)", pm(izz, 2)],
            ["AWGN capacity of the 16 complex uses at 10 dB (nats)", f"{est[0]['awgn_capacity_nats_k32_10dB']:.1f}"],
            ["Discrete RSE: bootstrap 95% CI (seed 0)", f"[{r0['disc_RSE_ci'][0]:.3f}, {r0['disc_RSE_ci'][1]:.3f}]"],
            ["Discrete RSE across seeds", pm(M(main, "disc_RSE", **kw))]])
        T["est_sens"] = md_table(["d_proj", "N", "RSE_kNN (seed 0, same data)"],
                                 [[s_["dproj"], s_["N"], f"{s_['RSE_knn']:.4f}"] for s_ in sens])
    # range check of the discrete metrics over every evaluated point (incl. bootstrap replicates)
    allv = [r[k_] for r in main + cmain for k_ in ("disc_RSE", "disc_NSMI") if k_ in r]
    allb = [r[k_] for r in main + cmain for k_ in ("disc_RSE_boot_min", "disc_RSE_boot_max", "disc_NSMI_boot_min",
                                                    "disc_NSMI_boot_max") if k_ in r]
    allv += [r[k_] for r in dig + cdig for k_ in ("disc_RSE", "disc_NSMI")]
    S["disc_range_min"] = round(float(min(allv + allb)), 4)
    S["disc_range_max"] = round(float(max(allv + allb)), 4)
    S["disc_range_points"] = len(allv)
    raw_mm = [r["disc_I_bits_MM_untruncated"] for r in main + cmain if "disc_I_bits_MM_untruncated" in r]
    S["disc_mm_truncated_points"] = int(sum(1 for v in raw_mm if v < 0))
    pl = [r[k_] for r in main + cmain for k_ in ("disc_RSE_plugin", "disc_NSMI_plugin") if k_ in r]
    S["disc_plugin_range_min"] = round(float(min(pl)), 4) if pl else None
    S["disc_plugin_range_max"] = round(float(max(pl)), 4) if pl else None
    knn_all = [r["knn_RSE_knn"] for r in main if "knn_RSE_knn" in r]
    S["knn_rse_min"] = round(float(min(knn_all)), 4)
    S["knn_rse_max"] = round(float(max(knn_all)), 4)

    # ---- CLIP
    if CL:
        ck = sorted({r["k"] for r in cmain if r["scheme"] == "prop"})
        csnr = sorted({r["snr"] for r in cmain})
        for ch in ["AWGN", "Rayleigh"]:
            rows = []
            for snr in csnr:
                rows.append([f"{snr:+.1f}"] + [pm(M(cmain, "TSR", scheme="prop", k=k, channel=ch, snr=snr)) for k in ck])
            T[f"clip_{ch}"] = md_table(["SNR (dB)"] + [f"k={k}" for k in ck], rows)
        rows = []
        for ch in ["AWGN", "Rayleigh"]:
            for snr in (0.0, 10.0, 20.0):
                b16, rr16 = b2_best(cdig, 16, ch, snr)
                rows.append([ch, int(snr), pm(M(cmain, "TSR", scheme="prop", k=32, channel=ch, snr=snr)),
                             pm(M(cmain, "TSR", scheme="b1", k=32, channel=ch, snr=snr)),
                             (pm([r["TSR"] for r in rr16]) + f" ({b16[0]} R={b16[1]:.2f}, m={b16[2]}, b={b16[3]})") if b16 else "n/a",
                             (lambda b, rr: (pm([r["TSR"] for r in rr]) + f" ({b[0]} R={b[1]:.2f}, m={b[2]}, b={b[3]})") if b else "n/a")(*b2_best(cdig, 32, ch, snr)),
                             pm([r["TSR"] for r in sel(cdig, scheme="DP-q4", channel=ch, snr=snr)]),
                             pm([r["TSR"] for r in sel(cbnd, n_sym=16, channel=ch, snr=snr)]),
                             pm(M(cmain, "disc_RSE", scheme="prop", k=32, channel=ch, snr=snr)),
                             pm(M(cmain, "S3I", scheme="prop", k=32, channel=ch, snr=snr)),
                             pm(M(cmain, "ID", scheme="prop", k=32, channel=ch, snr=snr))])
        T["clip_cmp"] = md_table(["Channel", "SNR (dB)", "Proposed k=32", "B1 DeepJSCC-type k=32",
                                  "B2 separate 16 c.u. (best MCS)", "B2 separate 32 c.u. (best MCS)",
                                  "DP-q4 (128 bits, QPSK R=1/2, 128 c.u.)", "Analytical bound 16 c.u.", "RSE (discrete), proposed", "S³I, proposed",
                                  "ID (nats), proposed"], rows)
        S.update(tsr_clip_awgn_k32_10db=round(ms(M(cmain, "TSR", scheme="prop", k=32, channel="AWGN", snr=10.0))[0], 3),
                 tsr_clip_awgn_k32_0db=round(ms(M(cmain, "TSR", scheme="prop", k=32, channel="AWGN", snr=0.0))[0], 3),
                 tsr_clip_rayleigh_k32_10db=round(ms(M(cmain, "TSR", scheme="prop", k=32, channel="Rayleigh", snr=10.0))[0], 3),
                 tsr_clip_deepjscc_awgn_k32_10db=round(ms(M(cmain, "TSR", scheme="b1", k=32, channel="AWGN", snr=10.0))[0], 3),
                 s3i_clip_awgn_k32_10db=round(ms(M(cmain, "S3I", scheme="prop", k=32, channel="AWGN", snr=10.0))[0], 3),
                 rse_disc_clip_awgn_k32_10db=round(ms(M(cmain, "disc_RSE", scheme="prop", k=32, channel="AWGN", snr=10.0))[0], 3),
                 clip_oracle_clean_acc=round(ms([CL[s]["info"]["oracle_test_acc"] for s in CL])[0], 3),
                 clip_n_test=CL[min(CL)]["info"]["n_test"], clip_n_train=CL[min(CL)]["info"]["n_train"])
        b16, rr16 = b2_best(cdig, 16, "AWGN", 10.0)
        if rr16:
            S["tsr_clip_sep_awgn_10db"] = round(ms([r["TSR"] for r in rr16])[0], 3)

    # ---- UAV
    if uav:
        rows = []
        for th in sorted({r["theta"] for r in uav}):
            cells = []
            for snr in (10.0, 20.0):
                rr = sel(uav, theta=th, lam=0.1, snr_nom=snr)
                cells += [pm([r["TSR"] for r in rr]), pm([r["ID"] for r in rr]), pm([r["sess_outage"] for r in rr])]
            r0u = sel(uav, theta=th, lam=0.1, snr_nom=10.0)[0]
            rows.append([th, f"{r0u['P_LoS']:.3f}", f"{r0u['K_dB']:.1f}"] + cells)
        T["uav_theta"] = md_table(["Elevation θ (deg)", "P_LoS(θ)", "K(θ) (dB)", "TSR (10 dB)", "ID (10 dB)",
                                   "Session outage (10 dB)", "TSR (20 dB)", "ID (20 dB)", "Session outage (20 dB)"], rows)
        rows = []
        for th in (30, 60):
            for lam in sorted({r["lam"] for r in uav if r["theta"] == th}):
                rr = sel(uav, theta=th, lam=lam, snr_nom=20.0)
                if not rr:
                    continue
                rows.append([th, lam, pm([r["TSR"] for r in rr]), pm([r["sess_TSR_p5"] for r in rr]),
                             pm([r["sess_TSR_std"] for r in rr]), pm([r["sess_outage"] for r in rr]),
                             pm([r["ID"] for r in rr]), pm([r["ID_session_marginal_mean"] for r in rr])])
        T["uav_lam"] = md_table(["θ (deg)", "Transition rate λ", "TSR", "Session TSR 5th pct.", "Std of session TSR",
                                 "Session outage", "ID per message (nats)", "Session-level ID (marginal KL, nats)"], rows)
        for th in (30, 90):
            rr = sel(uav, theta=th, lam=0.1, snr_nom=20.0)
            S[f"tsr_uav_theta{th}_20db"] = round(ms([r["TSR"] for r in rr])[0], 3)
            rr = sel(uav, theta=th, lam=0.1, snr_nom=10.0)
            S[f"tsr_uav_theta{th}_10db"] = round(ms([r["TSR"] for r in rr])[0], 3)

    # ---- Proposition 1
    if p1:
        rows = []
        for ch in ("AWGN", "Rayleigh"):
            for N in (2, 4, 8):
                rr = sel(p1, channel=ch, N=N)
                rows.append([ch, N, sum(r["trials"] for r in rr), sum(r["violations"] for r in rr),
                             f"{min(r['min_slack'] for r in rr):.2e}", pm([r["mean_slack"] for r in rr]),
                             pm([r["mean_SCI"] for r in rr]), pm([r["mean_bound"] for r in rr]),
                             pm([r["ratio_Dhat_over_boundgap"] for r in rr])])
        T["prop1"] = md_table(["Channel", "N receivers", "Trials", "Violations", "Min slack", "Mean slack",
                               "Mean SCI", "Mean bound", "D̂ / bound gap"], rows)
        S["prop1_violations"] = int(sum(r["violations"] for r in p1))
        S["prop1_trials"] = int(sum(r["trials"] for r in p1))
        S["prop1_min_slack"] = float(f"{min(r['min_slack'] for r in p1):.2e}")
        S["prop1_ratio_mean"] = round(ms([r["ratio_Dhat_over_boundgap"] for r in p1])[0], 2)

    # ---- runtimes
    rt = []
    for s in seeds:
        tm = G[s]["timing"]
        rt.append([s, "gauss", f"{sum(tm['train_s'].values()):.0f}"] +
                  [f"{tm.get(p, float('nan')):.0f}" for p in ("main", "est", "digital", "adv", "uav", "prop1")] +
                  [f"{tm['total_s']:.0f}"])
    for s in sorted(CL):
        tm = CL[s]["timing"]
        rt.append([s, "clip", f"{sum(tm['train_s'].values()):.0f}", f"{tm.get('main', 0):.0f}", "—",
                   f"{tm.get('digital', 0):.0f}", "—", "—", "—", f"{tm['total_s']:.0f}"])
    T["runtime"] = md_table(["Seed", "Dataset", "Training (s)", "Main sweep", "Estimators", "Digital (LDPC)",
                             "Adversarial", "UAV", "Prop. 1", "Total (s)"], rt)

    # ---- old vs new
    v1p = os.path.join(res_dir, "v1_protocol_check.json")
    if os.path.exists(v1p):
        v1 = json.load(open(v1p))
        S.update(v1code_tsr_k32_10db_in_sample=round(v1["tsr_k32_awgn_10db_in_sample"], 3),
                 v1code_tsr_k32_10db_fresh=round(v1["tsr_k32_awgn_10db_fresh"], 3),
                 v1code_oracle_acc_in_sample=round(v1["oracle_acc_in_sample"], 3),
                 v1code_oracle_acc_fresh=round(v1["oracle_acc_fresh"], 3))
    S.update(subset_reference(seeds))
    with open(os.path.join(res_dir, "summary.json"), "w") as f:
        json.dump(S, f, indent=1, default=float)
    with open(os.path.join(res_dir, "tables.md"), "w", encoding="utf-8") as f:
        for k, v in T.items():
            f.write(f"## {k}\n\n{v}\n\n")
    if not quick:
        render_results(T, S)
    if not quick or True:
        figures(G, CL, main, dig, bnd, cmain, cdig, cbnd, uav, adv, eps, seeds, ks, chans, snrs, quick)
    print("summary keys:", len(S))
    return S, T


def _fmt(key, v):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if "flops" in key or "min_slack" in key:
        return f"{v:.3g}".replace("e+0", "e").replace("e-0", "e-")
    d = 3
    if any(t in key for t in ("knn", "marginal", "smoothed_k32", "disc_range", "disc_plugin")):
        d = 4
    if any(t in key for t in ("i_z_zhat", "swd", "_ms_", "ratio")):
        d = 2
    if "capacity" in key:
        d = 1
    return f"{v:.{d}f}"


def render_results(T, S):
    import re
    src = os.path.join(HERE, "RESULTS.src.md")
    if not os.path.exists(src):
        return
    txt = open(src, encoding="utf-8").read()
    missing = []

    def rep(m):
        kind, key = m.group(1), m.group(2)
        if kind == "T":
            if key in T:
                return T[key]
        elif key in S:
            return _fmt(key, S[key])
        missing.append(m.group(0))
        return m.group(0)
    txt = re.sub(r"<!-- RESULTS.md is GENERATED.*?-->", "<!-- GENERATED from simulations/RESULTS.src.md by run_all.py --report-only; do not edit by hand. -->", txt, flags=re.S)
    out = re.sub(r"[{][{]([TS]):([A-Za-z0-9_]+)[}][}]", rep, txt)
    with open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(out)
    if missing:
        print("RESULTS.md: unresolved tokens", sorted(set(missing)))


def subset_reference(seeds):
    """Clean task-model accuracy on the test subsets used by B3 (first 500) and DP (first 2000)."""
    try:
        import torch
        from .campaign import FULL, get_oracle
        from .common import gaussian_dataset, clip_dataset
        from .campaign import CLIP_PATH
    except Exception:
        return {}
    a500, a2000, c2000 = [], [], []
    for s in seeds:
        ds = gaussian_dataset(s, FULL["n_train"], FULL["n_test"])
        clf, _ = get_oracle(ds, s, FULL, False)
        with torch.no_grad():
            p = clf(ds.x_test).argmax(1)
        a500.append(float((p[:500] == ds.y_test[:500]).float().mean()))
        a2000.append(float((p[:2000] == ds.y_test[:2000]).float().mean()))
        if os.path.exists(CLIP_PATH):
            dc = clip_dataset(CLIP_PATH)
            cc, _ = get_oracle(dc, s, FULL, False)
            with torch.no_grad():
                pc = cc(dc.x_test[:2000]).argmax(1)
            c2000.append(float((pc == dc.y_test[:2000]).float().mean()))
    out = {"oracle_acc_first500": round(ms(a500)[0], 3), "oracle_acc_first2000": round(ms(a2000)[0], 3)}
    if c2000:
        out["clip_oracle_acc_first2000"] = round(ms(c2000)[0], 3)
    return out


# ---------------------------------------------------------------------------- figures
def figures(G, CL, main, dig, bnd, cmain, cdig, cbnd, uav, adv, eps, seeds, ks, chans, snrs, quick):
    plt = _plt()
    sfx = "_quick" if quick else ""

    def curve(rows, key, xs, xkey, **kw):
        m, s = [], []
        for x in xs:
            a = ms([r[key] for r in sel(rows, **{xkey: x}, **kw)])
            m.append(a[0]); s.append(a[1])
        return np.array(m), np.array(s)

    # 1 fidelity vs k
    panels = [("Gaussian proxy", main)] + ([("CLIP ViT-B/32, CIFAR-10", cmain)] if cmain else [])
    fig, axs = plt.subplots(1, len(panels), figsize=(6.3 if len(panels) == 2 else 3.4, 2.7), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (title, rows) in zip(axs, panels):
        kk = sorted({r["k"] for r in rows if r["scheme"] == "prop"})
        for key, lab, col, mk in [("TSR", "TSR", C["prop"], "o"), ("disc_RSE", "RSE (discrete)", C["b1"], "s"),
                                  ("disc_NSMI", "NSMI (discrete)", C["b2"], "^"), ("S3I", "S³I", C["adv"], "D")]:
            m, s = curve(rows, key, kk, "k", scheme="prop", channel="AWGN", snr=10.0)
            ax.errorbar(kk, m, yerr=s, color=col, marker=mk, label=lab, capsize=2, lw=1.5)
        ax.set_xscale("log", base=2); ax.set_xticks(kk); ax.set_xticklabels([str(k) for k in kk])
        ax.set_xlabel("Bottleneck dimension $k$ (real channel uses)"); ax.set_title(title)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("Metric value (AWGN, SNR = 10 dB)")
    axs[0].legend(loc="upper left", ncol=2)
    _save(fig, f"fig_fidelity_vs_k{sfx}.png")

    # 2 TSR vs SNR with baselines
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, ch in zip(axs, ["AWGN", "Rayleigh"]):
        xs = snrs
        m, s = curve(main, "TSR", xs, "snr", scheme="prop", k=32, channel=ch)
        ax.errorbar(xs, m, s, color=C["prop"], marker="o", label="Proposed analog, $k$=32", capsize=2)
        m, s = curve(main, "TSR", xs, "snr", scheme="b1", k=32, channel=ch)
        ax.errorbar(xs, m, s, color=C["b1"], marker="s", label="B1 DeepJSCC-type, $k$=32", capsize=2)
        # B2 envelope over MCS at 16 c.u.
        env = []
        for x in xs:
            best = 0
            for key in {(r["mod"], round(r["rate"], 3)) for r in sel(dig, scheme="B2", n_sym=16, channel=ch, snr=x)}:
                best = max(best, ms([r["TSR"] for r in sel(dig, scheme="B2", n_sym=16, channel=ch, snr=x) if (r["mod"], round(r["rate"], 3)) == key])[0])
            env.append(best)
        ax.plot(xs, env, color=C["b2"], marker="^", label="B2 separate, 16 c.u. (best MCS)")
        m, s = curve([r for r in dig if r["scheme"] == "DP-q4" and r["mod"] == "QPSK"], "TSR", xs, "snr", channel=ch)
        ax.errorbar(xs, m, s, color=C["dp"], marker="v", label="Digital proposed q4, 128 c.u.", capsize=2)
        m, s = curve([r for r in dig if r["scheme"] == "B3" and r["mod"] == "QPSK"], "TSR", xs, "snr", channel=ch)
        ax.errorbar(xs, m, s, color=C["b3"], marker="D", label="B3 bit-exact, 4096 c.u.", capsize=2)
        m, s = curve(bnd, "TSR", xs, "snr", n_sym=16, channel=ch)
        ax.plot(xs, m, color=C["bound"], ls="--", lw=1.2, label="Analytical bound, 16 c.u.")
        oc = ms([G[s_]["info"]["oracle_test_acc"] for s_ in seeds])[0]
        ax.axhline(oc, color=INK2, lw=0.9, ls=":", label="Task model on clean input")
        ax.set_title(ch); ax.set_xlabel("SNR per channel use (dB)")
    axs[0].set_ylabel("TSR (Gaussian proxy)"); axs[0].set_ylim(0, 0.75)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    _save(fig, f"fig_tsr_vs_snr{sfx}.png")

    # 3 TSR vs channel uses / payload bits (AWGN 10 dB)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axs[0]
    for sch, col, mk, lab in [("prop", C["prop"], "o", "Proposed analog"), ("b1", C["b1"], "s", "B1 DeepJSCC-type")]:
        kk = sorted({r["k"] for r in main if r["scheme"] == sch})
        m, s = curve(main, "TSR", kk, "k", scheme=sch, channel="AWGN", snr=10.0)
        ax.errorbar([k // 2 for k in kk], m, s, color=col, marker=mk, label=lab, capsize=2)
    pts = defaultdict(list)
    for r in sel(dig, channel="AWGN", snr=10.0):
        pts[(r["scheme"], r["n_sym"], r["mod"], round(r["rate"], 3))].append(r["TSR"])
    for sch, col, mk, lab in [("B2", C["b2"], "^", "B2 separate (all MCS)"), ("DP-q4", C["dp"], "v", "Digital proposed q4"),
                              ("DP-q8", C["dp8"], "P", "Digital proposed q8"), ("B3", C["b3"], "D", "B3 bit-exact")]:
        xs_ = [(k_[1], np.mean(v)) for k_, v in pts.items() if k_[0] == sch]
        if xs_:
            ax.scatter([a for a, _ in xs_], [b for _, b in xs_], color=col, marker=mk, s=18, label=lab, zorder=3)
    ax.set_xscale("log", base=2); ax.set_xlabel("Complex channel uses per sample")
    ax.set_ylabel("TSR (AWGN, 10 dB)"); ax.set_title("(a) TSR vs channel uses")
    ax.legend(loc="lower right", fontsize=6.3)
    ax = axs[1]
    cv = [{int(k): v for k, v in G[s]["source_curve"].items()} for s in seeds if "source_curve" in G[s]]
    if cv:
        Ks = sorted(cv[0])
        ax.plot(Ks, [np.mean([c[K][0] for c in cv]) for K in Ks], color=C["b2"], lw=1.4,
                label="PCA + uniform quantizer (noiseless)")
    for sch, col, mk, lab in [("DP-q4", C["dp"], "v", "Digital proposed q4 (noiseless)"),
                              ("DP-q8", C["dp8"], "P", "Digital proposed q8 (noiseless)"),
                              ("B3", C["b3"], "D", "B3 8-bit full embedding (noiseless)")]:
        rr = sel(dig, scheme=sch, channel="AWGN", snr=max(snrs))
        if rr:
            ax.scatter([rr[0]["payload_bits"]], [np.mean([r["noiseless_TSR"] for r in rr])], color=col, marker=mk, s=22, label=lab, zorder=3)
    tgt = ms([r["TSR"] for r in sel(main, scheme="prop", k=32, channel="AWGN", snr=10.0)])[0]
    ax.axhline(tgt, color=C["prop"], ls="--", lw=1)
    ax.text(1.2, tgt - 0.035, "proposed analog $k$=32,"+chr(10)+"AWGN 10 dB (16 c.u.)", color=INK2, fontsize=6.5, va="top")
    ax.set_xscale("log", base=2); ax.set_xlabel("Payload bits per sample"); ax.set_title("(b) TSR vs payload bits")
    ax.legend(loc="lower right", fontsize=6.3)
    _save(fig, f"fig_tsr_vs_bits{sfx}.png")

    # 4 CLIP SNR x k
    if cmain:
        ck = sorted({r["k"] for r in cmain if r["scheme"] == "prop"})
        csn = sorted({r["snr"] for r in cmain})
        seq = ["#a8c8f0", "#6fa3e3", "#2a78d6", "#1c5aa8", "#0f3a73"]  # one hue, light->dark
        fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
        for ax, ch in zip(axs, ["AWGN", "Rayleigh"]):
            for i, k in enumerate(ck):
                m, s = curve(cmain, "TSR", csn, "snr", scheme="prop", k=k, channel=ch)
                ax.errorbar(csn, m, s, color=seq[i % 5], marker="o", ms=3, label=f"$k$={k}", capsize=1.5)
            env = []
            for x in csn:
                best = 0
                for key in {(r["mod"], round(r["rate"], 3)) for r in sel(cdig, scheme="B2", n_sym=16, channel=ch, snr=x)}:
                    best = max(best, ms([r["TSR"] for r in sel(cdig, scheme="B2", n_sym=16, channel=ch, snr=x) if (r["mod"], round(r["rate"], 3)) == key])[0])
                env.append(best)
            ax.plot(csn, env, color=C["b2"], ls="--", marker="^", ms=3, label="B2 separate, 16 c.u.")
            ax.set_title(f"CLIP ViT-B/32 embeddings, {ch}"); ax.set_xlabel("SNR per channel use (dB)")
        axs[0].set_ylabel("TSR"); axs[0].set_ylim(0, 1)
        axs[0].legend(loc="lower right", ncol=2)
        _save(fig, f"fig_clip_snr_k{sfx}.png")

    # 5 UAV
    if uav:
        fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
        ths = sorted({r["theta"] for r in uav})
        for snr, col, mk in [(10.0, C["prop"], "o"), (20.0, C["b1"], "s")]:
            m, s = curve(uav, "TSR", ths, "theta", lam=0.1, snr_nom=snr)
            axs[0].errorbar(ths, m, s, color=col, marker=mk, capsize=2, label=f"TSR, nominal SNR {snr:.0f} dB")
        plos = [sel(uav, theta=t, lam=0.1, snr_nom=10.0)[0]["P_LoS"] for t in ths]
        axs[0].plot(ths, np.array(plos) * ms([r["clean_acc"] for r in uav])[0], color=INK2, ls=":", lw=1,
                    label="$P_{LoS}(\\theta)$ × clean accuracy")
        axs[0].set_xlabel("Elevation angle θ (deg)"); axs[0].set_ylabel("TSR"); axs[0].set_title("(a) TSR vs elevation (λ = 0.1)")
        axs[0].legend(loc="lower right", fontsize=6.5); axs[0].set_ylim(0, 0.7)
        for th, col, mk in [(30, C["b2"], "^"), (60, C["adv"], "D")]:
            lams = sorted({r["lam"] for r in uav if r["theta"] == th})
            m, s = curve(uav, "sess_outage", lams, "lam", theta=th, snr_nom=20.0)
            axs[1].errorbar(lams, m, s, color=col, marker=mk, capsize=2, label=f"session outage, θ={th}°")
            m, s = curve(uav, "TSR", lams, "lam", theta=th, snr_nom=20.0)
            axs[1].errorbar(lams, m, s, color=col, marker=mk, ls="--", mfc="white", capsize=2, label=f"mean TSR, θ={th}°")
        axs[1].set_xscale("log"); axs[1].set_xlabel("Blockage transition rate λ (per message)")
        axs[1].set_ylabel("Outage probability / mean TSR"); axs[1].set_title("(b) Session statistics vs λ (20 dB)")
        axs[1].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); axs[1].set_ylim(0, 1)
        _save(fig, f"fig_uav{sfx}.png")

    # 6 SASR vs eps (replaces V1 Figure 4)
    if adv:
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        e255 = np.array(eps) * 255
        thr = {}
        for tag, col, mk, lab in [("prop", C["prop"], "o", "Proposed, standard training"),
                                  ("adv", C["adv"], "D", "Proposed, PGD-trained"),
                                  ("b1", C["b1"], "s", "B1 DeepJSCC-type")]:
            m = np.mean([adv[s][tag]["SASR"] for s in adv], 0); sd = np.std([adv[s][tag]["SASR"] for s in adv], 0, ddof=1) if len(adv) > 1 else 0 * m
            ax.errorbar(e255, m, sd, color=col, marker=mk, capsize=2, label=lab)
            thr[tag] = (ms([adv[s][tag]["ARR"] for s in adv])[0] * 255, col)
        m = np.mean([adv[s]["adv"]["SASR_smoothed"] for s in adv], 0)
        sd = np.std([adv[s]["adv"]["SASR_smoothed"] for s in adv], 0, ddof=1) if len(adv) > 1 else 0 * m
        ax.errorbar(e255, m, sd, color=C["smooth"], marker="^", ls="--", capsize=2,
                    label="PGD-trained + smoothing (EOT attack)")
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        for tag, (x, col) in thr.items():
            ax.plot([x, x], [0, 0.5], color=col, lw=1.0, ls=(0, (2, 2)))
            ax.plot([x], [0.5], marker="o", ms=5, mfc="white", mec=col, mew=1.5, zorder=4)
        ax.add_patch(plt.Rectangle((7.4, 0), 1.2, 0.30, color="#1baf7a", alpha=0.18, lw=0))
        ax.text(8.9, 0.02, "Test Case 3 pass:\nSASR ≤ 0.30 at ε=8/255", fontsize=6.2, color=INK2)
        ax.text(e255[-1], 0.51, "ε$_{threshold}$ (SASR = 0.5)", fontsize=6.2, color=INK2, ha="right", va="bottom")
        ax.set_xlabel("PGD budget ε (×1/255, $l_\\infty$)"); ax.set_ylabel("SASR")
        ax.set_ylim(0, 1.02); ax.set_xlim(-0.5, e255[-1] + 0.5)
        ax.legend(fontsize=6.2, loc="lower right", bbox_to_anchor=(1.0, 0.13))
        _save(fig, f"fig_sasr{sfx}.png")
