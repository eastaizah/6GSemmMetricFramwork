#!/usr/bin/env python3
"""Round-1 revision simulation campaign (single entry point).

Run with the Sionna 2.2 venv interpreter (5G NR LDPC is needed by the digital
baselines); CLIP embeddings must have been cached first with the system Python
(``python simulations/clip_embed.py``).

    PY=../Tutoriales/Sabatico/build/venv_sionna/Scripts/python.exe
    $PY simulations/run_all.py --quick                 # smoke run (~5 min, results_quick/)
    $PY simulations/run_all.py                         # full run: 5 seeds in parallel
    $PY simulations/run_all.py --report-only           # re-aggregate tables/figures/RESULTS data

Options: --seeds 0,1,2,3,4  --jobs 5  --parts main,est,digital,adv,uav,prop1,clip
Per-seed raw results go to ``results/raw/<dataset>_seed<s>.json``; aggregation writes
``results/*.csv``, ``results/summary.json`` and ``work/figs/fig_*.png``.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ALL_PARTS = "main,est,digital,adv,uav,prop1,clip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--parts", default=ALL_PARTS)
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    res_dir = os.path.join(HERE, "results_quick" if args.quick else "results")
    raw = os.path.join(res_dir, "raw")
    parts = args.parts.split(",")

    if args.worker is not None:
        import warnings
        warnings.filterwarnings("ignore")
        from semsim.campaign import run_seed
        run_seed(args.worker, args.quick, parts, raw)
        return

    seeds = [int(s) for s in (args.seeds or ("0,1" if args.quick else "0,1,2,3,4")).split(",")]
    if not args.report_only:
        os.makedirs(os.path.join(res_dir, "logs"), exist_ok=True)
        t0 = time.time()
        threads = max(1, (os.cpu_count() or 4) // max(1, min(args.jobs, len(seeds))))
        env = dict(os.environ, SIM_THREADS=str(threads), PYTHONWARNINGS="ignore")
        pending = list(seeds)
        running = []
        while pending or running:
            while pending and len(running) < args.jobs:
                s = pending.pop(0)
                lf = open(os.path.join(res_dir, "logs", f"seed{s}.log"), "w")
                cmd = [sys.executable, os.path.abspath(__file__), "--worker", str(s), "--parts", args.parts]
                if args.quick:
                    cmd.append("--quick")
                running.append((s, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
                print(f"started seed {s}", flush=True)
            for item in list(running):
                s, p, lf = item
                if p.poll() is not None:
                    lf.close()
                    running.remove(item)
                    print(f"seed {s} finished rc={p.returncode} ({time.time() - t0:.0f}s)", flush=True)
                    if p.returncode != 0:
                        print(open(lf.name).read()[-3000:])
                        sys.exit(f"seed {s} failed")
            time.sleep(2)
        with open(os.path.join(res_dir, "logs", "wallclock.txt"), "w") as f:
            f.write(f"seeds={seeds} jobs={args.jobs} threads/job={threads} wall_s={time.time() - t0:.0f}\n")
    from semsim.report import build_all
    build_all(res_dir, quick=args.quick)


if __name__ == "__main__":
    main()
