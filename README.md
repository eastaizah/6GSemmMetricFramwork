# A Multi-Dimensional Semantic Metric Standardization Framework for Evaluating AI-Native Systems in 6G Networks

One seed-controlled campaign produces every number of the revised Section 11.
The authors' original code is kept unchanged in `original/` for reference; the
revision code lives in `semsim/` and is driven by `run_all.py`.

## Layout

| Path | Content |
|---|---|
| `clip_embed.py` | Extracts CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) image embeddings of real CIFAR-10 images (20,000 train = 2,000/class, 10,000 test) into `cache/clip_vitb32_cifar10.npz` (git-ignored). Needs `transformers` (system Python). |
| `run_all.py` | Entry point (runs the seeds in parallel, then aggregates). Must run with the Sionna 2.2 venv Python (5G NR LDPC). |
| `semsim/common.py` | Data (Gaussian proxy and CLIP), networks, training, channel models, analog I/Q link. |
| `semsim/metrics.py` | TSR + Wilson CI, discrete RSE/NSMI (Miller–Madow, bootstrap), k-NN RSE (KSG / Kozachenko–Leonenko), S³I, SWD (Sinkhorn), ID, ICC, AP, SU, SCI (TV). |
| `semsim/digital.py` | B2 separate source–channel coding, B3 bit-exact, digital variant of the proposed encoder (Sionna LDPC5GEncoder/Decoder + QAM mapper/APP demapper), analytical separation bound. |
| `semsim/adversarial.py` | PGD, ARR (binary search), SASR curve, MSD_norm, randomized smoothing CERTIFY (Cohen et al. 2019), EOT attack on the smoothed chain. |
| `semsim/uav.py` | Low-altitude A2G channel: Al-Hourani LoS probability, elevation-dependent Rician K, shadowed blocked state, two-state Markov blockage. |
| `semsim/campaign.py` | One seed: trains every model once (cached in `cache/models/`) and runs all parts. |
| `semsim/report.py` | Aggregation → `results/*.csv`, `results/tables.md`, `results/summary.json`, `../work/figs/fig_*.png`. |
| `v1_protocol_check.py` | Diagnostic that re-runs the ORIGINAL V1 code and evaluates its models in-sample (as V1) and on fresh samples. |
| `RESULTS.md` | Results, tables for the revised Section 11, provenance, changes vs V1. |

## Reproduce

```bash
cd MetricasSemanticas6G
# 1. CLIP embeddings (system Python with transformers; downloads the model from Hugging Face once)
python simulations/clip_embed.py
# 2. campaign (Sionna 2.2 venv; read-only use)
PY=../Tutoriales/Sabatico/build/venv_sionna/Scripts/python.exe
$PY simulations/run_all.py --quick        # smoke run: 2 seeds, reduced sizes -> simulations/results_quick/
$PY simulations/run_all.py                # full run: seeds 0-4, 5 parallel workers -> simulations/results/
$PY simulations/run_all.py --report-only  # re-aggregate without re-simulating
# 3. optional diagnostic of the V1 protocol (system Python)
python simulations/v1_protocol_check.py
```

`--seeds 0,1,2`, `--jobs N` and `--parts main,est,digital,adv,uav,prop1,clip` restrict a run.
Trained models are cached under `simulations/cache/models/` (git-ignored); delete that folder
to retrain from scratch.  Each seed controls data generation, network initialisation,
minibatch order and every channel/noise draw (per-point generators derived by hashing
`(seed, scheme, k, channel, SNR)`), so a rerun reproduces the JSON files bit-for-bit on
the same machine/library versions.

Software used: Python 3.13, PyTorch 2.14.1 (CPU), Sionna 2.2.0, NumPy 2.x, SciPy, matplotlib;
transformers 5.21 for the CLIP extraction only.  Full run wall-clock: see `RESULTS.md`
(Runtimes) and `results/logs/wallclock.txt`.
