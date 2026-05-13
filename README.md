# Multi-Dimensional Semantic Metric Standardization Framework

Python simulation suite for the paper:

> **"A Multi-Dimensional Semantic Metric Standardization Framework for Evaluating AI-Native Systems in 6G Networks"**
> IEEE — proposed submission

---

## The Proposed Framework

### Motivation

The transition to 6G introduces AI-native communication paradigms in which meaning transfer—not bit accuracy—is the primary design objective. Traditional metrics (BER, throughput, SINR) operate at Shannon's Level A (syntactic) and are fundamentally insufficient for characterizing semantic systems. No current 3GPP specification defines normative semantic performance metrics, creating a critical standardization gap that prevents interoperability and conformance testing of AI-native systems.

### Four-Dimensional Taxonomy

The framework organizes **16 formally defined metrics** into four orthogonal evaluation dimensions grounded in information theory, optimal transport, and game theory:

| Dimension | Metrics | Primary Applications |
|-----------|---------|----------------------|
| **Semantic Fidelity** | RSE, SWD, S³I, NSMI | Digital twins, teleoperation, high-fidelity sensory transmission |
| **Task Completion Accuracy** | TSR, AP, SU, CE | Robotic control, autonomous vehicles, industrial automation |
| **Intent Alignment** | ID, ICC, SCI, PF | Intent-driven network services, multi-agent coordination, HMI |
| **Resilience to Semantic Attacks** | ARR, SASR, CertCost, MSD | Critical infrastructure, defense systems, healthcare automation |

#### Dimension 1 — Semantic Fidelity

| Metric | Full Name | Definition |
|--------|-----------|------------|
| RSE | Relative Semantic Entropy | `RSE = I_s(X;Y;T) / H_s(X;T) ∈ [0,1]` — fraction of task-relevant semantic information preserved |
| SWD | Semantic Wasserstein Distance | Earth Mover Distance between transmitter and receiver embedding distributions, computed via Sinkhorn entropic regularization |
| S³I | Semantic Structural Similarity Index | SSIM adapted to the semantic embedding domain, aggregated over task-relevant regions with importance weights |
| NSMI | Normalized Semantic Mutual Information | `NSMI = I_s(X;Y;T) / sqrt(H_s(X;T)·H_s(Y;T)) ∈ [0,1]` — invariant to invertible affine transformations of the embedding space |

#### Dimension 2 — Task Completion Accuracy

| Metric | Full Name | Definition |
|--------|-----------|------------|
| TSR | Task Success Rate | Empirical probability of successful task execution given the received signal; includes Wilson score 95% CI |
| AP | Action Precision | `AP = 1 − d_A(a_exec, a_opt) / d_A_max ∈ [0,1]` |
| SU | Semantic Utility | Expected utility incorporating correctness, temporal value, and decision consequences |
| CE | Completion Efficiency | `CE = TSR / ρ` where `ρ = k/d` is the compression ratio |

#### Dimension 3 — Intent Alignment

| Metric | Full Name | Definition |
|--------|-----------|------------|
| ID | Intent Divergence | KL divergence between transmitter and receiver intent distributions |
| ICC | Intentional-Context Coherence | `ICC = tanh(log p(I_R|C) − log p(I_R)) ∈ (−1,1]` — context-conditioned intent consistency |
| SCI | Semantic Consensus Index | `SCI = 1 − (1/|R|²)Σ D(I_i,I_j) ∈ [0,1]` — inter-receiver semantic agreement in multicast scenarios |
| PF | Purpose Fidelity | Probability that the action induced by the decoded intent achieves the transmitter's purpose |

#### Dimension 4 — Resilience to Semantic Attacks

| Metric | Full Name | Definition |
|--------|-----------|------------|
| ARR | Adversarial Robustness Radius | Minimum perturbation magnitude required to alter semantic interpretation, found via binary-search PGD |
| SASR | Semantic Attack Success Rate | Empirical probability that a PGD adversary with budget ε = 8/255 corrupts semantic interpretation |
| CertCost | Certification Cost | Wall-clock time to certify robustness via Randomized Smoothing (Cohen et al., 2019) |
| MSD | Maximum Semantic Degradation | `MSD_norm = (L_adv − L_clean) / (L_max − L_clean) ∈ [0,1]` |

### Multi-Dimensional Aggregation

Normalized dimension scores M_i ∈ [0,1] are combined via weighted linear aggregation:

```
M_composite = Σ w_i · M_i,   Σ w_i = 1
```

For safety-critical applications with strict minimum requirements per dimension:

```
M_composite = min_i (M_i / M_i_threshold)
```

### 3GPP Standardization Pathway

The framework proposes a new **TS 39.xxx** specification series:

| Specification | Content |
|---------------|---------|
| TS 39.101 | General Aspects and Principles — definitions, reference architecture, 3GPP layer interfaces |
| TS 39.201 | Semantic Metrics Definition and Measurement — normative specification of all 16 metrics |
| TS 39.202 | Measurement Configuration and Reporting — signaling, report formats, SON/MDT integration |
| TS 39.521 | Conformance Testing — test cases and procedures |

Proposed timeline: Release 20 (2025–2026) foundational Study Items → Release 21 (2026–2027) normative TS → Release 22 (2027–2028) full AI-native integration.

---

## Repository Structure

```
SemmMetricsFr/
├── Framework_Semanticas_Summary_IEEE.md  # Full article (16 sections, 73+ IEEE references)
├── simulate_semantic_metrics.py          # Simulation script reproducing all article figures/tables
├── plot_results.py                       # Generates PNG figures from simulation_results/
├── measurement_algorithms.md            # Pseudocode and complexity for all 16 measurement algorithms
├── complexity_channels_analysis.md      # Detailed channel model and complexity analysis
├── simulation_results/                   # Saved .npz numerical results (auto-created on first run)
│   ├── results_k8.npz
│   ├── results_k16.npz
│   ├── results_k32.npz
│   ├── results_k64.npz
│   ├── results_k128.npz
│   └── results_combined.npz
└── readme                               # This file
```

---

## Dependencies

| Package    | Version  | Notes                                              |
|------------|----------|----------------------------------------------------|
| Python     | ≥ 3.9    |                                                    |
| PyTorch    | ≥ 2.0    | CPU sufficient; CUDA optional                      |
| NumPy      | ≥ 1.23   |                                                    |
| SciPy      | ≥ 1.9    | Kraskov k-NN MI estimator, Sinkhorn OT, norm.ppf   |
| Matplotlib | ≥ 3.6    | Figure generation (imported but plots saved to npz) |

### Installation

```bash
pip install torch>=2.0 numpy>=1.23 scipy>=1.9 matplotlib>=3.6
```

---

## Script: `simulate_semantic_metrics.py`

Implements a complete Monte Carlo sweep evaluating all 16 proposed metrics across five channel models and five bottleneck dimensions. The script is self-contained and requires no external datasets.

### System Model

A DeepJSCC-like semantic autoencoder is implemented as a 3-layer MLP:

```
Encoder f_θ:  d=512 → 128 → 64 → k
Decoder g_φ:  k → 64 → 128 → d=512
```

An **oracle classifier** is trained on clean (uncompressed) embeddings to ~97% accuracy and kept frozen. The autoencoder is trained with a joint loss:

```
L = (1 − λ_t) · MSE(x, x̂) + λ_t · CE(oracle(x̂), y),   λ_t = 0.02
```

This forces the encoder to preserve task-relevant features—the core semantic communications idea—while minimizing reconstruction error.

**Input data**: Class-conditional Gaussian embeddings (10 classes, d=512) serving as a tractable proxy for CLIP (ViT-B/32) or Sentence-BERT (all-MiniLM-L6-v2) real-world embeddings.

### Channel Models

| Channel | Function | Description |
|---------|----------|-------------|
| AWGN | `ch_awgn` | y = z + n, n ~ N(0,σ²I) |
| Rayleigh | `ch_rayleigh` | y = \|h\|·z + n, h ~ CN(0,1) flat fading |
| Rician K=5 dB | `ch_rician(K=5)` | Rician fading with LoS component |
| Rician K=10 dB | `ch_rician(K=10)` | Rician fading with strong LoS component |
| TDL-A (3GPP) | `ch_tdla` | Simplified 3GPP TDL-A, 3 dominant taps [0.60, 0.24, 0.16] |

### Sweep Configuration

| Parameter | Values |
|-----------|--------|
| Bottleneck dimension k | 8, 16, 32, 64, 128 |
| Compression ratio ρ = k/512 | 1.56%, 3.12%, 6.25%, 12.5%, 25% |
| SNR range | −5.0 to +25.0 dB in 2.5 dB steps (13 points) |
| Channel models | 5 (AWGN, Rayleigh, Rician_K5, Rician_K10, TDL-A) |
| Monte Carlo samples N | 1,000 per configuration |
| Total configurations | 5 × 5 × 13 = 325 |

Full adversarial evaluation (PGD-based ARR, SASR, MSD, CertCost) is computed at SNR ∈ {0, 5, 10, 15, 20} dB on AWGN only; the remaining grid points use a closed-form sigmoid approximation to avoid prohibitive runtime.

### Key Hyperparameters (edit at top of file)

```python
D_INPUT         = 512         # input embedding dimension d
NUM_CLASSES     = 10          # number of classification labels
N_SAMPLES       = 1000        # Monte Carlo samples per configuration
TASK_WEIGHT     = 0.02        # λ_t: trade-off between reconstruction and task loss
TRAIN_SNR_DB    = 18.0        # channel SNR during autoencoder training
MIN_EPOCHS      = 300         # minimum training epochs
BASE_EPOCHS     = 800         # base epochs for k=32; scaled as (32/k)^0.4
DEFAULT_ADV_EPS = 8/255       # PGD adversarial budget (Madry et al.)
TDLA_TAP_POWERS = [0.60,0.24,0.16]  # 3GPP TDL-A tap power profile
```

---

## Running the Simulation

```bash
python simulate_semantic_metrics.py
```

The script executes the full pipeline automatically:

1. Generates structured class-conditional Gaussian embeddings (N=1,000, d=512, 10 classes)
2. Trains the oracle classifier on clean data (~97% accuracy)
3. For each bottleneck dimension k ∈ {8, 16, 32, 64, 128}:
   a. Trains the semantic autoencoder with joint MSE + task loss
   b. Evaluates all 16 metrics across all 5 channels × 13 SNR points
4. Saves results to `simulation_results/`
5. Prints Table IV (system comparison), the full metric summary table, and key claims verification

**Estimated runtime:** ~20–60 min on CPU depending on hardware.

---

## Generating Figures

After running the simulation (or using the pre-computed results in `simulation_results/`), generate the two representative PNG figures with:

```bash
python plot_results.py
```

This produces two publication-quality PNG files in `simulation_results/`:

| File | Description |
|------|-------------|
| `fig_semantic_fidelity_vs_k.png` | Semantic Fidelity (RSE, S³I, NSMI) vs. Bottleneck Dimension $k$ at SNR = 10 dB, AWGN |
| `fig_tsr_vs_snr.png` | TSR vs. SNR for k=32, all five channel models, vs. classical JPEG2000+LDPC baseline |

An optional `--output-dir` argument can redirect the PNG output:

```bash
python plot_results.py --output-dir /path/to/output
```

Both figures require only pre-computed `.npz` files; they do not re-run the full simulation.

---

## Output Files

All outputs are written to `simulation_results/` (auto-created):

| File | Contents |
|------|----------|
| `results_k{k}.npz` | All 16 metrics per channel and SNR for bottleneck dimension k (one file per k value) |
| `results_combined.npz` | Single file with all configurations merged |
| `tsr_k_sweep.csv` | TSR vs. k at SNR=10 dB for all five channel models (CSV) |
| `fig_semantic_fidelity_vs_k.png` | Figure: Semantic Fidelity (RSE, S³I) vs. bottleneck dimension k |
| `fig_tsr_vs_snr.png` | Figure: TSR vs. SNR graceful degradation for all channels (k=32) |

Each `.npz` file stores arrays indexed by keys of the form `{channel}__snr{snr}__{metric}`, e.g.:

```python
import numpy as np
r = np.load("simulation_results/results_k32.npz")
tsr = r["AWGN__snr10.0__TSR"]        # Task Success Rate at SNR=10 dB, AWGN
rse = r["Rayleigh__snr10.0__RSE"]    # RSE at SNR=10 dB, Rayleigh
```

Available metric keys per configuration:
`RSE`, `SWD`, `S3I`, `NSMI`, `TSR`, `TSR_CI_lo`, `TSR_CI_hi`, `AP`, `SU`, `CE`, `ID`, `ICC`, `SCI`, `PF`, `ARR`, `SASR`, `CertCost_s`, `CertRadius`, `MSD`

---

## Reproducing Article Results

### Table IV — System Performance Comparison (AWGN, SNR = 10 dB)

Printed automatically to stdout. Reference values from the article:

| System | TSR @ 10 dB | Compression ρ | ARR | CertCost |
|--------|-------------|---------------|-----|----------|
| **Proposed (k=32)** | **0.897** | **6.25%** | **0.043** | **0.026 s** |
| DeepJSCC [46] | 0.867 | 6.25% | 0.08 | ~1.2× |
| JPEG2000+LDPC | 0.888 | ~12.5% | 0.02 | ~2.5× |
| Bit-exact | 0.950 | 100% | <0.01 | ~4.0× |

### Figure 7 — Semantic Fidelity vs. Bottleneck Dimension k

Load `results_combined.npz` and plot RSE and S³I at SNR=10 dB, AWGN channel across k ∈ {8,16,32,64,128}.

### Figure 8 — TSR vs. SNR (Graceful Degradation)

Load `results_k32.npz` and plot TSR for AWGN channel vs. SNR range −5 to 25 dB. Compare against the classical cliff-effect baseline (logistic function):

```python
def classical_tsr(snr_db, cliff=6.0, width=1.5, mx=0.95):
    return mx / (1 + np.exp(-(snr_db - cliff) / width))
```

### Table V — Multi-Channel Performance Comparison (k=32)

Load `results_k32.npz` and extract TSR, RSE, S³I, SWD at SNR ∈ {10, 20} dB across all five channel models.

### Key Claims Verification

The script automatically verifies:

1. **TSR ≈ 0.897** at k=32, SNR=10 dB, AWGN (95% CI: [0.877, 0.914])
2. **60–80% overhead reduction**: ρ = 32/512 = 6.25% → 93.75% overhead reduction
3. **Graceful degradation**: semantic TSR at 0 dB >> classical JPEG2000+LDPC TSR at 0 dB
4. **Spectral efficiency gain**: d/k = 512/32 = 16× raw compression; ~25.6× with intent-driven pruning

---

## Reproducibility

All scripts use fixed random seeds:

```python
torch.manual_seed(42)
np.random.seed(42)
```

Results may vary slightly across platforms due to floating-point non-determinism in PyTorch. For exact bitwise reproducibility on the same hardware:

```python
torch.use_deterministic_algorithms(True)
```

Complete provenance metadata required per measurement report (as specified in TS 39.201 §6.4): UUID, timestamp, metric value + CI, neural model version, hardware spec, channel condition, dataset version, random seed.

---

## Key Mathematical Definitions

### Semantic Entropy (Eq. 1)
```
H_s(X;T) = −Σ_{s∈S_T} p_T(s) log₂ p_T(s)
```
Theorem 1: H_s(X;T) ≤ H(X), with equality iff all source symbols are semantically distinguishable for task T.

### Relative Semantic Entropy (RSE)
```
RSE = I_s(X;Y;T) / H_s(X;T) ∈ [0,1]
```
Estimated via Kraskov k-NN mutual information estimator (k=3), complexity O(N·d·log N).

### Semantic Wasserstein Distance (SWD)
```
SWD_ε(P,Q) = min_{γ∈Γ(P,Q)} Σ_{i,j} γ_{ij} c_{ij} + ε H(γ)
```
Computed via Sinkhorn entropic regularization, complexity O(n²/ε) per iteration.

### Task Success Rate (TSR) with Wilson CI
```
TSR = (1/N) Σ 𝟙[success(x_i, t_i, goal_i)]

CI_Wilson = [p̂ + z²/(2n)] / [1 + z²/n] ± z/(1+z²/n) · sqrt(p̂(1−p̂)/n + z²/(4n²))
```

### Adversarial Robustness Radius (ARR)
```
ARR = min_{δ} ‖δ‖_p  s.t.  S(X+δ) ≠ S(X)
```
Computed via binary-search PGD; bisects interval [ε_lo, ε_hi] over 8 iterations.

### Randomized Smoothing Certificate (Theorem 4)
```
r = (σ/2) · (Φ⁻¹(p_c) − Φ⁻¹(p_2))
```
where p_c is the top-class probability and p_2 is the runner-up probability under Gaussian noise ξ ~ N(0,σ²I).

### Multi-Dimensional Composite Score
```
M_composite = Σ_{i=1}^{4} w_i · M_i      (weighted linear)
M_composite = min_i (M_i / M_i^threshold)  (safety-critical, conjunction-type)
```

---

## Citation

If you use this framework or simulation code in your research, please cite:

```bibtex
@article{semmetrics2024,
  title   = {A Multi-Dimensional Semantic Metric Standardization Framework
             for Evaluating AI-Native Systems in 6G Networks},
  journal = {IEEE},
  year    = {2024},
}
```
