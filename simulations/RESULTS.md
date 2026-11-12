# Simulation results of the round-1 revision (futureinternet-4684256)

<!-- GENERATED from simulations/RESULTS.src.md by run_all.py --report-only; do not edit by hand. -->

Every number below comes from ONE seed-controlled campaign (`simulations/run_all.py`,
R = 5 seeds, seeds 0-4). The same trained models feed every table. Unless
stated otherwise a cell is **mean ± sample standard deviation across the 5 seeds**;
Wilson 95 % confidence intervals are those of the reference seed 0 with N_test = 10,000.
Provenance labels used throughout: **Simulated** (Monte-Carlo output of this code),
**Analytical** (closed-form quantity; where a part of it is simulated this is said explicitly),
**Projected** (no row in any table carries this label any more; every former "estimated"
or "projected" value has been replaced by a simulated one).

## 1. Headline findings (read this first)

1. **V1's TSR = 0.897 was an in-sample number.** The original script trained the task model
   and the autoencoder on the same 1,000 samples it evaluated, so the "100 % accurate oracle"
   had memorised them. Re-running the unchanged V1 code (`v1_protocol_check.py`) gives
   TSR = 0.881 in-sample but 0.255 on fresh samples from the same
   distribution (oracle accuracy 1.000 in-sample vs 0.375 on fresh data). The
   Gaussian proxy with μ_c = 2.0·v̂_c, Σ = I_512 has a Bayes-optimal accuracy of only
   0.670, so no system can exceed TSR ≈ 0.67 on held-out data. The revision uses
   50,000 training and 10,000 held-out test samples; the task model reaches 0.631 on clean test data.
2. **Proposed analog scheme, k = 32, AWGN, 10 dB:** TSR = 0.593 ± 0.002
   (seed 0: 0.590, Wilson 95 % CI [0.580, 0.600]); 0 dB: 0.403;
   Rayleigh 10 dB: 0.548. The SAME models give the k-sweep, so Table 7, 8, 9, 10
   and 11 now agree (the V1 0.855 vs 0.897 discrepancy disappears).
3. **The proposed analog scheme is NOT better than an implemented separate source-channel
   baseline at moderate/high SNR on the Gaussian proxy.** With the same 16 complex channel uses,
   B2 (PCA + 3-bit quantisation, 30 bits, 5G NR LDPC R = 1/2, 16-QAM) reaches TSR = 0.622 at
   10 dB vs 0.593. The analog scheme wins only at SNR ≤ 0 dB (graceful degradation, Table 9).
   On real CLIP embeddings the analog scheme is better than B2 at the same 16 channel uses at every
   SNR (10 dB: 0.922 vs 0.901) and equals B2 with twice the channel uses.
4. **The joint (task-aware) loss gives a small but consistent gain** over the implemented
   DeepJSCC-type analog JSCC (MSE only, same networks and channel uses): 0.593 vs
   0.576 (AWGN 10 dB), CLIP 0.922 vs 0.915. V1's 0.848/0.867 (0.966 × own result) is withdrawn.
5. **"93.75 % overhead reduction" is a channel-use ratio, not a bit saving.** The analog
   scheme sends 32 unquantised real values (16 complex channel uses). The digital variant of the
   proposed encoder needs 128 bits (4 bits/latent) or 256 bits (8 bits/latent); the bit-exact
   reference needs 4096 bits; B2 needs only 30 bits; the PCA source coder alone matches the
   analog TSR with 18 bits (noiseless). See Section 7.
6. **Adversarial training did not improve robustness.** ε_threshold (= ARR, l∞) is
   0.027 (standard), 0.027 (PGD + noise-augmented training of the codec), 0.027 (B1);
   SASR(8/255) = 0.566 / 0.560 / 0.553; none meets the Test Case 3 criterion SASR ≤ 0.30.
   The vulnerability is dominated by the frozen task model and by the small class margins of the
   proxy (Bayes accuracy 0.670). V1's projected ARR ≈ 0.15 is withdrawn. Randomized smoothing gives a
   certified l2 average radius ACR = 0.271 (σ = 0.25, n = 1000, α = 0.001) at
   CertCost = 4.97e8 FLOPs ≈ 4.99 ms per sample (this CPU).
7. **Discrete RSE/NSMI are in [0, 1] by construction and numerically** (min 0.0000,
   max 0.8917 over 18,270 evaluated points and all bootstrap replicates; Miller–Madow
   truncation was never needed). At the reference point RSE = 0.321, NSMI = 0.321
   (bits/bits). The informative k-NN score is 0.0056 ± 0.0019 across seeds and takes negative values
   at some points (min -0.0053), confirming that it is not range-guaranteed.
8. **Proposition 1 (consensus bound) holds in every one of 300,000 multicast trials**
   (N ∈ {2, 4, 8}, AWGN and Rayleigh, receivers at different SNRs); the smallest slack is
   -1.19e-7 (float32 rounding of an equality case, N = 2). The bound is about 2× loose
   on average (D̂ / bound gap ≈ 0.56).

## 2. Setup

| Item | Value |
|---|---|
| Synthetic proxy | 10 classes, d = 512, x = μ_c + n, μ_c = 2.0·v̂_c (v̂_c random unit vectors drawn per seed), n ~ N(0, I_512); N_train = 50,000, N_test = 10,000 (held out) |
| Real embeddings | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`, `get_image_features`, 512-dim) of real CIFAR-10 images: 20,000 training images (2,000 per class, first of each class in data_batch_1-5), the full 10,000-image test set; per-dimension z-scoring fitted on the training split |
| Task model ("oracle") | MLP 512-128-64-10, Adam 1e-3, weight decay 1e-2, 3,000 minibatches of 256, best validation checkpoint (10 % of the training split); frozen afterwards. Clean test accuracy: Gaussian 0.631, CLIP 0.944 |
| Semantic codec | encoder 512-128-64-k, decoder k-64-128-512 (ReLU, orthogonal init), k ∈ {8, 16, 32, 64, 128}; loss (1 − λ_t)·MSE + λ_t·CE(task model(x̂), y), λ_t = 0.02; 3,000 minibatches of 256; AWGN training noise at 18 dB (as V1) |
| B1 DeepJSCC-type analog JSCC | identical networks, channel uses and training, λ_t = 0 (MSE only) — "DeepJSCC-type analog JSCC [61], re-implemented in the embedding domain" |
| PGD-trained codec | same as proposed, every minibatch half clean / half x + σξ + δ_PGD (σ = 0.25, l∞ ε = 8/255, 7 steps, random start), reconstruction target = clean x (SmoothAdv-style, Salman et al. 2019); the task model stays frozen |
| Channel use | the k real latents are sent as k/2 complex symbols (I/Q pairing); power normalised with the training-set average latent power so that E\|s\|² = 1; SNR = E\|s\|²/N0 per complex channel use (identical definition for the digital schemes) |
| Receiver | perfect CSI, per-symbol LMMSE equalisation ŝ = h*y/(\|h\|² + 1/SNR) (analog); APP demapping with per-symbol noise variance N0/\|h\|² after ZF (digital) |
| Channels | AWGN; Rayleigh h ~ CN(0,1); Rician K = 5 / 10 dB (fixed LoS phase); "TDL-A (simplified)": three resolvable Rayleigh taps with powers [0.60, 0.24, 0.16] combined coherently with equal gain and normalised to unit mean power — a flat diversity abstraction of the 3GPP TR 38.901 TDL-A power-delay profile, NOT the full 23-tap frequency-selective model; low-altitude A2G (Section 8). All fading i.i.d. across channel uses (fast fading / ideal interleaving) |
| SNR grid | −5 to 25 dB in 2.5 dB steps |
| Seeds | 0-4; a seed fixes data, initialisation, minibatches and every channel draw (per-point generators derived from (seed, scheme, k, channel, SNR)) |
| Digital chain | Sionna 2.2.0 (PyTorch): `LDPC5GEncoder(K, n)` (rate matching inside), `LDPC5GDecoder` (BP, boxplus-phi, 20 iterations, hard output), Gray QAM `Mapper`/`Demapper("app")`; LLR convention ln P(b=1)/P(b=0) checked (no sign flip needed); no CRC — a failed codeword is de-quantised from whatever bits the decoder outputs |
| N per point | analog schemes and B2: 10,000; digital proposed: first 2,000 test samples; B3: first 500 test samples (clean task-model accuracy on those subsets: 0.630 and 0.641) |

Metric estimators (fixed parameters):

| Metric | Estimator |
|---|---|
| TSR | fraction of test messages whose task-model decision on x̂ equals the true label; Wilson 95 % CI |
| RSE, NSMI (normative, discrete) | S_T = true task label, Ŝ_T = task-model decision on x̂; plug-in entropies from the 10×10 confusion matrix with Miller–Madow correction (truncated to [0, min(H(S), H(Ŝ))], never activated), bits; 95 % percentile bootstrap (1,000 resamples). RSE = I/H(S_T), NSMI = I/√(H(S_T)H(Ŝ_T)) |
| RSE_kNN (informative only) | KSG estimator (algorithm 1) of I over Kozachenko–Leonenko H; fixed random orthonormal projection 512 → d_proj = 16 (projection seed 1234), N = 500 sub-sampled points, k = 3; each projected coordinate standardised |
| S³I | embedding-domain SSIM exactly as the original script |
| SWD | entropic OT (log-domain Sinkhorn) between 500 transmitted and 500 received embeddings, Euclidean cost, ε = 0.1 × median cost, 200 iterations |
| ID | KL(I_T ‖ I_R) in nats, I = task-model softmax posterior; reported per message (mean of per-sample KL, primary) and with the V1 batch-marginal definition |
| SCI | 1 − (1/N²) Σ_{i≠j} D_TV(I_i, I_j), 5 receivers at the same SNR for the reference table |
| AP, SU, ICC, PF, CE | as the original script (CE = TSR/ρ) |
| ARR = ε_threshold | binary search (10 halvings on [0, 0.5]) of the smallest l∞ PGD budget (20 steps, α = 2.5ε/20) that changes the decision of ≥ 50 % of 500 test messages; noiseless channel (white-box attack on encoder → decoder → task model) |
| SASR(ε) | fraction of 500 test messages whose decision changes under l∞ PGD (20 steps) with budget ε (Eq. 44: S(x+δ) ≠ S(x), so SASR(0) = 0) |
| MSD_norm | (L_adv − L_clean)/(ln 10 − L_clean), cross-entropy w.r.t. the true label, ε = 8/255 |
| Randomized smoothing | Cohen et al. (2019) CERTIFY, σ = 0.25, n0 = 100, n = 1000, α = 0.001 (Clopper–Pearson); 500 test messages; ACR = mean certified radius with 0 for abstentions/errors; CertCost = (n0 + n) × forward FLOPs of encoder + decoder + task model, plus measured wall-clock (batched, 4 CPU threads) |

## 3. Semantic fidelity vs compression (replaces V1 Table 7) — Simulated

AWGN, SNR = 10 dB, proposed codec.

| k | ρ (%) | TSR | RSE (discrete, bits/bits) | NSMI (discrete) | RSE_kNN (informative) | S³I | SWD | ID (nats) |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.56 | 0.568 ± 0.005 | 0.293 ± 0.005 | 0.293 ± 0.005 | 0.0028 ± 0.0022 | 0.101 ± 0.000 | 22.96 ± 0.02 | 0.421 ± 0.019 |
| 16 | 3.12 | 0.578 ± 0.002 | 0.303 ± 0.003 | 0.303 ± 0.003 | 0.0036 ± 0.0014 | 0.133 ± 0.001 | 23.10 ± 0.04 | 0.336 ± 0.019 |
| 32 | 6.25 | 0.593 ± 0.002 | 0.321 ± 0.003 | 0.321 ± 0.003 | 0.0056 ± 0.0019 | 0.180 ± 0.001 | 23.40 ± 0.04 | 0.262 ± 0.015 |
| 64 | 12.50 | 0.599 ± 0.003 | 0.329 ± 0.002 | 0.330 ± 0.002 | 0.0073 ± 0.0020 | 0.248 ± 0.004 | 23.84 ± 0.03 | 0.233 ± 0.018 |
| 128 | 25.00 | 0.614 ± 0.005 | 0.346 ± 0.004 | 0.347 ± 0.004 | 0.0082 ± 0.0025 | 0.236 ± 0.011 | 23.73 ± 0.09 | 0.197 ± 0.021 |

## 4. TSR vs compression across channels (replaces V1 Table 8) — Simulated

SNR = 10 dB, same models as Table 7 (no retraining).

| Channel | TSR (k=8) | TSR (k=16) | TSR (k=32) | TSR (k=64) | TSR (k=128) |
|---|---|---|---|---|---|
| AWGN | 0.568 ± 0.005 | 0.578 ± 0.002 | 0.593 ± 0.002 | 0.599 ± 0.003 | 0.614 ± 0.005 |
| Rayleigh | 0.517 ± 0.005 | 0.526 ± 0.006 | 0.548 ± 0.003 | 0.564 ± 0.004 | 0.590 ± 0.004 |
| Rician K=5 dB | 0.546 ± 0.005 | 0.554 ± 0.006 | 0.574 ± 0.005 | 0.583 ± 0.001 | 0.604 ± 0.005 |
| Rician K=10 dB | 0.562 ± 0.006 | 0.573 ± 0.004 | 0.587 ± 0.006 | 0.595 ± 0.001 | 0.610 ± 0.006 |
| TDL-A (simplified) | 0.552 ± 0.005 | 0.565 ± 0.006 | 0.579 ± 0.006 | 0.586 ± 0.005 | 0.610 ± 0.004 |

Reference seed with Wilson 95 % CI (N = 10,000):

| Channel (seed 0, Wilson 95% CI) | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---|---|---|---|---|
| AWGN | 0.561 [0.551, 0.571] | 0.578 [0.568, 0.588] | 0.590 [0.580, 0.600] | 0.596 [0.586, 0.606] | 0.621 [0.611, 0.630] |
| Rayleigh | 0.517 [0.507, 0.527] | 0.517 [0.508, 0.527] | 0.545 [0.535, 0.555] | 0.562 [0.552, 0.572] | 0.592 [0.582, 0.602] |
| Rician K=5 dB | 0.543 [0.533, 0.553] | 0.554 [0.544, 0.563] | 0.572 [0.562, 0.581] | 0.585 [0.575, 0.594] | 0.608 [0.598, 0.618] |
| Rician K=10 dB | 0.556 [0.546, 0.566] | 0.566 [0.557, 0.576] | 0.589 [0.579, 0.598] | 0.594 [0.584, 0.603] | 0.614 [0.605, 0.624] |
| TDL-A (simplified) | 0.551 [0.541, 0.561] | 0.558 [0.549, 0.568] | 0.578 [0.568, 0.587] | 0.587 [0.577, 0.597] | 0.615 [0.606, 0.625] |

## 5. TSR vs SNR with all implemented baselines (replaces V1 Table 9) — Simulated (last column Analytical)

AWGN:

| SNR (dB) | Proposed k=32 (analog) | B1 DeepJSCC-type k=32 | Proposed, PGD-trained k=32 | B2 separate, 16 c.u. (best MCS) | DP-q4 (128 bits, QPSK R=1/2) | B3 bit-exact (4096 bits, QPSK R=1/2) | Analytical bound, 16 c.u. |
|---|---|---|---|---|---|---|---|
| -5.0 | 0.261 ± 0.006 | 0.238 ± 0.004 | 0.258 ± 0.003 | 0.193 ± 0.005 (QPSK R=0.38) | 0.174 ± 0.007 | 0.164 ± 0.013 | 0.405 ± 0.010 |
| -2.5 | 0.328 ± 0.007 | 0.301 ± 0.008 | 0.327 ± 0.005 | 0.267 ± 0.003 (QPSK R=0.38) | 0.221 ± 0.008 | 0.206 ± 0.016 | 0.497 ± 0.008 |
| +0.0 | 0.403 ± 0.006 | 0.373 ± 0.009 | 0.401 ± 0.006 | 0.389 ± 0.006 (QPSK R=0.38) | 0.361 ± 0.010 | 0.256 ± 0.018 | 0.563 ± 0.003 |
| +2.5 | 0.474 ± 0.001 | 0.445 ± 0.007 | 0.473 ± 0.005 | 0.503 ± 0.003 (QPSK R=0.50) | 0.625 ± 0.007 | 0.641 ± 0.026 | 0.594 ± 0.003 |
| +5.0 | 0.532 ± 0.003 | 0.504 ± 0.007 | 0.526 ± 0.006 | 0.574 ± 0.003 (QPSK R=0.66) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.626 ± 0.002 |
| +7.5 | 0.569 ± 0.003 | 0.551 ± 0.005 | 0.568 ± 0.003 | 0.594 ± 0.003 (QPSK R=0.66) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.635 ± 0.003 |
| +10.0 | 0.593 ± 0.002 | 0.576 ± 0.003 | 0.594 ± 0.005 | 0.622 ± 0.002 (16QAM R=0.50) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.637 ± 0.003 |
| +12.5 | 0.607 ± 0.002 | 0.595 ± 0.004 | 0.610 ± 0.005 | 0.631 ± 0.002 (16QAM R=0.75) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.639 ± 0.004 |
| +15.0 | 0.616 ± 0.002 | 0.609 ± 0.002 | 0.619 ± 0.003 | 0.637 ± 0.002 (16QAM R=0.75) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.638 ± 0.004 |
| +17.5 | 0.622 ± 0.001 | 0.618 ± 0.003 | 0.624 ± 0.004 | 0.637 ± 0.002 (16QAM R=0.75) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.639 ± 0.003 |
| +20.0 | 0.624 ± 0.002 | 0.619 ± 0.003 | 0.626 ± 0.006 | 0.637 ± 0.002 (16QAM R=0.75) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.638 ± 0.003 |
| +22.5 | 0.626 ± 0.002 | 0.622 ± 0.002 | 0.629 ± 0.005 | 0.637 ± 0.002 (16QAM R=0.75) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.638 ± 0.003 |
| +25.0 | 0.627 ± 0.003 | 0.622 ± 0.004 | 0.630 ± 0.005 | 0.637 ± 0.002 (16QAM R=0.75) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.638 ± 0.002 |

Rayleigh:

| SNR (dB) | Proposed k=32 (analog) | B1 DeepJSCC-type k=32 | Proposed, PGD-trained k=32 | B2 separate, 16 c.u. (best MCS) | DP-q4 (128 bits, QPSK R=1/2) | B3 bit-exact (4096 bits, QPSK R=1/2) | Analytical bound, 16 c.u. |
|---|---|---|---|---|---|---|---|
| -5.0 | 0.236 ± 0.010 | 0.219 ± 0.004 | 0.240 ± 0.002 | 0.176 ± 0.004 (QPSK R=0.38) | 0.161 ± 0.009 | 0.157 ± 0.005 | 0.380 ± 0.011 |
| -2.5 | 0.292 ± 0.007 | 0.269 ± 0.009 | 0.289 ± 0.003 | 0.227 ± 0.002 (QPSK R=0.38) | 0.193 ± 0.005 | 0.168 ± 0.021 | 0.497 ± 0.008 |
| +0.0 | 0.357 ± 0.005 | 0.323 ± 0.009 | 0.353 ± 0.003 | 0.312 ± 0.003 (QPSK R=0.38) | 0.242 ± 0.014 | 0.215 ± 0.005 | 0.497 ± 0.008 |
| +2.5 | 0.413 ± 0.003 | 0.383 ± 0.006 | 0.408 ± 0.001 | 0.413 ± 0.007 (QPSK R=0.38) | 0.433 ± 0.011 | 0.421 ± 0.027 | 0.595 ± 0.003 |
| +5.0 | 0.467 ± 0.004 | 0.437 ± 0.006 | 0.464 ± 0.005 | 0.494 ± 0.005 (QPSK R=0.50) | 0.622 ± 0.007 | 0.641 ± 0.026 | 0.626 ± 0.002 |
| +7.5 | 0.511 ± 0.003 | 0.485 ± 0.007 | 0.510 ± 0.001 | 0.547 ± 0.003 (QPSK R=0.50) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.634 ± 0.002 |
| +10.0 | 0.548 ± 0.003 | 0.522 ± 0.009 | 0.544 ± 0.006 | 0.574 ± 0.004 (QPSK R=0.66) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.636 ± 0.003 |
| +12.5 | 0.571 ± 0.003 | 0.553 ± 0.006 | 0.573 ± 0.004 | 0.607 ± 0.002 (16QAM R=0.50) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.638 ± 0.004 |
| +15.0 | 0.593 ± 0.005 | 0.576 ± 0.004 | 0.591 ± 0.003 | 0.623 ± 0.001 (16QAM R=0.50) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.639 ± 0.004 |
| +17.5 | 0.605 ± 0.002 | 0.591 ± 0.006 | 0.606 ± 0.003 | 0.625 ± 0.002 (16QAM R=0.50) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.638 ± 0.004 |
| +20.0 | 0.615 ± 0.004 | 0.604 ± 0.002 | 0.614 ± 0.004 | 0.633 ± 0.002 (16QAM R=0.75) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.639 ± 0.003 |
| +22.5 | 0.618 ± 0.003 | 0.611 ± 0.004 | 0.621 ± 0.003 | 0.636 ± 0.002 (16QAM R=0.75) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.638 ± 0.003 |
| +25.0 | 0.621 ± 0.002 | 0.617 ± 0.004 | 0.623 ± 0.005 | 0.637 ± 0.002 (16QAM R=0.75) | 0.630 ± 0.008 | 0.641 ± 0.026 | 0.638 ± 0.003 |

"Best MCS" = the B2 configuration (of QPSK R ∈ {0.38, 1/2, 2/3}, 16-QAM R ∈ {1/2, 3/4}, each with the source
quantiser (m, b) chosen on a validation split for its payload) with the highest seed-mean TSR at that SNR, i.e.
ideal link adaptation. The analytical bound assumes a capacity-achieving code carrying ⌊16·C(SNR)⌋ bits
(AWGN: log2(1+SNR); Rayleigh: ergodic capacity) through the same PCA quantiser; its source part is simulated.
It is not a strict upper bound on TSR (finite block length, a better source coder could exist).

## 6. Multi-channel analysis at k = 32 (replaces V1 Table 10) — Simulated

| Channel | SNR (dB) | TSR | Seed-0 Wilson 95% CI | RSE (discrete) | S³I | SWD | ID (nats) | TSR Δ vs AWGN |
|---|---|---|---|---|---|---|---|---|
| AWGN | 10 | 0.593 ± 0.002 | [0.580, 0.600] | 0.321 ± 0.003 | 0.180 ± 0.001 | 23.40 ± 0.04 | 0.262 ± 0.015 | — (reference) |
| Rayleigh | 10 | 0.548 ± 0.003 | [0.535, 0.555] | 0.272 ± 0.002 | 0.170 ± 0.001 | 23.36 ± 0.04 | 0.374 ± 0.020 | -7.6% |
| Rician K=5 dB | 10 | 0.574 ± 0.005 | [0.562, 0.581] | 0.299 ± 0.006 | 0.175 ± 0.001 | 23.38 ± 0.02 | 0.315 ± 0.017 | -3.3% |
| Rician K=10 dB | 10 | 0.587 ± 0.006 | [0.579, 0.598] | 0.314 ± 0.007 | 0.178 ± 0.001 | 23.38 ± 0.02 | 0.279 ± 0.018 | -1.0% |
| TDL-A (simplified) | 10 | 0.579 ± 0.006 | [0.568, 0.587] | 0.305 ± 0.007 | 0.177 ± 0.001 | 23.39 ± 0.05 | 0.297 ± 0.016 | -2.4% |
| AWGN | 20 | 0.624 ± 0.002 | [0.612, 0.631] | 0.356 ± 0.002 | 0.187 ± 0.001 | 23.41 ± 0.03 | 0.184 ± 0.014 | — (reference) |
| Rayleigh | 20 | 0.615 ± 0.004 | [0.601, 0.620] | 0.346 ± 0.004 | 0.184 ± 0.001 | 23.41 ± 0.02 | 0.214 ± 0.014 | -1.5% |
| Rician K=5 dB | 20 | 0.621 ± 0.004 | [0.605, 0.624] | 0.352 ± 0.004 | 0.186 ± 0.001 | 23.41 ± 0.03 | 0.194 ± 0.014 | -0.6% |
| Rician K=10 dB | 20 | 0.624 ± 0.003 | [0.610, 0.629] | 0.357 ± 0.003 | 0.187 ± 0.001 | 23.41 ± 0.02 | 0.186 ± 0.014 | +0.0% |
| TDL-A (simplified) | 20 | 0.622 ± 0.004 | [0.612, 0.631] | 0.354 ± 0.005 | 0.186 ± 0.001 | 23.41 ± 0.03 | 0.190 ± 0.014 | -0.3% |

All sixteen metrics at the reference point (k = 32, AWGN, 10 dB):

| Metric | Value (k=32, AWGN, 10 dB) |
|---|---|
| RSE (discrete) | 0.321 ± 0.003 |
| NSMI (discrete) | 0.321 ± 0.003 |
| RSE_kNN (informative) | 0.0056 ± 0.0019 |
| S³I | 0.180 ± 0.001 |
| SWD | 23.40 ± 0.04 |
| TSR | 0.593 ± 0.002 |
| AP | 0.567 ± 0.004 |
| SU | 0.032 ± 0.000 |
| CE = TSR/ρ | 9.49 ± 0.04 |
| ID (per-message KL, nats) | 0.262 ± 0.015 |
| ID (V1 marginal definition, nats) | 0.0044 ± 0.0018 |
| ICC | 0.223 ± 0.001 |
| SCI (5 receivers, TV) | 0.825 ± 0.002 |
| PF | 0.132 ± 0.001 |

## 7. Bit accounting and implemented baselines (replaces V1 Table 11; new 11.6) — Simulated / Analytical

AWGN; TSR @ 0/10/20 dB are seed means (std ≤ 0.01 for the analog and B2 rows, ≤ 0.03 for B3 with N = 500).
"Channel uses" counts real dimensions / complex symbols per transmitted embedding.

| Scheme | Channel uses / sample (real dims / complex symbols) | Quant. bits per dim | Payload bits / sample | Modulation | Code rate | TSR @0 dB | TSR @10 dB | TSR @20 dB | ARR (l∞) | SASR(8/255) | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed analog, k=32 (standard) | 32 / 16 | unquantized | n/a (analog) | analog (I/Q) | — | 0.403 | 0.593 | 0.624 | 0.0268 ± 0.0018 | 0.566 ± 0.025 | Simulated |
| Proposed analog, k=32, PGD-trained | 32 / 16 | unquantized | n/a (analog) | analog (I/Q) | — | 0.401 | 0.594 | 0.626 | 0.0270 ± 0.0014 | 0.560 ± 0.013 | Simulated |
| B1 DeepJSCC-type analog JSCC, k=32 (λ_t=0) | 32 / 16 | unquantized | n/a (analog) | analog (I/Q) | — | 0.373 | 0.576 | 0.619 | 0.0271 ± 0.0022 | 0.553 ± 0.029 | Simulated |
| B1 DeepJSCC-type, k=64 | 64 / 32 | unquantized | n/a (analog) | analog (I/Q) | — | 0.370 | 0.583 | 0.623 | — | — | Simulated |
| Proposed analog, k=64 | 64 / 32 | unquantized | n/a (analog) | analog (I/Q) | — | 0.434 | 0.599 | 0.623 | — | — | Simulated |
| B2 separate (PCA m=12, b=1) | 32 / 16 | 1 | 12 | QPSK | 0.375 | 0.389 | 0.497 | 0.497 | — | — | Simulated (Sionna 5G NR LDPC) |
| B2 separate (PCA m=8, b=2) | 32 / 16 | 2 | 16 | QPSK | 0.500 | 0.352 | 0.563 | 0.563 | — | — | Simulated (Sionna 5G NR LDPC) |
| B2 separate (PCA m=10, b=3) | 32 / 16 | 3 | 30 | 16QAM | 0.500 | 0.168 | 0.622 | 0.626 | — | — | Simulated (Sionna 5G NR LDPC) |
| B2 separate (PCA m=10, b=2) | 32 / 16 | 2 | 20 | QPSK | 0.656 | 0.259 | 0.594 | 0.594 | — | — | Simulated (Sionna 5G NR LDPC) |
| B2 separate (PCA m=12, b=4) | 32 / 16 | 4 | 48 | 16QAM | 0.750 | 0.154 | 0.495 | 0.636 | — | — | Simulated (Sionna 5G NR LDPC) |
| B2 separate (PCA m=10, b=2) | 64 / 32 | 2 | 20 | QPSK | 0.328 | 0.529 | 0.594 | 0.594 | — | — | Simulated (Sionna 5G NR LDPC) |
| B2 separate (PCA m=10, b=3) | 64 / 32 | 3 | 30 | QPSK | 0.500 | 0.360 | 0.626 | 0.626 | — | — | Simulated (Sionna 5G NR LDPC) |
| B2 separate (PCA m=10, b=6) | 64 / 32 | 6 | 60 | 16QAM | 0.500 | 0.140 | 0.638 | 0.638 | — | — | Simulated (Sionna 5G NR LDPC) |
| B2 separate (PCA m=10, b=4) | 64 / 32 | 4 | 40 | QPSK | 0.672 | 0.221 | 0.635 | 0.635 | — | — | Simulated (Sionna 5G NR LDPC) |
| B2 separate (PCA m=24, b=4) | 64 / 32 | 4 | 96 | 16QAM | 0.750 | 0.115 | 0.477 | 0.638 | — | — | Simulated (Sionna 5G NR LDPC) |
| B3 bit-exact (512 dims × 8 bits) | 4096 / 2048 | 8 | 4096 | 16QAM | 0.500 | 0.200 | 0.641 | 0.641 | — | — | Simulated (Sionna 5G NR LDPC) |
| B3 bit-exact (512 dims × 8 bits) | 8192 / 4096 | 8 | 4096 | QPSK | 0.500 | 0.256 | 0.641 | 0.641 | — | — | Simulated (Sionna 5G NR LDPC) |
| Digital proposed (k=32 latents × 4 bits) | 128 / 64 | 4 | 128 | 16QAM | 0.500 | 0.210 | 0.630 | 0.630 | — | — | Simulated (Sionna 5G NR LDPC) |
| Digital proposed (k=32 latents × 4 bits) | 256 / 128 | 4 | 128 | QPSK | 0.500 | 0.361 | 0.630 | 0.630 | — | — | Simulated (Sionna 5G NR LDPC) |
| Digital proposed (k=32 latents × 8 bits) | 256 / 128 | 8 | 256 | 16QAM | 0.500 | 0.187 | 0.633 | 0.633 | — | — | Simulated (Sionna 5G NR LDPC) |
| Digital proposed (k=32 latents × 8 bits) | 512 / 256 | 8 | 256 | QPSK | 0.500 | 0.272 | 0.633 | 0.633 | — | — | Simulated (Sionna 5G NR LDPC) |
| Analytical bound: capacity-achieving code + PCA quantizer, 16 c.u. | 32 / 16 | best b | ⌊16·log2(1+SNR)⌋ (55 at 10 dB) | Gaussian (ideal) | C | 0.563 | 0.637 | 0.638 | — | — | Analytical (source part simulated) |
| Analytical bound: capacity-achieving code + PCA quantizer, 32 c.u. | 64 / 32 | best b | ⌊32·log2(1+SNR)⌋ (110 at 10 dB) | Gaussian (ideal) | C | 0.626 | 0.638 | 0.637 | — | — | Analytical (source part simulated) |

Savings at matched TSR (AWGN, 10 dB; target = proposed analog k = 32):

| Scheme (AWGN, 10 dB) | TSR | Channel uses (real / complex) | Payload bits | Channel-use saving of the proposed analog k=32 at matched TSR | Bit comparison with digital proposed q4 (128 bits) |
|---|---|---|---|---|---|
| Target: proposed analog k=32 at AWGN 10 dB | 0.593 | 32 / 16 | n/a (analog) | — | — |
| B2 separate, best MCS at 16 c.u. (16QAM R=0.50, PCA m=10, b=3) | 0.622 | 32 / 16 | 30 | none: baseline reaches the target with the same 16 c.u. | baseline needs 4.3× fewer bits than DP-q4 |
| B2 separate, best MCS at 32 c.u. (16QAM R=0.50, PCA m=10, b=6) | 0.641 | 64 / 32 | 60 | 50.0% fewer c.u. (2× fewer) | baseline needs 2.1× fewer bits than DP-q4 |
| B2 separate, best MCS at 64 c.u. (QPSK R=0.66, PCA m=14, b=6) | 0.641 | 128 / 64 | 84 | 75.0% fewer c.u. (4× fewer) | baseline needs 1.5× fewer bits than DP-q4 |
| B2 separate, best MCS at 128 c.u. (16QAM R=0.50, PCA m=42, b=6) | 0.641 | 256 / 128 | 252 | 87.5% fewer c.u. (8× fewer) | DP-q4 uses 49.2% fewer bits |
| B3 bit-exact, QPSK R=1/2 | 0.641 | 8192 / 4096 | 4096 | 99.6% fewer c.u. (256× fewer) | DP-q4 uses 96.9% fewer bits |
| B3 bit-exact, 16QAM R=1/2 | 0.641 | 4096 / 2048 | 4096 | 99.2% fewer c.u. (128× fewer) | DP-q4 uses 96.9% fewer bits |
| Digital proposed q4, QPSK R=1/2 | 0.630 | 256 / 128 | 128 | 87.5% fewer c.u. (8× fewer) | — |
| Digital proposed q4, 16QAM R=1/2 | 0.630 | 128 / 64 | 128 | 75.0% fewer c.u. (4× fewer) | — |
| Digital proposed q8, QPSK R=1/2 | 0.633 | 512 / 256 | 256 | 93.8% fewer c.u. (16× fewer) | — |
| Digital proposed q8, 16QAM R=1/2 | 0.633 | 256 / 128 | 256 | 87.5% fewer c.u. (8× fewer) | — |
| Source coder alone (PCA + uniform quantizer, noiseless): smallest payload reaching the target | 0.595 | — | 18 | — | — |

Reading: at matched TSR the proposed analog scheme uses 16 complex channel uses vs 2,048-4,096 for the
bit-exact reference (≥ 99.2 % fewer) and 64-128 for its own digital variant, but **separate coding (B2) reaches
the same TSR with the same 16 channel uses and only 30 payload bits**. The digital variant of the
proposed encoder (4 bits per latent, 128 bits) needs 96.9 % fewer bits than the bit-exact reference but
4.3× more bits than B2. The V1 statements "93.75 % transmission overhead reduction", "60-80 %" and
"16×/25.6×/10-16× spectral efficiency" are therefore not supported as bit or spectral-efficiency savings;
ρ = k/d = 6.25 % is only the ratio of channel uses to those of an uncompressed 512-dim analog transmission
under identical per-dimension signalling.

## 8. Adversarial resilience (replaces the adversarial rows of V1 Table 11 and V1 Figure 4) — Simulated

| Model | Clean decision error (noiseless) | ARR = ε_threshold (l∞) | SASR(ε=8/255) | MSD_norm(8/255) | ACR (certified l2, σ=0.25) | Certified acc. at r=0.25 |
|---|---|---|---|---|---|---|
| Proposed k=32, standard training | 0.364 ± 0.014 | 0.0268 ± 0.0018 | 0.566 ± 0.025 | 0.781 ± 0.020 | 0.272 ± 0.007 | 0.481 ± 0.007 |
| B1 DeepJSCC-type k=32 | 0.370 ± 0.023 | 0.0271 ± 0.0022 | 0.553 ± 0.029 | 0.645 ± 0.033 | n/a | n/a |
| Proposed k=32, PGD + noise-augmented training | 0.367 ± 0.012 | 0.0270 ± 0.0014 | 0.560 ± 0.013 | 0.625 ± 0.021 | 0.271 ± 0.008 | 0.479 ± 0.013 |
| Proposed k=32, PGD-trained + randomized smoothing (σ=0.25) | — | 0.0281 ± 0.0021 (EOT-PGD threshold) | 0.552 ± 0.034 | — | (see row above) | — |

CertCost of randomized smoothing:

| Quantity | Value |
|---|---|
| n0 / n / α | 100 / 1000 / 0.001 |
| Forward FLOPs per pass (enc+dec+task model) | 4.518e+05 |
| CertCost FLOPs per certified sample | 4.970e+08 |
| CertCost wall-clock per sample (this CPU, batched) | 5.09 ± 0.44 ms |
| Abstain rate (std / PGD-trained) | 0.090 ± 0.014 / 0.090 ± 0.014 |
| Mean radius over certified-correct samples (PGD-trained) | 0.449 ± 0.007 |
| TSR of smoothed PGD-trained chain, AWGN 10 dB (100 votes) | 0.632 ± 0.010 |

Notes. (i) Under Eq. (44) SASR is a decision-change rate, so SASR(0) = 0; the clean decision error
(first column) is reported separately. (ii) The smoothed chain was attacked adaptively (EOT-PGD, 8 noise
draws per step, decision by 100-vote majority, 200 messages). (iii) The certificate applies to the smoothed
chain; certified accuracy at l2 radius 0.25 is ≈ 0.48 for both the standard and the PGD-trained codec — the
certificate does not depend on adversarial training here. (iv) Test Case 3 (SASR ≤ 0.30 at ε = 8/255) is
failed by every configuration; the certified-radius criterion r ≥ 0.03 is met on average (ACR 0.271).

## 9. Real embeddings: CLIP ViT-B/32 on CIFAR-10 (R2.5; new 11.8) — Simulated

TSR vs SNR and k, AWGN:

| SNR (dB) | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---|---|---|---|---|
| -5.0 | 0.357 ± 0.015 | 0.418 ± 0.009 | 0.491 ± 0.012 | 0.545 ± 0.014 | 0.724 ± 0.012 |
| -2.5 | 0.460 ± 0.008 | 0.536 ± 0.006 | 0.630 ± 0.017 | 0.702 ± 0.009 | 0.830 ± 0.006 |
| +0.0 | 0.578 ± 0.013 | 0.669 ± 0.008 | 0.759 ± 0.010 | 0.818 ± 0.006 | 0.880 ± 0.006 |
| +2.5 | 0.700 ± 0.007 | 0.781 ± 0.007 | 0.845 ± 0.009 | 0.879 ± 0.003 | 0.907 ± 0.002 |
| +5.0 | 0.793 ± 0.004 | 0.850 ± 0.002 | 0.889 ± 0.004 | 0.908 ± 0.003 | 0.922 ± 0.001 |
| +7.5 | 0.849 ± 0.005 | 0.888 ± 0.003 | 0.912 ± 0.002 | 0.922 ± 0.002 | 0.930 ± 0.002 |
| +10.0 | 0.881 ± 0.003 | 0.908 ± 0.003 | 0.922 ± 0.002 | 0.928 ± 0.001 | 0.934 ± 0.001 |
| +12.5 | 0.901 ± 0.003 | 0.918 ± 0.002 | 0.927 ± 0.001 | 0.932 ± 0.001 | 0.936 ± 0.001 |
| +15.0 | 0.911 ± 0.002 | 0.923 ± 0.001 | 0.930 ± 0.002 | 0.935 ± 0.002 | 0.937 ± 0.001 |
| +17.5 | 0.917 ± 0.001 | 0.926 ± 0.001 | 0.933 ± 0.001 | 0.936 ± 0.001 | 0.938 ± 0.001 |
| +20.0 | 0.919 ± 0.002 | 0.927 ± 0.001 | 0.933 ± 0.001 | 0.937 ± 0.001 | 0.938 ± 0.001 |
| +22.5 | 0.921 ± 0.002 | 0.929 ± 0.001 | 0.934 ± 0.002 | 0.938 ± 0.000 | 0.939 ± 0.001 |
| +25.0 | 0.921 ± 0.002 | 0.929 ± 0.001 | 0.935 ± 0.002 | 0.937 ± 0.001 | 0.939 ± 0.001 |

TSR vs SNR and k, Rayleigh:

| SNR (dB) | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---|---|---|---|---|
| -5.0 | 0.319 ± 0.008 | 0.376 ± 0.009 | 0.440 ± 0.010 | 0.487 ± 0.016 | 0.671 ± 0.010 |
| -2.5 | 0.399 ± 0.014 | 0.473 ± 0.009 | 0.558 ± 0.012 | 0.623 ± 0.009 | 0.777 ± 0.008 |
| +0.0 | 0.492 ± 0.008 | 0.574 ± 0.007 | 0.674 ± 0.011 | 0.741 ± 0.004 | 0.845 ± 0.005 |
| +2.5 | 0.590 ± 0.008 | 0.680 ± 0.009 | 0.768 ± 0.009 | 0.822 ± 0.002 | 0.883 ± 0.001 |
| +5.0 | 0.678 ± 0.011 | 0.764 ± 0.006 | 0.832 ± 0.005 | 0.870 ± 0.006 | 0.905 ± 0.004 |
| +7.5 | 0.755 ± 0.007 | 0.821 ± 0.003 | 0.875 ± 0.004 | 0.899 ± 0.003 | 0.918 ± 0.002 |
| +10.0 | 0.812 ± 0.006 | 0.867 ± 0.003 | 0.899 ± 0.003 | 0.913 ± 0.003 | 0.926 ± 0.002 |
| +12.5 | 0.848 ± 0.007 | 0.887 ± 0.004 | 0.913 ± 0.003 | 0.922 ± 0.002 | 0.931 ± 0.002 |
| +15.0 | 0.877 ± 0.005 | 0.903 ± 0.002 | 0.922 ± 0.001 | 0.929 ± 0.002 | 0.933 ± 0.001 |
| +17.5 | 0.894 ± 0.004 | 0.914 ± 0.001 | 0.925 ± 0.002 | 0.932 ± 0.002 | 0.935 ± 0.002 |
| +20.0 | 0.904 ± 0.002 | 0.920 ± 0.001 | 0.930 ± 0.001 | 0.934 ± 0.001 | 0.936 ± 0.002 |
| +22.5 | 0.912 ± 0.003 | 0.923 ± 0.001 | 0.931 ± 0.001 | 0.935 ± 0.001 | 0.937 ± 0.001 |
| +25.0 | 0.916 ± 0.001 | 0.925 ± 0.001 | 0.933 ± 0.002 | 0.936 ± 0.002 | 0.938 ± 0.001 |

Baselines at k = 32 (B2/DP rows use the CLIP task model; DP on the first 2,000 test images, whose clean accuracy is 0.954):

| Channel | SNR (dB) | Proposed k=32 | B1 DeepJSCC-type k=32 | B2 separate 16 c.u. (best MCS) | B2 separate 32 c.u. (best MCS) | DP-q4 (128 bits, QPSK R=1/2, 128 c.u.) | Analytical bound 16 c.u. | RSE (discrete), proposed | S³I, proposed | ID (nats), proposed |
|---|---|---|---|---|---|---|---|---|---|---|
| AWGN | 0 | 0.759 ± 0.010 | 0.748 ± 0.006 | 0.598 ± 0.006 (QPSK R=0.38, m=12, b=1) | 0.772 ± 0.004 (QPSK R=0.33, m=21, b=1) | 0.569 ± 0.017 | 0.861 ± 0.004 | 0.582 ± 0.014 | 0.469 ± 0.004 | 0.623 ± 0.017 |
| AWGN | 10 | 0.922 ± 0.002 | 0.915 ± 0.002 | 0.901 ± 0.002 (16QAM R=0.50, m=16, b=2) | 0.922 ± 0.002 (16QAM R=0.50, m=21, b=3) | 0.947 ± 0.003 | 0.918 ± 0.001 | 0.834 ± 0.003 | 0.768 ± 0.002 | 0.062 ± 0.002 |
| AWGN | 20 | 0.933 ± 0.001 | 0.928 ± 0.002 | 0.917 ± 0.002 (16QAM R=0.75, m=16, b=3) | 0.927 ± 0.002 (16QAM R=0.75, m=32, b=3) | 0.947 ± 0.003 | 0.928 ± 0.002 | 0.855 ± 0.002 | 0.814 ± 0.001 | 0.026 ± 0.001 |
| Rayleigh | 0 | 0.674 ± 0.011 | 0.657 ± 0.013 | 0.467 ± 0.004 (QPSK R=0.38, m=12, b=1) | 0.636 ± 0.004 (QPSK R=0.33, m=21, b=1) | 0.368 ± 0.020 | 0.830 ± 0.005 | 0.465 ± 0.015 | 0.391 ± 0.004 | 0.892 ± 0.024 |
| Rayleigh | 10 | 0.899 ± 0.003 | 0.889 ± 0.003 | 0.856 ± 0.003 (QPSK R=0.50, m=16, b=1) | 0.905 ± 0.002 (QPSK R=0.50, m=16, b=2) | 0.947 ± 0.003 | 0.913 ± 0.001 | 0.795 ± 0.006 | 0.696 ± 0.003 | 0.143 ± 0.004 |
| Rayleigh | 20 | 0.930 ± 0.001 | 0.922 ± 0.003 | 0.912 ± 0.002 (16QAM R=0.75, m=16, b=3) | 0.927 ± 0.001 (16QAM R=0.75, m=32, b=3) | 0.947 ± 0.003 | 0.926 ± 0.002 | 0.849 ± 0.002 | 0.797 ± 0.001 | 0.038 ± 0.001 |

Reference seed, k = 32, AWGN 10 dB: TSR 0.919 (Wilson 95 % CI [0.913, 0.924]); discrete RSE 0.831 (bootstrap
[0.823, 0.842]). Relative to the clean task-model accuracy the proxy is the harder case: TSR/clean accuracy
= 0.94 (Gaussian) vs 0.98 (CLIP) at k = 32, 10 dB. The V1 sentence "the synthetic proxy provides a conservative
lower bound" is not a valid inference from one operating point; the supported statement is that the
qualitative trends (monotone in k and SNR, graceful degradation, ranking proposed > DeepJSCC-type) are the
same on real embeddings, while absolute TSR depends on the task model's clean accuracy (0.631 vs 0.944).

## 10. Low-altitude (UAV air-to-ground) channel (R2.5; new 11.8) — Simulated

Model and parameters (k = 32 proposed codec, 100 sessions × 100 messages per point and seed):

| Parameter | Value | Source |
|---|---|---|
| LoS probability | P_LoS(θ) = 1/(1 + a·exp(−b(θ − a))), θ in degrees, urban a = 9.61, b = 0.16 | Al-Hourani, Kandeepan, Lardner, IEEE Wireless Commun. Lett. 3(6):569–572, 2014 |
| Mean excess loss | η_LoS = 1 dB, η_NLoS = 20 dB (urban) over the nominal (free-space-referenced) SNR | same reference |
| LoS fading | Rician, K(θ) = K0·exp((2/π)·ln(K90/K0)·θ_rad), K0 = 0 dB, K90 = 15 dB | exponential form of Azari et al., IEEE Trans. Commun. 66(1):330–344, 2018; end-point values are assumptions of this study |
| Blocked state | Rayleigh + log-normal shadowing σ = 6 dB, one draw per blockage episode | σ = NLoS shadow-fading std. of the 3GPP TR 36.777 aerial-UE models |
| State process | two-state Markov chain per message slot, p(L→B) = λ(1 − P_LoS), p(B→L) = λ·P_LoS (stationary LoS probability = P_LoS(θ) for every λ); stationary initial state | this study |
| Session | T = 100 messages; session outage = session TSR < 0.9 × clean task-model accuracy | this study |

TSR and ID vs elevation (λ = 0.1):

| Elevation θ (deg) | P_LoS(θ) | K(θ) (dB) | TSR (10 dB) | ID (10 dB) | Session outage (10 dB) | TSR (20 dB) | ID (20 dB) | Session outage (20 dB) |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.100 | 1.7 | 0.220 ± 0.012 | 1.085 ± 0.042 | 1.000 ± 0.000 | 0.376 ± 0.011 | 0.774 ± 0.037 | 0.972 ± 0.029 |
| 20 | 0.354 | 3.3 | 0.311 ± 0.012 | 0.888 ± 0.029 | 0.994 ± 0.005 | 0.454 ± 0.008 | 0.592 ± 0.021 | 0.896 ± 0.051 |
| 30 | 0.731 | 5.0 | 0.458 ± 0.008 | 0.569 ± 0.013 | 0.898 ± 0.013 | 0.550 ± 0.005 | 0.363 ± 0.018 | 0.544 ± 0.044 |
| 40 | 0.931 | 6.7 | 0.540 ± 0.007 | 0.390 ± 0.019 | 0.608 ± 0.056 | 0.604 ± 0.003 | 0.235 ± 0.010 | 0.232 ± 0.016 |
| 50 | 0.985 | 8.3 | 0.568 ± 0.004 | 0.324 ± 0.018 | 0.464 ± 0.042 | 0.617 ± 0.001 | 0.201 ± 0.014 | 0.148 ± 0.033 |
| 60 | 0.997 | 10.0 | 0.575 ± 0.003 | 0.305 ± 0.015 | 0.378 ± 0.034 | 0.621 ± 0.001 | 0.191 ± 0.013 | 0.126 ± 0.021 |
| 70 | 0.999 | 11.7 | 0.580 ± 0.003 | 0.297 ± 0.015 | 0.382 ± 0.033 | 0.624 ± 0.004 | 0.189 ± 0.014 | 0.102 ± 0.029 |
| 80 | 1.000 | 13.3 | 0.583 ± 0.003 | 0.291 ± 0.014 | 0.382 ± 0.031 | 0.623 ± 0.002 | 0.188 ± 0.015 | 0.112 ± 0.016 |
| 90 | 1.000 | 15.0 | 0.584 ± 0.003 | 0.289 ± 0.017 | 0.362 ± 0.031 | 0.624 ± 0.004 | 0.188 ± 0.014 | 0.114 ± 0.018 |

Effect of the blockage transition rate (nominal SNR 20 dB):

| θ (deg) | Transition rate λ | TSR | Session TSR 5th pct. | Std of session TSR | Session outage | ID per message (nats) | Session-level ID (marginal KL, nats) |
|---|---|---|---|---|---|---|---|
| 30 | 0.01 | 0.549 ± 0.014 | 0.309 ± 0.047 | 0.119 ± 0.013 | 0.416 ± 0.042 | 0.366 ± 0.035 | 0.014 ± 0.001 |
| 30 | 0.03 | 0.546 ± 0.015 | 0.353 ± 0.045 | 0.099 ± 0.008 | 0.494 ± 0.055 | 0.370 ± 0.033 | 0.013 ± 0.002 |
| 30 | 0.1 | 0.550 ± 0.005 | 0.422 ± 0.013 | 0.075 ± 0.004 | 0.544 ± 0.044 | 0.363 ± 0.018 | 0.012 ± 0.002 |
| 30 | 0.3 | 0.546 ± 0.005 | 0.452 ± 0.011 | 0.058 ± 0.005 | 0.614 ± 0.042 | 0.369 ± 0.014 | 0.012 ± 0.002 |
| 30 | 1.0 | 0.549 ± 0.004 | 0.474 ± 0.017 | 0.048 ± 0.003 | 0.640 ± 0.054 | 0.366 ± 0.013 | 0.011 ± 0.002 |
| 60 | 0.01 | 0.621 ± 0.003 | 0.542 ± 0.008 | 0.053 ± 0.010 | 0.126 ± 0.021 | 0.194 ± 0.013 | 0.008 ± 0.002 |
| 60 | 0.03 | 0.622 ± 0.003 | 0.542 ± 0.008 | 0.049 ± 0.003 | 0.122 ± 0.029 | 0.191 ± 0.014 | 0.008 ± 0.002 |
| 60 | 0.1 | 0.621 ± 0.001 | 0.542 ± 0.008 | 0.049 ± 0.002 | 0.126 ± 0.021 | 0.191 ± 0.013 | 0.008 ± 0.002 |
| 60 | 0.3 | 0.622 ± 0.002 | 0.544 ± 0.005 | 0.049 ± 0.004 | 0.132 ± 0.016 | 0.191 ± 0.015 | 0.008 ± 0.002 |
| 60 | 1.0 | 0.622 ± 0.002 | 0.540 ± 0.010 | 0.049 ± 0.002 | 0.126 ± 0.013 | 0.190 ± 0.014 | 0.008 ± 0.002 |

Reading: the mean TSR follows P_LoS(θ) and saturates above θ ≈ 50°; it is invariant to λ (stationary
marginals), but the within-session statistics are not: at θ = 30° slow blockage (λ = 0.01) gives a 5th-percentile
session TSR of 0.31 vs 0.47 for λ = 1, i.e. some sessions lose most of their messages, which a per-message
average hides. This is the case for reporting session-level TSR/ID percentiles in the test procedure.

## 11. Numerical check of Proposition 1 (consensus bound, R1.A) — Simulated

Per trial (= one transmitted test message received by N receivers whose SNRs are drawn once per
configuration and seed, uniformly in [−5, 25] dB), I_i = task-model posterior on receiver i's reconstruction,
I_T = posterior on the clean input, δ_i = D_TV(I_i, I_T), SCI = 1 − (1/N²)Σ_{i≠j} D_TV(I_i, I_j), bound =
1 − 2(N−1)/N · mean δ_i. Slack = SCI − bound.

| Channel | N receivers | Trials | Violations | Min slack | Mean slack | Mean SCI | Mean bound | D̂ / bound gap |
|---|---|---|---|---|---|---|---|---|
| AWGN | 2 | 50000 | 0 | -1.19e-07 | 0.126 ± 0.025 | 0.831 ± 0.085 | 0.705 ± 0.065 | 0.536 ± 0.221 |
| AWGN | 4 | 50000 | 0 | 1.19e-04 | 0.188 ± 0.011 | 0.803 ± 0.035 | 0.615 ± 0.029 | 0.510 ± 0.055 |
| AWGN | 8 | 50000 | 0 | 5.12e-04 | 0.240 ± 0.031 | 0.695 ± 0.052 | 0.455 ± 0.082 | 0.559 ± 0.018 |
| Rayleigh | 2 | 50000 | 0 | -1.19e-07 | 0.136 ± 0.025 | 0.775 ± 0.023 | 0.639 ± 0.010 | 0.623 ± 0.066 |
| Rayleigh | 4 | 50000 | 0 | 3.11e-04 | 0.204 ± 0.022 | 0.730 ± 0.060 | 0.526 ± 0.076 | 0.564 ± 0.049 |
| Rayleigh | 8 | 50000 | 0 | 5.04e-04 | 0.236 ± 0.029 | 0.696 ± 0.048 | 0.460 ± 0.077 | 0.562 ± 0.015 |

No violation in 300,000 trials; the only negative slacks (−1.2e−7) are float32 round-off in the N = 2
equality case (when I_1, I_T, I_2 lie "in line", the triangle inequality is tight). On average the bound overstates
the disagreement D̂ by a factor ≈ 1/0.56 ≈ 1.8.

## 12. Estimator variance and the empirical I(Z;Ẑ) (R2.2, SELF-4) — Simulated

| Quantity | Value |
|---|---|
| RSE_kNN, across seeds (mean ± std) | 0.0056 ± 0.0019 |
| RSE_kNN, within seed: std over 20 sub-samples (mean over seeds) | 0.0019 |
| RSE_kNN, std over 10 projection seeds (seed 0) | 0.0019 |
| k-NN MI Î(X;X̂) on the 16-dim projection (nats) | 0.131 ± 0.045 |
| KL entropy Ĥ(X) on the 16-dim projection (nats) | 23.55 ± 0.04 |
| Kraskov Î(Z;Ẑ), k=32 latents, N=500, k=3 (nats) | 2.64 ± 0.02 |
| AWGN capacity of the 16 complex uses at 10 dB (nats) | 38.4 |
| Discrete RSE: bootstrap 95% CI (seed 0) | [0.310, 0.331] |
| Discrete RSE across seeds | 0.321 ± 0.003 |

Sensitivity of the k-NN score to its parameters (seed 0, same reconstruction, AWGN 10 dB):

| d_proj | N | RSE_kNN (seed 0, same data) |
|---|---|---|
| 8 | 250 | 0.0060 |
| 8 | 500 | 0.0136 |
| 8 | 1000 | 0.0122 |
| 16 | 250 | 0.0044 |
| 16 | 500 | 0.0071 |
| 16 | 1000 | 0.0065 |
| 32 | 250 | 0.0022 |
| 32 | 500 | 0.0013 |
| 32 | 1000 | 0.0025 |

The k-NN score changes by a factor of ~10 with d_proj and by ~2 with N on identical data, its relative
standard deviation across seeds is ~35 %, and its denominator (Kozachenko–Leonenko entropy ≈ 23.5 nats)
is a differential entropy that depends on scaling. It can only be an informative, estimator-dependent score
with all parameters fixed. The Kraskov estimate Î(Z;Ẑ) = 2.64 ± 0.02 nats on the
32-dim latents is far below the AWGN capacity of the 16 channel uses at 10 dB (38.4 nats): in 32
dimensions with N = 500 the estimator is strongly biased downwards, so no "operates near capacity" statement
can be drawn from it (SELF-4: the bound TSR ≤ 1 − 2^(−C_s/R_s) and that claim are withdrawn).

## 13. Runtimes

Wall-clock of the full run: 4,701 s (5 seeds in parallel, 4 CPU threads each, 24-core CPU), plus 4 min
for the CLIP extraction. Per seed (seconds, under 5-way contention):

| Seed | Dataset | Training (s) | Main sweep | Estimators | Digital (LDPC) | Adversarial | UAV | Prop. 1 | Total (s) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | gauss | 121 | 265 | 5 | 3391 | 13 | 1 | 1 | 3802 |
| 1 | gauss | 121 | 266 | 5 | 3397 | 12 | 1 | 1 | 3808 |
| 2 | gauss | 121 | 264 | 5 | 3395 | 13 | 1 | 1 | 3805 |
| 3 | gauss | 121 | 264 | 5 | 3394 | 13 | 1 | 1 | 3804 |
| 4 | gauss | 121 | 265 | 5 | 3387 | 14 | 1 | 1 | 3799 |
| 0 | clip | 55 | 225 | — | 605 | — | — | — | 890 |
| 1 | clip | 54 | 226 | — | 604 | — | — | — | 888 |
| 2 | clip | 54 | 226 | — | 605 | — | — | — | 889 |
| 3 | clip | 54 | 228 | — | 602 | — | — | — | 889 |
| 4 | clip | 54 | 227 | — | 606 | — | — | — | 891 |

## 14. Changes vs V1 numbers

| V1 value (location) | New value | Reason |
|---|---|---|
| TSR = 0.897, CI [0.877, 0.914] (k=32, AWGN, 10 dB; blocks 305, 306, 325, 337, 348) | 0.593 ± 0.002; seed 0 CI [0.580, 0.600] | V1 evaluated on its 1,000 training samples (V1 code: 0.881 in-sample vs 0.255 on fresh data); held-out test set; ceiling = Bayes accuracy 0.670 |
| TSR = 0.855 (Table 8, same point) | 0.593 (identical to Table 9/10/11) | one campaign, same models for every table |
| Table 8 AWGN 0.851-0.940, Rayleigh 0.568-0.709 | AWGN 0.568-0.614, Rayleigh 0.517-0.590 | held-out evaluation; CSI + LMMSE equalisation (V1 decoder received \|h\|z + n without equalisation) |
| TSR @0 dB = 0.419; @−5 dB 0.265; @20 dB 0.998 (Table 9) | 0.403; 0.261; 0.624 | as above |
| Classical JPEG2000+LDPC 0.017 @0 dB, 0.888 @10 dB (logistic model) | B2 implemented (PCA + quantiser + 5G LDPC + QAM, 16 c.u.): 0.389 @0 dB, 0.622 @10 dB | logistic cliff model removed; images are never transmitted, so "JPEG2000" was the wrong label |
| Rayleigh −30.4 % vs AWGN at 10 dB (TSR 0.624); −31.9 % at 20 dB | −7.6 % (TSR 0.548); −1.5 % at 20 dB | coherent receiver with CSI; the "fading-limited floor" was an artefact of no equalisation |
| Rician K=10 within 9.6 % of AWGN (0.811) | −1.0 % (0.587) | as above |
| TDL-A highest S³I (0.540 vs 0.488), worst SWD (29.46) | S³I 0.177 vs 0.180 (AWGN); SWD 23.39 vs 23.40 | V1 TDL-A summed tap magnitudes without power normalisation (+2.6 dB array gain); claim not reproduced |
| S³I 0.488 (k=32), 0.38-0.52 (Table 7) | 0.180; 0.10-0.25 | held-out data (in-sample reconstructions were memorised) |
| SWD 25.75 (k=32) | 23.40 | fixed N = 500 and ε, log-domain Sinkhorn; dominated by the irreducible N(0, I) noise of the proxy |
| RSE 0.014-0.023, NSMI 0.014-0.023 (k-NN) | discrete RSE 0.321 (0.293-0.346 over k), NSMI 0.321; k-NN score 0.0056 ± 0.0019 | normative metric redefined on the discrete task variable (R1.C); k-NN kept as informative score with variance |
| Î(Z;Ẑ) = 0.87 nats; "operates near capacity" (block 306) | 2.64 ± 0.02 nats vs capacity 38.4 nats | estimator-dependent; capacity claim withdrawn (SELF-4) |
| ID = 0.024 nats (classifier, marginal); 0.18 nats (embedding-marginal proxy) (block 175) | per-message ID 0.262 nats; V1 marginal definition 0.0044 nats; embedding-marginal proxy not recomputed (dropped) | per-message KL is the intent divergence of each message; the batch-marginal KL averages intents away |
| CLIP TSR 0.841 [0.818, 0.862], N = 1,000, one point; S³I 0.471 | 0.922 (seed 0: 0.919 [0.913, 0.924]), N_test = 10,000, full SNR × k grid; S³I 0.768 | real CIFAR-10 test set, 2,000 training images per class |
| "synthetic proxy is a conservative lower bound" (block 305) | proxy is harder in absolute TSR because its clean accuracy is 0.631 vs 0.944; normalised TSR 0.94 vs 0.98 | SELF-6 |
| DeepJSCC 0.848 (table) / 0.867 (text), = 0.966 × proposed | B1 implemented: 0.576 (CLIP 0.915) | R2.4 |
| Bit-exact 0.950, "100 %" | B3 implemented: 0.641 (N = 500; clean task-model accuracy on that subset 0.641), 4,096 bits, 2,048-4,096 complex c.u. | R2.4 |
| ARR = 0.043 (standard) | 0.027 | 500 messages instead of 50, 20 PGD steps, held-out data |
| ARR ≈ 0.15 (adv + smoothing, "projected") | 0.027 (PGD-trained codec); 0.0281 (smoothed, EOT attack) | actually trained and attacked; no gain |
| DeepJSCC ARR 0.08, JPEG2000 0.02, bit-exact < 0.01 | B1 0.027; digital baselines: not defined (non-differentiable quantiser; not attacked) | invented values removed |
| Certified l2 radius 0.137 / 0.138 (σ = 0.25, M = 100) | ACR 0.272 (standard) / 0.271 (PGD-trained); n0 = 100, n = 1000, α = 0.001 | Cohen et al. CERTIFY with Clopper–Pearson bound |
| CertCost "~1.2×/2.5×/4.0× relative" | 4.97e8 FLOPs, 4.99 ms per sample | measured |
| "93.75 % overhead reduction", "60-80 %", "16×/25.6×/10-16× SE" | channel-use ratio ρ = 6.25 % only; bit savings in Section 7 | R2.1 |
| "Theorem 3" consensus bound (not checked) | Proposition 1 verified in 300,000 trials, 0 violations | R1.A |

## 15. Figures (300 dpi, `work/figs/`)

| File | Replaces | Suggested caption |
|---|---|---|
| `fig_fidelity_vs_k.png` | V1 Figure 7 | **Figure 7.** Task success rate and fidelity metrics (discrete RSE and NSMI, S³I) versus the bottleneck dimension k at SNR = 10 dB over AWGN, for the Gaussian proxy (left) and CLIP ViT-B/32 embeddings of CIFAR-10 (right). Markers: mean over 5 seeds; error bars: ±1 standard deviation (smaller than the markers). Simulated. |
| `fig_tsr_vs_snr.png` | V1 Figure 8 | **Figure 8.** TSR versus SNR per channel use over AWGN (left) and Rayleigh fading (right), Gaussian proxy: proposed analog codec (k = 32, 16 complex channel uses), DeepJSCC-type analog JSCC (B1, same channel uses), separate source–channel coding with 5G NR LDPC and best MCS (B2, 16 channel uses), digital variant of the proposed encoder (4 bits per latent, QPSK, R = 1/2, 128 channel uses) and bit-exact transmission (B3, 4,096 bits, 4,096 channel uses); dashed: analytical separation bound for 16 channel uses; dotted: task-model accuracy on clean input. Mean ± std over 5 seeds. Simulated (bound: Analytical). |
| `fig_tsr_vs_bits.png` | new (Section 11.6) | **Figure @fig:bits.** (a) TSR at 10 dB (AWGN) versus complex channel uses per sample for the analog codecs (k = 8…128) and every implemented digital configuration; (b) noiseless TSR of the PCA + uniform-quantiser source coder versus payload bits, with the digital variants of the proposed encoder and the 8-bit bit-exact reference; dashed: proposed analog codec at 10 dB. Simulated. |
| `fig_clip_snr_k.png` | new (Section 11.8) | **Figure @fig:clip.** TSR of the proposed codec on CLIP ViT-B/32 embeddings of real CIFAR-10 test images versus SNR for k ∈ {8, 16, 32, 64, 128} over AWGN (left) and Rayleigh fading (right), with the separate-coding baseline B2 at 16 channel uses (dashed). Mean ± std over 5 seeds, N_test = 10,000. Simulated. |
| `fig_uav.png` | new (Section 11.8) | **Figure @fig:uav.** Low-altitude air-to-ground link (k = 32): (a) TSR versus elevation angle for nominal SNR 10 and 20 dB (urban LoS probability of Al-Hourani et al., elevation-dependent Rician K, shadowed blocked state, blockage rate λ = 0.1); dotted: P_LoS(θ) × clean accuracy; (b) session outage probability (session TSR below 90 % of the clean accuracy, 100 messages per session) and mean TSR versus blockage transition rate λ at θ = 30° and 60°, 20 dB. Mean ± std over 5 seeds. Simulated. |
| `fig_sasr.png` | V1 Figure 4 | **Figure 4.** Simulated semantic attack success rate (SASR, Eq. (44)) versus l∞ PGD budget ε for the proposed codec with standard training, with PGD + noise-augmented training, the latter with randomized smoothing (σ = 0.25, adaptive EOT attack), and the DeepJSCC-type baseline (k = 32, Gaussian proxy, 500 test messages, mean ± std over 5 seeds). Open circles and dashed verticals: ε_threshold (SASR = 0.5, Eq. (45)); shaded bar: Test Case 3 pass region (SASR ≤ 0.30 at ε = 8/255). Simulated. |

## 16. summary.json keys

All keys of the contract are present (numeric, rounded as they should print). Additional keys:
`tsr_awgn_k32_10db_std`, `tsr_awgn_k32_10db_seed0`, `tsr_awgn_k32_20db`, `rse_disc_awgn_k32_10db_ci_lo/hi`,
`id_clf_marginal_awgn_k32_10db`, `s3i_awgn_k32_10db`, `swd_awgn_k32_10db`, `bits_proposed_q4_k32`,
`sep_code_rate`, `sep_pca_dims`, `sep_bits_per_coeff`, `sep_bits_per_qam_symbol` (configuration behind `tsr_sep_awgn_10db`/`bits_sep`),
`tsr_bitexact_awgn_10db`, `tsr_dp_q4_qpsk_awgn_k32_10db`, `tsr_dp_q8_qpsk_awgn_k32_10db`, `src_bits_to_match_proposed_10db`,
`arr_deepjscc_k32`, `eps_thr_std_k32`, `eps_thr_adv_k32`, `eps_thr_deepjscc_k32` (= ARR of the same model),
`eps_thr_adv_smoothed_k32`, `sasr_*_k32_8_255`, `msd_std_k32`, `msd_adv_k32`, `cert_radius_std_k32`,
`certcost_flops_per_sample`, `certcost_ms_per_sample`, `tsr_adv_smoothed_awgn_k32_10db`, `oracle_clean_acc`, `bayes_acc`,
`clip_*`, `tsr_clip_*`, `s3i_clip_awgn_k32_10db`, `rse_disc_clip_awgn_k32_10db`, `tsr_uav_theta{30,90}_{10,20}db`,
`prop1_*`, `rse_knn_within_seed_std`, `i_z_zhat_knn_nats_*`, `awgn_capacity_nats_k32_10db`, `disc_*`, `knn_rse_min/max`,
`v1code_*` (V1-protocol diagnostic), `oracle_acc_first500/2000`, `clip_oracle_acc_first2000`, `n_test_b3`, `n_test_dp`.

## DEVIATIONS

1. **Held-out evaluation** (not in the contract, required for validity): V1's protocol evaluated in-sample.
   All new numbers are on held-out test data; this lowers every Gaussian-proxy TSR (ceiling 0.67).
   The contract's μ_c = 2.0·v̂_c was kept as instructed so that the proxy stays comparable; the editors should
   present TSR together with the clean task-model accuracy, and CLIP as the realistic case.
2. **Receiver CSI and equalisation** for the analog link (V1: none). Without it the comparison with the
   digital baselines (which need CSI for demapping) would be unfair.
3. **TDL-A simplified** is normalised to unit mean power (V1 was not). It remains a flat diversity
   abstraction, documented as such.
4. **SASR(0) = 0**, not the clean error rate: Eq. (44) defines SASR as a decision change relative to the
   clean decision; the clean decision error is reported in its own column.
5. **ε_threshold = ARR** for the same model (same binary-search procedure, Eq. (45)); the smoothed
   chain's threshold is interpolated from its EOT SASR curve.
6. **B3 and the digital proposed variant** use the first 500 / 2,000 test samples (LDPC decoding cost);
   their subset clean accuracies are given so that TSR differences of ±0.01 are not over-interpreted.
7. **B2 "best MCS"** selects, per SNR, the configuration with the best seed-mean TSR (ideal link adaptation);
   every individual configuration is listed in `results/digital_gauss.csv`.
