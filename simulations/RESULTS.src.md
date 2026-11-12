# Simulation results of the round-1 revision (futureinternet-4684256)

<!-- RESULTS.md is GENERATED from simulations/RESULTS.src.md by `run_all.py --report-only`:
     {{T:name}} = table from results/tables.md, {{S:key}} = value from results/summary.json.
     Edit the .src.md file, never RESULTS.md. -->

Every number below comes from ONE seed-controlled campaign (`simulations/run_all.py`,
R = {{S:n_seeds}} seeds, seeds 0-4). The same trained models feed every table. Unless
stated otherwise a cell is **mean ± sample standard deviation across the {{S:n_seeds}} seeds**;
Wilson 95 % confidence intervals are those of the reference seed 0 with N_test = {{S:n_test}}.
Provenance labels used throughout: **Simulated** (Monte-Carlo output of this code),
**Analytical** (closed-form quantity; where a part of it is simulated this is said explicitly),
**Projected** (no row in any table carries this label any more; every former "estimated"
or "projected" value has been replaced by a simulated one).

## 1. Headline findings (read this first)

1. **V1's TSR = 0.897 was an in-sample number.** The original script trained the task model
   and the autoencoder on the same 1,000 samples it evaluated, so the "100 % accurate oracle"
   had memorised them. Re-running the unchanged V1 code (`v1_protocol_check.py`) gives
   TSR = {{S:v1code_tsr_k32_10db_in_sample}} in-sample but {{S:v1code_tsr_k32_10db_fresh}} on fresh samples from the same
   distribution (oracle accuracy {{S:v1code_oracle_acc_in_sample}} in-sample vs {{S:v1code_oracle_acc_fresh}} on fresh data). The
   Gaussian proxy with μ_c = 2.0·v̂_c, Σ = I_512 has a Bayes-optimal accuracy of only
   {{S:bayes_acc}}, so no system can exceed TSR ≈ 0.67 on held-out data. The revision uses
   50,000 training and 10,000 held-out test samples; the task model reaches {{S:oracle_clean_acc}} on clean test data.
2. **Proposed analog scheme, k = 32, AWGN, 10 dB:** TSR = {{S:tsr_awgn_k32_10db}} ± {{S:tsr_awgn_k32_10db_std}}
   (seed 0: {{S:tsr_awgn_k32_10db_seed0}}, Wilson 95 % CI [{{S:tsr_awgn_k32_10db_ci_lo}}, {{S:tsr_awgn_k32_10db_ci_hi}}]); 0 dB: {{S:tsr_awgn_k32_0db}};
   Rayleigh 10 dB: {{S:tsr_rayleigh_k32_10db}}. The SAME models give the k-sweep, so Table 7, 8, 9, 10
   and 11 now agree (the V1 0.855 vs 0.897 discrepancy disappears).
3. **The proposed analog scheme is NOT better than an implemented separate source-channel
   baseline at moderate/high SNR on the Gaussian proxy.** With the same 16 complex channel uses,
   B2 (PCA + 3-bit quantisation, 30 bits, 5G NR LDPC R = 1/2, 16-QAM) reaches TSR = {{S:tsr_sep_awgn_10db}} at
   10 dB vs {{S:tsr_awgn_k32_10db}}. The analog scheme wins only at SNR ≤ 0 dB (graceful degradation, Table 9).
   On real CLIP embeddings the analog scheme is better than B2 at the same 16 channel uses at every
   SNR (10 dB: {{S:tsr_clip_awgn_k32_10db}} vs {{S:tsr_clip_sep_awgn_10db}}) and equals B2 with twice the channel uses.
4. **The joint (task-aware) loss gives a small but consistent gain** over the implemented
   DeepJSCC-type analog JSCC (MSE only, same networks and channel uses): {{S:tsr_awgn_k32_10db}} vs
   {{S:tsr_deepjscc_awgn_k32_10db}} (AWGN 10 dB), CLIP {{S:tsr_clip_awgn_k32_10db}} vs {{S:tsr_clip_deepjscc_awgn_k32_10db}}. V1's 0.848/0.867 (0.966 × own result) is withdrawn.
5. **"93.75 % overhead reduction" is a channel-use ratio, not a bit saving.** The analog
   scheme sends 32 unquantised real values (16 complex channel uses). The digital variant of the
   proposed encoder needs {{S:bits_proposed_q4_k32}} bits (4 bits/latent) or {{S:bits_proposed_q8_k32}} bits (8 bits/latent); the bit-exact
   reference needs {{S:bits_bitexact}} bits; B2 needs only {{S:bits_sep}} bits; the PCA source coder alone matches the
   analog TSR with {{S:src_bits_to_match_proposed_10db}} bits (noiseless). See Section 7.
6. **Adversarial training did not improve robustness.** ε_threshold (= ARR, l∞) is
   {{S:eps_thr_std_k32}} (standard), {{S:eps_thr_adv_k32}} (PGD + noise-augmented training of the codec), {{S:eps_thr_deepjscc_k32}} (B1);
   SASR(8/255) = {{S:sasr_std_k32_8_255}} / {{S:sasr_adv_k32_8_255}} / {{S:sasr_deepjscc_k32_8_255}}; none meets the Test Case 3 criterion SASR ≤ 0.30.
   The vulnerability is dominated by the frozen task model and by the small class margins of the
   proxy (Bayes accuracy {{S:bayes_acc}}). V1's projected ARR ≈ 0.15 is withdrawn. Randomized smoothing gives a
   certified l2 average radius ACR = {{S:cert_radius_smooth_k32}} (σ = 0.25, n = 1000, α = 0.001) at
   CertCost = {{S:certcost_flops_per_sample}} FLOPs ≈ {{S:certcost_ms_per_sample}} ms per sample (this CPU).
7. **Discrete RSE/NSMI are in [0, 1] by construction and numerically** (min {{S:disc_range_min}},
   max {{S:disc_range_max}} over {{S:disc_range_points}} evaluated points and all bootstrap replicates; Miller–Madow
   truncation was never needed). At the reference point RSE = {{S:rse_disc_awgn_k32_10db}}, NSMI = {{S:nsmi_disc_awgn_k32_10db}}
   (bits/bits). The informative k-NN score is {{S:rse_knn_awgn_k32_10db_mean}} ± {{S:rse_knn_awgn_k32_10db_std}} across seeds and takes negative values
   at some points (min {{S:knn_rse_min}}), confirming that it is not range-guaranteed.
8. **Proposition 1 (consensus bound) holds in every one of {{S:prop1_trials}} multicast trials**
   (N ∈ {2, 4, 8}, AWGN and Rayleigh, receivers at different SNRs); the smallest slack is
   {{S:prop1_min_slack}} (float32 rounding of an equality case, N = 2). The bound is about 2× loose
   on average (D̂ / bound gap ≈ {{S:prop1_ratio_mean}}).

## 2. Setup

| Item | Value |
|---|---|
| Synthetic proxy | 10 classes, d = 512, x = μ_c + n, μ_c = 2.0·v̂_c (v̂_c random unit vectors drawn per seed), n ~ N(0, I_512); N_train = 50,000, N_test = 10,000 (held out) |
| Real embeddings | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`, `get_image_features`, 512-dim) of real CIFAR-10 images: 20,000 training images (2,000 per class, first of each class in data_batch_1-5), the full 10,000-image test set; per-dimension z-scoring fitted on the training split |
| Task model ("oracle") | MLP 512-128-64-10, Adam 1e-3, weight decay 1e-2, 3,000 minibatches of 256, best validation checkpoint (10 % of the training split); frozen afterwards. Clean test accuracy: Gaussian {{S:oracle_clean_acc}}, CLIP {{S:clip_oracle_clean_acc}} |
| Semantic codec | encoder 512-128-64-k, decoder k-64-128-512 (ReLU, orthogonal init), k ∈ {8, 16, 32, 64, 128}; loss (1 − λ_t)·MSE + λ_t·CE(task model(x̂), y), λ_t = 0.02; 3,000 minibatches of 256; AWGN training noise at 18 dB (as V1) |
| B1 DeepJSCC-type analog JSCC | identical networks, channel uses and training, λ_t = 0 (MSE only) — "DeepJSCC-type analog JSCC [61], re-implemented in the embedding domain" |
| PGD-trained codec | same as proposed, every minibatch half clean / half x + σξ + δ_PGD (σ = 0.25, l∞ ε = 8/255, 7 steps, random start), reconstruction target = clean x (SmoothAdv-style, Salman et al. 2019); the task model stays frozen |
| Channel use | the k real latents are sent as k/2 complex symbols (I/Q pairing); power normalised with the training-set average latent power so that E\|s\|² = 1; SNR = E\|s\|²/N0 per complex channel use (identical definition for the digital schemes) |
| Receiver | perfect CSI, per-symbol LMMSE equalisation ŝ = h*y/(\|h\|² + 1/SNR) (analog); APP demapping with per-symbol noise variance N0/\|h\|² after ZF (digital) |
| Channels | AWGN; Rayleigh h ~ CN(0,1); Rician K = 5 / 10 dB (fixed LoS phase); "TDL-A (simplified)": three resolvable Rayleigh taps with powers [0.60, 0.24, 0.16] combined coherently with equal gain and normalised to unit mean power — a flat diversity abstraction of the 3GPP TR 38.901 TDL-A power-delay profile, NOT the full 23-tap frequency-selective model; low-altitude A2G (Section 8). All fading i.i.d. across channel uses (fast fading / ideal interleaving) |
| SNR grid | −5 to 25 dB in 2.5 dB steps |
| Seeds | 0-4; a seed fixes data, initialisation, minibatches and every channel draw (per-point generators derived from (seed, scheme, k, channel, SNR)) |
| Digital chain | Sionna 2.2.0 (PyTorch): `LDPC5GEncoder(K, n)` (rate matching inside), `LDPC5GDecoder` (BP, boxplus-phi, 20 iterations, hard output), Gray QAM `Mapper`/`Demapper("app")`; LLR convention ln P(b=1)/P(b=0) checked (no sign flip needed); no CRC — a failed codeword is de-quantised from whatever bits the decoder outputs |
| N per point | analog schemes and B2: 10,000; digital proposed: first 2,000 test samples; B3: first 500 test samples (clean task-model accuracy on those subsets: {{S:oracle_acc_first2000}} and {{S:oracle_acc_first500}}) |

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

{{T:t7}}

## 4. TSR vs compression across channels (replaces V1 Table 8) — Simulated

SNR = 10 dB, same models as Table 7 (no retraining).

{{T:t8}}

Reference seed with Wilson 95 % CI (N = 10,000):

{{T:t8_refci}}

## 5. TSR vs SNR with all implemented baselines (replaces V1 Table 9) — Simulated (last column Analytical)

AWGN:

{{T:t9_AWGN}}

Rayleigh:

{{T:t9_Rayleigh}}

"Best MCS" = the B2 configuration (of QPSK R ∈ {0.38, 1/2, 2/3}, 16-QAM R ∈ {1/2, 3/4}, each with the source
quantiser (m, b) chosen on a validation split for its payload) with the highest seed-mean TSR at that SNR, i.e.
ideal link adaptation. The analytical bound assumes a capacity-achieving code carrying ⌊16·C(SNR)⌋ bits
(AWGN: log2(1+SNR); Rayleigh: ergodic capacity) through the same PCA quantiser; its source part is simulated.
It is not a strict upper bound on TSR (finite block length, a better source coder could exist).

## 6. Multi-channel analysis at k = 32 (replaces V1 Table 10) — Simulated

{{T:t10}}

All sixteen metrics at the reference point (k = 32, AWGN, 10 dB):

{{T:all16}}

## 7. Bit accounting and implemented baselines (replaces V1 Table 11; new 11.6) — Simulated / Analytical

AWGN; TSR @ 0/10/20 dB are seed means (std ≤ 0.01 for the analog and B2 rows, ≤ 0.03 for B3 with N = 500).
"Channel uses" counts real dimensions / complex symbols per transmitted embedding.

{{T:t11}}

Savings at matched TSR (AWGN, 10 dB; target = proposed analog k = 32):

{{T:savings}}

Reading: at matched TSR the proposed analog scheme uses 16 complex channel uses vs 2,048-4,096 for the
bit-exact reference (≥ 99.2 % fewer) and 64-128 for its own digital variant, but **separate coding (B2) reaches
the same TSR with the same 16 channel uses and only {{S:bits_sep}} payload bits**. The digital variant of the
proposed encoder (4 bits per latent, 128 bits) needs 96.9 % fewer bits than the bit-exact reference but
4.3× more bits than B2. The V1 statements "93.75 % transmission overhead reduction", "60-80 %" and
"16×/25.6×/10-16× spectral efficiency" are therefore not supported as bit or spectral-efficiency savings;
ρ = k/d = 6.25 % is only the ratio of channel uses to those of an uncompressed 512-dim analog transmission
under identical per-dimension signalling.

## 8. Adversarial resilience (replaces the adversarial rows of V1 Table 11 and V1 Figure 4) — Simulated

{{T:adv}}

CertCost of randomized smoothing:

{{T:certcost}}

Notes. (i) Under Eq. (44) SASR is a decision-change rate, so SASR(0) = 0; the clean decision error
(first column) is reported separately. (ii) The smoothed chain was attacked adaptively (EOT-PGD, 8 noise
draws per step, decision by 100-vote majority, 200 messages). (iii) The certificate applies to the smoothed
chain; certified accuracy at l2 radius 0.25 is ≈ 0.48 for both the standard and the PGD-trained codec — the
certificate does not depend on adversarial training here. (iv) Test Case 3 (SASR ≤ 0.30 at ε = 8/255) is
failed by every configuration; the certified-radius criterion r ≥ 0.03 is met on average (ACR {{S:cert_radius_smooth_k32}}).

## 9. Real embeddings: CLIP ViT-B/32 on CIFAR-10 (R2.5; new 11.8) — Simulated

TSR vs SNR and k, AWGN:

{{T:clip_AWGN}}

TSR vs SNR and k, Rayleigh:

{{T:clip_Rayleigh}}

Baselines at k = 32 (B2/DP rows use the CLIP task model; DP on the first 2,000 test images, whose clean accuracy is {{S:clip_oracle_acc_first2000}}):

{{T:clip_cmp}}

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

{{T:uav_theta}}

Effect of the blockage transition rate (nominal SNR 20 dB):

{{T:uav_lam}}

Reading: the mean TSR follows P_LoS(θ) and saturates above θ ≈ 50°; it is invariant to λ (stationary
marginals), but the within-session statistics are not: at θ = 30° slow blockage (λ = 0.01) gives a 5th-percentile
session TSR of 0.31 vs 0.47 for λ = 1, i.e. some sessions lose most of their messages, which a per-message
average hides. This is the case for reporting session-level TSR/ID percentiles in the test procedure.

## 11. Numerical check of Proposition 1 (consensus bound, R1.A) — Simulated

Per trial (= one transmitted test message received by N receivers whose SNRs are drawn once per
configuration and seed, uniformly in [−5, 25] dB), I_i = task-model posterior on receiver i's reconstruction,
I_T = posterior on the clean input, δ_i = D_TV(I_i, I_T), SCI = 1 − (1/N²)Σ_{i≠j} D_TV(I_i, I_j), bound =
1 − 2(N−1)/N · mean δ_i. Slack = SCI − bound.

{{T:prop1}}

No violation in {{S:prop1_trials}} trials; the only negative slacks (−1.2e−7) are float32 round-off in the N = 2
equality case (when I_1, I_T, I_2 lie "in line", the triangle inequality is tight). On average the bound overstates
the disagreement D̂ by a factor ≈ 1/{{S:prop1_ratio_mean}} ≈ 1.8.

## 12. Estimator variance and the empirical I(Z;Ẑ) (R2.2, SELF-4) — Simulated

{{T:est}}

Sensitivity of the k-NN score to its parameters (seed 0, same reconstruction, AWGN 10 dB):

{{T:est_sens}}

The k-NN score changes by a factor of ~10 with d_proj and by ~2 with N on identical data, its relative
standard deviation across seeds is ~35 %, and its denominator (Kozachenko–Leonenko entropy ≈ 23.5 nats)
is a differential entropy that depends on scaling. It can only be an informative, estimator-dependent score
with all parameters fixed. The Kraskov estimate Î(Z;Ẑ) = {{S:i_z_zhat_knn_nats_mean}} ± {{S:i_z_zhat_knn_nats_std}} nats on the
32-dim latents is far below the AWGN capacity of the 16 channel uses at 10 dB ({{S:awgn_capacity_nats_k32_10db}} nats): in 32
dimensions with N = 500 the estimator is strongly biased downwards, so no "operates near capacity" statement
can be drawn from it (SELF-4: the bound TSR ≤ 1 − 2^(−C_s/R_s) and that claim are withdrawn).

## 13. Runtimes

Wall-clock of the full run: 4,701 s (5 seeds in parallel, 4 CPU threads each, 24-core CPU), plus 4 min
for the CLIP extraction. Per seed (seconds, under 5-way contention):

{{T:runtime}}

## 14. Changes vs V1 numbers

| V1 value (location) | New value | Reason |
|---|---|---|
| TSR = 0.897, CI [0.877, 0.914] (k=32, AWGN, 10 dB; blocks 305, 306, 325, 337, 348) | {{S:tsr_awgn_k32_10db}} ± {{S:tsr_awgn_k32_10db_std}}; seed 0 CI [{{S:tsr_awgn_k32_10db_ci_lo}}, {{S:tsr_awgn_k32_10db_ci_hi}}] | V1 evaluated on its 1,000 training samples (V1 code: {{S:v1code_tsr_k32_10db_in_sample}} in-sample vs {{S:v1code_tsr_k32_10db_fresh}} on fresh data); held-out test set; ceiling = Bayes accuracy {{S:bayes_acc}} |
| TSR = 0.855 (Table 8, same point) | {{S:tsr_awgn_k32_10db}} (identical to Table 9/10/11) | one campaign, same models for every table |
| Table 8 AWGN 0.851-0.940, Rayleigh 0.568-0.709 | AWGN 0.568-0.614, Rayleigh 0.517-0.590 | held-out evaluation; CSI + LMMSE equalisation (V1 decoder received \|h\|z + n without equalisation) |
| TSR @0 dB = 0.419; @−5 dB 0.265; @20 dB 0.998 (Table 9) | {{S:tsr_awgn_k32_0db}}; 0.261; {{S:tsr_awgn_k32_20db}} | as above |
| Classical JPEG2000+LDPC 0.017 @0 dB, 0.888 @10 dB (logistic model) | B2 implemented (PCA + quantiser + 5G LDPC + QAM, 16 c.u.): 0.389 @0 dB, {{S:tsr_sep_awgn_10db}} @10 dB | logistic cliff model removed; images are never transmitted, so "JPEG2000" was the wrong label |
| Rayleigh −30.4 % vs AWGN at 10 dB (TSR 0.624); −31.9 % at 20 dB | −7.6 % (TSR {{S:tsr_rayleigh_k32_10db}}); −1.5 % at 20 dB | coherent receiver with CSI; the "fading-limited floor" was an artefact of no equalisation |
| Rician K=10 within 9.6 % of AWGN (0.811) | −1.0 % (0.587) | as above |
| TDL-A highest S³I (0.540 vs 0.488), worst SWD (29.46) | S³I 0.177 vs 0.180 (AWGN); SWD 23.39 vs 23.40 | V1 TDL-A summed tap magnitudes without power normalisation (+2.6 dB array gain); claim not reproduced |
| S³I 0.488 (k=32), 0.38-0.52 (Table 7) | {{S:s3i_awgn_k32_10db}}; 0.10-0.25 | held-out data (in-sample reconstructions were memorised) |
| SWD 25.75 (k=32) | {{S:swd_awgn_k32_10db}} | fixed N = 500 and ε, log-domain Sinkhorn; dominated by the irreducible N(0, I) noise of the proxy |
| RSE 0.014-0.023, NSMI 0.014-0.023 (k-NN) | discrete RSE {{S:rse_disc_awgn_k32_10db}} (0.293-0.346 over k), NSMI {{S:nsmi_disc_awgn_k32_10db}}; k-NN score {{S:rse_knn_awgn_k32_10db_mean}} ± {{S:rse_knn_awgn_k32_10db_std}} | normative metric redefined on the discrete task variable (R1.C); k-NN kept as informative score with variance |
| Î(Z;Ẑ) = 0.87 nats; "operates near capacity" (block 306) | {{S:i_z_zhat_knn_nats_mean}} ± {{S:i_z_zhat_knn_nats_std}} nats vs capacity {{S:awgn_capacity_nats_k32_10db}} nats | estimator-dependent; capacity claim withdrawn (SELF-4) |
| ID = 0.024 nats (classifier, marginal); 0.18 nats (embedding-marginal proxy) (block 175) | per-message ID {{S:id_clf_awgn_k32_10db}} nats; V1 marginal definition {{S:id_clf_marginal_awgn_k32_10db}} nats; embedding-marginal proxy not recomputed (dropped) | per-message KL is the intent divergence of each message; the batch-marginal KL averages intents away |
| CLIP TSR 0.841 [0.818, 0.862], N = 1,000, one point; S³I 0.471 | {{S:tsr_clip_awgn_k32_10db}} (seed 0: 0.919 [0.913, 0.924]), N_test = 10,000, full SNR × k grid; S³I {{S:s3i_clip_awgn_k32_10db}} | real CIFAR-10 test set, 2,000 training images per class |
| "synthetic proxy is a conservative lower bound" (block 305) | proxy is harder in absolute TSR because its clean accuracy is 0.631 vs 0.944; normalised TSR 0.94 vs 0.98 | SELF-6 |
| DeepJSCC 0.848 (table) / 0.867 (text), = 0.966 × proposed | B1 implemented: {{S:tsr_deepjscc_awgn_k32_10db}} (CLIP {{S:tsr_clip_deepjscc_awgn_k32_10db}}) | R2.4 |
| Bit-exact 0.950, "100 %" | B3 implemented: {{S:tsr_bitexact_awgn_10db}} (N = 500; clean task-model accuracy on that subset {{S:oracle_acc_first500}}), 4,096 bits, 2,048-4,096 complex c.u. | R2.4 |
| ARR = 0.043 (standard) | {{S:arr_std_k32}} | 500 messages instead of 50, 20 PGD steps, held-out data |
| ARR ≈ 0.15 (adv + smoothing, "projected") | {{S:arr_adv_k32}} (PGD-trained codec); {{S:eps_thr_adv_smoothed_k32}} (smoothed, EOT attack) | actually trained and attacked; no gain |
| DeepJSCC ARR 0.08, JPEG2000 0.02, bit-exact < 0.01 | B1 {{S:arr_deepjscc_k32}}; digital baselines: not defined (non-differentiable quantiser; not attacked) | invented values removed |
| Certified l2 radius 0.137 / 0.138 (σ = 0.25, M = 100) | ACR {{S:cert_radius_std_k32}} (standard) / {{S:cert_radius_smooth_k32}} (PGD-trained); n0 = 100, n = 1000, α = 0.001 | Cohen et al. CERTIFY with Clopper–Pearson bound |
| CertCost "~1.2×/2.5×/4.0× relative" | {{S:certcost_flops_per_sample}} FLOPs, {{S:certcost_ms_per_sample}} ms per sample | measured |
| "93.75 % overhead reduction", "60-80 %", "16×/25.6×/10-16× SE" | channel-use ratio ρ = 6.25 % only; bit savings in Section 7 | R2.1 |
| "Theorem 3" consensus bound (not checked) | Proposition 1 verified in {{S:prop1_trials}} trials, 0 violations | R1.A |

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
