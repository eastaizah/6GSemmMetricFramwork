#!/usr/bin/env python3
"""Diagnostic: why V1 reported TSR = 0.897 (k=32, AWGN, 10 dB).

Runs the authors' ORIGINAL functions (simulations/original/simulate_semantic_metrics.py,
unchanged: seed 42, 1,000 samples, oracle and autoencoder trained on those samples,
original channel) and evaluates the same trained models (a) in-sample, exactly as V1
did, and (b) on 10,000 fresh samples drawn from the same class centroids.

    python simulations/v1_protocol_check.py   ->  results/v1_protocol_check.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "original"))
torch.set_num_threads(4)
import simulate_semantic_metrics as O  # noqa: E402  (seeds 42 at import, as in V1)


def main():
    X, y, cent = O.generate_structured_data(O.N_SAMPLES, O.D_INPUT, O.NUM_CLASSES, separation=2.0)
    oracle = O._train_oracle(X, y)
    enc, dec = O._train_autoencoder(32, X, y, oracle)
    g = torch.Generator().manual_seed(2024)
    yf = torch.randint(0, O.NUM_CLASSES, (10000,), generator=g)
    Xf = cent[yf] + torch.randn(10000, O.D_INPUT, generator=g)
    out = {}
    with torch.no_grad():
        out["oracle_acc_in_sample"] = (oracle(X).argmax(1) == y).float().mean().item()
        out["oracle_acc_fresh"] = (oracle(Xf).argmax(1) == yf).float().mean().item()
        out["bayes_acc_fresh"] = (torch.cdist(Xf, cent).argmin(1) == yf).float().mean().item()
        for name, xx, yy in (("in_sample", X, y), ("fresh", Xf, yf)):
            for snr in (0.0, 10.0, 20.0):
                accs = []
                for _ in range(5):
                    xh = dec(O.ch_awgn(enc(xx), snr))
                    accs.append((oracle(xh).argmax(1) == yy).float().mean().item())
                out[f"tsr_k32_awgn_{int(snr)}db_{name}"] = float(np.mean(accs))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "v1_protocol_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
