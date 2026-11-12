#!/usr/bin/env python3
"""Extract CLIP ViT-B/32 image embeddings of real CIFAR-10 images (R2.5).

Run with the *system* Python (needs ``transformers`` + ``torch``); the result is
cached as ``simulations/cache/clip_vitb32_cifar10.npz`` and then consumed by
``run_all.py`` (which runs in the Sionna venv and never imports transformers).

    python simulations/clip_embed.py [--per-class-train 2000] [--cifar-dir DIR]

* Model: ``openai/clip-vit-base-patch32`` (Hugging Face), ``get_image_features``
  -> 512-dim projected image embedding (the same space CLIP uses for zero-shot
  classification).  Images are pre-processed by the official ``CLIPProcessor``
  (resize 32->224, centre crop, CLIP normalisation).
* Data: CIFAR-10 python batches (Krizhevsky 2009).  Training subset: the first
  ``per_class_train`` images of each class from ``data_batch_1..5``; test: the
  complete ``test_batch`` (10,000 images, 1,000 per class).
* No embeddings are ever fabricated: if the model or dataset cannot be loaded,
  the script exits with an error and nothing is written.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
DEFAULT_CIFAR = os.path.join(os.path.expanduser("~"), ".keras", "datasets", "cifar-10-batches-py")
MODEL_ID = "openai/clip-vit-base-patch32"


def load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC uint8
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class-train", type=int, default=2000)
    ap.add_argument("--cifar-dir", default=DEFAULT_CIFAR)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(CACHE, "clip_vitb32_cifar10.npz"))
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    xs, ys = [], []
    for i in range(1, 6):
        x, y = load_batch(os.path.join(args.cifar_dir, f"data_batch_{i}"))
        xs.append(x); ys.append(y)
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    sel = np.concatenate([np.where(ytr == c)[0][: args.per_class_train] for c in range(10)])
    sel.sort()
    xtr, ytr = xtr[sel], ytr[sel]
    xte, yte = load_batch(os.path.join(args.cifar_dir, "test_batch"))
    print(f"train {xtr.shape[0]} images, test {xte.shape[0]} images", flush=True)

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)

    def embed(x):
        out = []
        t0 = time.time()
        for i in range(0, x.shape[0], args.batch):
            imgs = [Image.fromarray(a) for a in x[i:i + args.batch]]
            inp = proc(images=imgs, return_tensors="pt")
            with torch.no_grad():
                f = model.get_image_features(pixel_values=inp["pixel_values"])
            if not torch.is_tensor(f):  # some transformers versions return a ModelOutput
                f = f.pooler_output if hasattr(f, "pooler_output") else f[0]
            out.append(f.float().numpy())
            if (i // args.batch) % 10 == 0:
                print(f"  {i + len(imgs)}/{x.shape[0]}  {time.time() - t0:.0f}s", flush=True)
        return np.concatenate(out)

    etr = embed(xtr)
    ete = embed(xte)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, x_train=etr.astype(np.float32), y_train=ytr,
                        x_test=ete.astype(np.float32), y_test=yte, model=MODEL_ID,
                        per_class_train=args.per_class_train)
    print("saved", args.out, etr.shape, ete.shape)


if __name__ == "__main__":
    sys.exit(main())
