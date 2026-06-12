#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coarse-to-Fine evaluation.

Evaluates a trained checkpoint at the three semantic prefix levels
(coarse / inter / fine) on ImageNet-style data:
  - rate (cumulative bpp per prefix)
  - pixel quality (MSE / PSNR / SSIM / MS-SSIM)
  - hierarchical accuracy (K=10 / K=100 / K=1000 prob-sum top-1)
  - Wu-Palmer similarity of the fine prediction
  - top-1/top-5 for multiple downstream classifiers

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval.py --checkpoint <CHECKPOINT.pth.tar>

Chunk widths and delta-network hyperparameters are resolved automatically from
the checkpoint (embedded config > sidecar config.json/args.json > state_dict
shapes); pass --coarse_chunk/--inter_chunk/--fine_chunk to override.
"""

import argparse
import math
import os
import sys
from typing import Dict

import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import (
    CoarseToFineCodec,
    LEVELS,
    infer_delta_config_from_state,
    resolve_chunk_config,
    state_has_level_ar,
    strip_and_upgrade_codec_state_keys,
)
from utils import (
    MS_SSIM,
    SSIM,
    Hier3ProbSumMeter,
    accuracy_topk,
    cls_preprocess_batch,
    set_seed,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Coarse-to-Fine evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="datasets/ImageNet/val")
    parser.add_argument("--hierarchy_dir", type=str, default="hierarchy")
    parser.add_argument("--save_dir", type=str, default="results/eval")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_images", type=int, default=None,
                        help="Evaluate only the first N images (default: all).")

    parser.add_argument("--backbone", type=str, default="tic")
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--coarse_chunk", type=int, default=None)
    parser.add_argument("--inter_chunk", type=int, default=None)
    parser.add_argument("--fine_chunk", type=int, default=None)

    parser.add_argument(
        "--classifiers", type=str,
        default="resnet50.a1_in1k,convnext_base.fb_in1k,mobilenetv3_large_100",
        help="Comma-separated timm model names for downstream evaluation. "
             "The first one is also used for hierarchical accuracy / WUP.",
    )
    parser.add_argument("--disable_wup", action="store_true",
                        help="Skip Wu-Palmer similarity (no NLTK/WordNet needed).")
    return parser.parse_args(argv)


def build_model(args, ckpt: Dict, state: Dict, device: torch.device) -> CoarseToFineCodec:
    chunk_config, chunk_source = resolve_chunk_config(
        checkpoint_path=args.checkpoint,
        ckpt=ckpt,
        state=state,
        coarse_chunk=args.coarse_chunk,
        inter_chunk=args.inter_chunk,
        fine_chunk=args.fine_chunk,
    )
    print(
        f"[info] chunk_config=coarse:{chunk_config['coarse_chunk']},"
        f"inter:{chunk_config['inter_chunk']},fine:{chunk_config['fine_chunk']} "
        f"(source={chunk_source})"
    )

    delta_enabled = state_has_level_ar(state)
    delta_cfg = {"hidden": 384, "depth": 3, "attn": True}
    clamp_delta_scale = 0.5

    model_cfg = ckpt.get("config", {}).get("model", {}) if isinstance(ckpt, dict) else {}
    if isinstance(model_cfg.get("delta_network"), dict):
        dn = model_cfg["delta_network"]
        delta_cfg.update({k: dn[k] for k in ("hidden", "depth", "attn") if k in dn})
        clamp_delta_scale = float(dn.get("clamp_delta_scale", clamp_delta_scale))
    elif delta_enabled:
        inferred = infer_delta_config_from_state(state)
        if inferred is not None:
            delta_cfg.update(inferred)

    if delta_enabled:
        print(f"[info] delta networks: {delta_cfg} clamp={clamp_delta_scale}")
    else:
        print("[info] checkpoint has no delta networks; evaluating base codec with prefix-masked context")

    net = CoarseToFineCodec(
        chunks={
            "coarse": chunk_config["coarse_chunk"],
            "inter": chunk_config["inter_chunk"],
            "fine": chunk_config["fine_chunk"],
        },
        quality=args.quality,
        backbone=args.backbone,
        delta_enabled=delta_enabled,
        delta_hidden=int(delta_cfg["hidden"]),
        delta_depth=int(delta_cfg["depth"]),
        delta_attn=bool(delta_cfg["attn"]),
        clamp_delta_scale=clamp_delta_scale,
    ).to(device)

    ret = net.load_checkpoint_state(state, strict=False)
    missing = list(getattr(ret, "missing_keys", []))
    unexpected = list(getattr(ret, "unexpected_keys", []))
    print(f"[info] loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[warn] missing keys (up to 10): {missing[:10]}")
    if unexpected:
        print(f"[warn] unexpected keys (up to 10): {unexpected[:10]}")

    net.eval()
    return net


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Data ([0,1] tensors at 256x256; classifier preprocessing happens per batch)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=args.dataset, transform=transform)
    if args.num_images is not None:
        dataset_eval = Subset(dataset, list(range(min(args.num_images, len(dataset)))))
    else:
        dataset_eval = dataset
    dataloader = DataLoader(
        dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[info] dataset={args.dataset} | images={len(dataset_eval)}")

    # Model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = strip_and_upgrade_codec_state_keys(state)
    net = build_model(args, ckpt if isinstance(ckpt, dict) else {}, state, device)

    # Downstream classifiers
    classifier_names = [n.strip() for n in args.classifiers.split(",") if n.strip()]
    classifiers: Dict[str, nn.Module] = {}
    for name in classifier_names:
        classifiers[name] = timm.create_model(name, pretrained=True).to(device).eval()
    hier_classifier = classifier_names[0]
    print(f"[info] classifiers: {classifier_names} (hierarchy/WUP on '{hier_classifier}')")

    meter = Hier3ProbSumMeter(
        hierarchy_dir=args.hierarchy_dir,
        wnid_list_in_fine_index_order=dataset.classes,
        enable_wup=(not args.disable_wup),
    )

    # Accumulators
    stats = {
        level: {
            "n": 0.0,
            "bpp_sum": 0.0,
            "mse_sum": 0.0,
            "psnr_sum": 0.0,
            "ssim_sum": 0.0,
            "ms_ssim_sum": 0.0,
            "coarse10_correct": 0.0,
            "inter100_correct": 0.0,
            "fine_correct": 0.0,
            "wup_sum": 0.0,
            **{name: {"top1": 0.0, "top5": 0.0, "ce_sum": 0.0} for name in classifier_names},
        }
        for level in LEVELS
    }

    with torch.no_grad():
        for x, targets in tqdm(dataloader, desc="Eval"):
            x = x.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            B = x.size(0)

            out = net(x, mode="eval")

            for level in LEVELS:
                x_hat = out[level]["x_hat"].clamp(0, 1)
                s = stats[level]
                s["n"] += B
                s["bpp_sum"] += out[level]["bpp"].item() * B

                mse = (x_hat - x).pow(2).mean().item()
                s["mse_sum"] += mse * B
                s["psnr_sum"] += 10 * math.log10(1.0 / max(mse, 1e-12)) * B
                s["ssim_sum"] += SSIM(x_hat, x).item() * B
                s["ms_ssim_sum"] += MS_SSIM(x_hat, x).item() * B

                x_cls = cls_preprocess_batch(x_hat)
                for name, clf in classifiers.items():
                    logits = clf(x_cls)
                    if name == hier_classifier:
                        h = meter.compute(logits, targets)
                        s["coarse10_correct"] += h["coarse10_correct"]
                        s["inter100_correct"] += h["inter100_correct"]
                        s["fine_correct"] += h["fine_correct"]
                        s["wup_sum"] += h["fine_wup_sum"]
                    acc = accuracy_topk(logits, targets, topk=(1, 5))
                    s[name]["top1"] += acc[1]
                    s[name]["top5"] += acc[5]
                    s[name]["ce_sum"] += acc["ce_loss"] * B

    # Aggregate + report
    rows = []
    header = (
        f"{'Level':<8} | {'BPP':<7} | {'PSNR':<6} | {'C-10':<6} | {'I-100':<6} | {'F-1k':<6} | {'WUP':<6} | "
        + " | ".join(f"{n.split('.')[0]:<12}" for n in classifier_names)
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for level in LEVELS:
        s = stats[level]
        n = max(s["n"], 1.0)
        row = {
            "level": level,
            "bpp": s["bpp_sum"] / n,
            "mse": s["mse_sum"] / n,
            "psnr": s["psnr_sum"] / n,
            "ssim": s["ssim_sum"] / n,
            "ms_ssim": s["ms_ssim_sum"] / n,
            "hier_coarse10": 100.0 * s["coarse10_correct"] / n,
            "hier_inter100": 100.0 * s["inter100_correct"] / n,
            "hier_fine1000": 100.0 * s["fine_correct"] / n,
            "hier_wup": s["wup_sum"] / n,
        }
        top1s = []
        for name in classifier_names:
            row[f"{name}_top1"] = 100.0 * s[name]["top1"] / n
            row[f"{name}_top5"] = 100.0 * s[name]["top5"] / n
            row[f"{name}_ce"] = s[name]["ce_sum"] / n
            top1s.append(row[f"{name}_top1"])

        print(
            f"{level:<8} | {row['bpp']:<7.4f} | {row['psnr']:<6.2f} | "
            f"{row['hier_coarse10']:<6.2f} | {row['hier_inter100']:<6.2f} | {row['hier_fine1000']:<6.2f} | "
            f"{row['hier_wup']:<6.4f} | "
            + " | ".join(f"{t:<12.2f}" for t in top1s)
        )
        rows.append(row)

    print("=" * len(header))

    out_csv = os.path.join(args.save_dir, "results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[info] saved: {out_csv}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
