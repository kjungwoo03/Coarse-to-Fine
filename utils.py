#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities: metrics, classifier preprocessing, hierarchical accuracy
meter (prob-sum + WUP), and DDP helpers.
"""

import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# -----------------------------
# General
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def setup_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    while root_logger.handlers:
        root_logger.handlers.pop()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


# -----------------------------
# Image quality metrics ([0,1] range)
# -----------------------------
def PSNR(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    mse = (x - x_hat).pow(2).mean()
    return 10 * torch.log10(1.0 / mse)


def SSIM(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return ssim(x, x_hat, data_range=1.0, size_average=True)


def MS_SSIM(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return ms_ssim(x, x_hat, data_range=1.0, size_average=True)


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> Dict:
    """Returns raw correct counts per k, plus mean CE loss."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        res[k] = correct[:k].reshape(-1).float().sum().item()
    res["ce_loss"] = F.cross_entropy(logits, targets).item()
    return res


# -----------------------------
# Classifier preprocessing
# -----------------------------
def cls_preprocess_batch(x: torch.Tensor) -> torch.Tensor:
    """x in [0,1], Bx3xHxW -> resized 256, center-cropped 224, ImageNet-normalized."""
    if x.shape[-2:] != (256, 256):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

    top = (256 - 224) // 2
    left = (256 - 224) // 2
    x = x[:, :, top:top + 224, left:left + 224]

    mean = x.new_tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
    return (x - mean) / std


# -----------------------------
# Hierarchy loading
# -----------------------------
def load_wnid_to_cluster_from_csv(
    csv_path: str,
    col_wnid: str = "wnid",
    col_cluster: str = "cluster_id",
) -> Dict[str, int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if col_wnid not in df.columns or col_cluster not in df.columns:
        raise ValueError(f"CSV must contain columns '{col_wnid}', '{col_cluster}'")

    return {str(r[col_wnid]).strip(): int(r[col_cluster]) for _, r in df.iterrows()}


def build_fine_to_cluster(
    wnid_list: List[str],
    wnid_to_cluster: Dict[str, int],
) -> Tuple[torch.Tensor, int]:
    fine_to_cluster = []
    missing = []

    for wn in wnid_list:
        wn = str(wn).strip()
        if wn not in wnid_to_cluster:
            missing.append(wn)
            fine_to_cluster.append(-1)
        else:
            fine_to_cluster.append(int(wnid_to_cluster[wn]))

    if missing:
        raise ValueError(f"Missing wnids in mapping CSV (showing up to 20): {missing[:20]}")

    fine_to_cluster = torch.tensor(fine_to_cluster, dtype=torch.long)
    K = int(fine_to_cluster.max().item()) + 1
    return fine_to_cluster, K


class Hier3ProbSumMeter:
    """
    Hierarchical accuracy meter.

    Coarse/intermediate predictions are obtained by summing fine softmax
    probabilities within each cluster (prob-sum). Optionally also computes the
    Wu-Palmer similarity between the fine prediction and the ground truth.

    Aggregate globally with the *_sum / *_correct fields divided by total n;
    averaging batch means is incorrect for uneven batch sizes.
    """

    def __init__(
        self,
        hierarchy_dir: str,
        wnid_list_in_fine_index_order: List[str],
        k10_name: str = "cluster_assignments_clip_K10.csv",
        k100_name: str = "cluster_assignments_clip_K100.csv",
        eps: float = 1e-12,
        enable_wup: bool = True,
    ):
        self.eps = float(eps)

        w2c10 = load_wnid_to_cluster_from_csv(os.path.join(hierarchy_dir, k10_name))
        w2c100 = load_wnid_to_cluster_from_csv(os.path.join(hierarchy_dir, k100_name))

        self.f2c10, self.K10 = build_fine_to_cluster(wnid_list_in_fine_index_order, w2c10)
        self.f2c100, self.K100 = build_fine_to_cluster(wnid_list_in_fine_index_order, w2c100)

        self.f2c10 = self.f2c10.long().cpu()
        self.f2c100 = self.f2c100.long().cpu()

        self.enable_wup = bool(enable_wup)
        self.wnid_list = [str(w).strip() for w in wnid_list_in_fine_index_order]

        self._wn_synsets = None
        self._wup_pair_cache = {}

        if self.enable_wup:
            self._init_wordnet_synsets()

    def _init_wordnet_synsets(self):
        try:
            from nltk.corpus import wordnet as wn
        except Exception as e:
            raise RuntimeError(
                "WUP enabled but NLTK/wordnet import failed. Install nltk and run:\n"
                "  import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\n"
                f"Original error: {e}"
            )

        try:
            wn.synsets("dog")
        except LookupError as e:
            raise RuntimeError(
                "NLTK wordnet data not found. Run once:\n"
                "  import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\n"
                f"Original error: {e}"
            )

        synsets = []
        for wnid in self.wnid_list:
            if len(wnid) < 2 or wnid[0] not in ("n", "v", "a", "r", "s"):
                raise ValueError(f"Unexpected wnid format: {wnid}")
            pos = wnid[0]
            off = int(wnid[1:])
            try:
                ss = wn.synset_from_pos_and_offset(pos, off)
            except Exception:
                ss = None
            synsets.append(ss)

        self._wn_synsets = synsets

    def _wup(self, pred_idx: int, gt_idx: int) -> float:
        key = (int(pred_idx), int(gt_idx))
        if key in self._wup_pair_cache:
            return self._wup_pair_cache[key]

        s1 = self._wn_synsets[key[0]] if self._wn_synsets is not None else None
        s2 = self._wn_synsets[key[1]] if self._wn_synsets is not None else None

        sim = 0.0
        if (s1 is not None) and (s2 is not None):
            try:
                v = s1.wup_similarity(s2)
                sim = float(v) if v is not None else 0.0
            except Exception:
                sim = 0.0

        self._wup_pair_cache[key] = sim
        return sim

    @staticmethod
    def _cluster_probs_sum(probs_fine: torch.Tensor, fine_to_cluster: torch.Tensor, K: int) -> torch.Tensor:
        out = torch.zeros((probs_fine.size(0), K), device=probs_fine.device, dtype=probs_fine.dtype)
        out.index_add_(dim=1, index=fine_to_cluster, source=probs_fine)
        return out

    @torch.no_grad()
    def compute(self, logits_fine: torch.Tensor, targets_fine: torch.Tensor) -> Dict[str, float]:
        if targets_fine.dim() != 1:
            targets_fine = targets_fine.view(-1)

        B = int(targets_fine.numel())
        device = logits_fine.device

        probs_fine = F.softmax(logits_fine, dim=1)
        pred_fine = probs_fine.argmax(dim=1)
        fine_correct = int((pred_fine == targets_fine).sum().item())
        fine_ce_sum = float(F.cross_entropy(logits_fine, targets_fine, reduction="sum").item())

        fine_wup_sum = 0.0
        fine_wup_mean = 0.0
        if self.enable_wup:
            pred_cpu = pred_fine.detach().cpu().numpy()
            gt_cpu = targets_fine.detach().cpu().numpy()
            s = 0.0
            for p, g in zip(pred_cpu.tolist(), gt_cpu.tolist()):
                s += self._wup(p, g)
            fine_wup_sum = float(s)
            fine_wup_mean = float(s / B) if B > 0 else 0.0

        f2c10 = self.f2c10.to(device)
        f2c100 = self.f2c100.to(device)

        t10 = f2c10[targets_fine]
        t100 = f2c100[targets_fine]

        p10 = self._cluster_probs_sum(probs_fine, f2c10, self.K10)
        p100 = self._cluster_probs_sum(probs_fine, f2c100, self.K100)

        c10_correct = int((p10.argmax(dim=1) == t10).sum().item())
        c100_correct = int((p100.argmax(dim=1) == t100).sum().item())

        c10_ce_sum = float((-torch.log(p10.gather(1, t10[:, None]).clamp_min(self.eps))).sum().item())
        c100_ce_sum = float((-torch.log(p100.gather(1, t100[:, None]).clamp_min(self.eps))).sum().item())

        return {
            "n": B,

            "fine_correct": fine_correct,
            "fine_ce_sum": fine_ce_sum,
            "fine_top1": 100.0 * fine_correct / B,
            "fine_ce": fine_ce_sum / B,

            "coarse10_correct": c10_correct,
            "coarse10_ce_sum": c10_ce_sum,
            "coarse10_top1": 100.0 * c10_correct / B,
            "coarse10_ce": c10_ce_sum / B,

            "inter100_correct": c100_correct,
            "inter100_ce_sum": c100_ce_sum,
            "inter100_top1": 100.0 * c100_correct / B,
            "inter100_ce": c100_ce_sum / B,

            "fine_wup_sum": fine_wup_sum,
            "fine_wup_mean": fine_wup_mean,
        }


# -----------------------------
# DDP helpers
# -----------------------------
def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier(device_id: Optional[int] = None):
    if not is_dist_avail_and_initialized():
        return

    if dist.get_backend() == "nccl":
        if device_id is None and torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        if device_id is not None:
            dist.barrier(device_ids=[int(device_id)])
            return

    dist.barrier()


def ddp_all_reduce_sum_count(sum_val: float, count_val: int, device: torch.device) -> Tuple[float, int]:
    if not is_dist_avail_and_initialized():
        return sum_val, count_val
    t = torch.tensor([sum_val, float(count_val)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(t[1].item())


def init_distributed(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = args.world_size > 1
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", args.local_rank),
        )


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
        barrier(device_id=device_id)
        dist.destroy_process_group()
