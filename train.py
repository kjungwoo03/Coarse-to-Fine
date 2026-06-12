#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coarse-to-Fine training.

Single GPU:
  CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml

Multi GPU (DDP):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config config.yaml

Override any config entry with dotted keys:
  python train.py --config config.yaml --opts train.epochs 50 loss.lmbda.fine 5e-3
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import trange

from model import CoarseToFineCodec, CoarseToFineLoss, LEVELS, strip_and_upgrade_codec_state_keys
from utils import (
    AverageMeter,
    Hier3ProbSumMeter,
    barrier,
    cleanup_distributed,
    ddp_all_reduce_sum_count,
    init_distributed,
    is_dist_avail_and_initialized,
    is_main_process,
    set_seed,
    setup_logger,
    unwrap_model,
)


# -----------------------------
# Config
# -----------------------------
def load_config(path: str, opts) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if opts:
        if len(opts) % 2 != 0:
            raise ValueError("--opts expects pairs: dotted.key value")
        for key, value in zip(opts[0::2], opts[1::2]):
            node = cfg
            parts = key.split(".")
            for p in parts[:-1]:
                node = node[p]
            old = node.get(parts[-1])
            parsed = yaml.safe_load(value)
            if isinstance(parsed, str):
                # YAML 1.1 does not treat "5e-3" as a float; coerce numeric strings
                try:
                    parsed = int(parsed)
                except ValueError:
                    try:
                        parsed = float(parsed)
                    except ValueError:
                        pass
            node[parts[-1]] = parsed
            if is_main_process():
                print(f"[config] override {key}: {old} -> {node[parts[-1]]}")
    return cfg


def make_exp_name(cfg: Dict[str, Any]) -> str:
    m, l = cfg["model"], cfg["loss"]
    dn = m["delta_network"]
    tag_delta = f"D{dn['hidden']}x{dn['depth']}{'-attn' if dn['attn'] else ''}" if dn["enabled"] else "Doff"
    return (
        f"c2f-q{m['quality']}"
        f"--c{m['chunks']['coarse']}-i{m['chunks']['inter']}-f{m['chunks']['fine']}"
        f"--lc{l['lmbda']['coarse']}-li{l['lmbda']['inter']}-lf{l['lmbda']['fine']}"
        f"--gc{l['ce_weight']['coarse']}-gi{l['ce_weight']['inter']}-gf{l['ce_weight']['fine']}"
        f"-ls{l['label_smoothing']}"
        f"--{tag_delta}"
        f"--ep{cfg['train']['epochs']}"
    )


# -----------------------------
# Data
# -----------------------------
def sample_subset_indices(num_total: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if k <= 0 or k >= num_total:
        return np.arange(num_total, dtype=np.int64)
    return rng.choice(num_total, size=k, replace=False).astype(np.int64)


def build_datasets(cfg: Dict[str, Any]):
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    data = cfg["data"]
    seed = int(cfg["train"]["seed"])
    full_train = ImageFolder(root=data["train_root"], transform=train_transforms)
    full_val = ImageFolder(root=data["val_root"], transform=test_transforms)

    train_idx = sample_subset_indices(len(full_train), int(data["train_subset"]), seed)
    val_idx = sample_subset_indices(len(full_val), int(data["val_subset"]), seed + 1)

    return Subset(full_train, train_idx.tolist()), Subset(full_val, val_idx.tolist()), full_val.classes


# -----------------------------
# Classifier (frozen ResNet-50)
# -----------------------------
def build_classifier(device: torch.device) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    if is_dist_avail_and_initialized():
        if is_main_process():
            logging.info("[info] Priming pretrained ResNet-50 weights on rank0 cache")
            _ = resnet50(weights=weights)
        barrier()

    model = resnet50(weights=weights).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# -----------------------------
# Optimizers
# -----------------------------
def configure_optimizers(net: nn.Module, cfg: Dict[str, Any]):
    parameters = {
        n for n, p in net.named_parameters()
        if p.requires_grad and (not n.endswith(".quantiles"))
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if p.requires_grad and n.endswith(".quantiles")
    }

    params_dict = dict(net.named_parameters())
    assert len(parameters & aux_parameters) == 0, "quantiles parameter overlap"
    assert len(parameters | aux_parameters) == len(params_dict), "parameter set mismatch"

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)), lr=float(cfg["train"]["learning_rate"])
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)), lr=float(cfg["train"]["aux_learning_rate"])
    )
    return optimizer, aux_optimizer


# -----------------------------
# Train / Eval epochs
# -----------------------------
def train_one_epoch(model, criterion, train_loader, optimizer, aux_optimizer, epoch, cfg, wandb_run, global_step):
    model.train()
    device = next(model.parameters()).device
    core = unwrap_model(model)
    clip_max_norm = float(cfg["train"]["clip_max_norm"])
    log_interval = int(cfg["wandb"]["log_interval"])

    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    for d, labels in train_loader:
        d = d.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)

        out_net = model(d, mode="train")
        out = criterion(out_net, d, labels)

        out["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = core.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if wandb_run is not None and is_main_process() and (global_step % log_interval == 0):
            logd = {f"train/{k}": v.item() for k, v in out.items()}
            logd["train/aux_loss"] = float(aux_loss.item())
            wandb_run.log(logd, step=global_step)

        global_step += 1

    return global_step


@torch.no_grad()
def test_epoch(epoch, val_loader, model, criterion, wandb_run, cfg, global_step):
    model.eval()
    device = next(model.parameters()).device

    meters: Dict[str, AverageMeter] = {}
    loss_m = AverageMeter()

    for d, labels in val_loader:
        d = d.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out_net = model(d, mode="eval")
        out = criterion(out_net, d, labels)

        bs = d.size(0)
        loss_m.update(out["loss"].item(), bs)
        for k, v in out.items():
            if k == "loss":
                continue
            meters.setdefault(k, AverageMeter()).update(v.item(), bs)

    l_sum, l_cnt = ddp_all_reduce_sum_count(loss_m.sum, loss_m.count, device)
    loss_avg = l_sum / max(l_cnt, 1)

    reduced = {}
    for k, m in meters.items():
        s, c = ddp_all_reduce_sum_count(m.sum, m.count, device)
        reduced[k] = s / max(c, 1)

    if is_main_process():
        logging.info(
            f"[val] epoch={epoch} loss={loss_avg:.4f} "
            + " ".join(f"bpp_{lvl}={reduced.get(f'bpp_{lvl}', float('nan')):.4f}" for lvl in LEVELS)
        )
        if wandb_run is not None:
            logd = {"val/loss": loss_avg, "epoch": epoch}
            for k, v in reduced.items():
                logd[f"val/{k}"] = v
            wandb_run.log(logd, step=global_step)

    return loss_avg


def save_checkpoint(state: dict, exp_dir: str, epoch: int, is_best: bool):
    latest_path = os.path.join(exp_dir, "checkpoint.pth.tar")
    torch.save(state, latest_path)
    shutil.copyfile(latest_path, os.path.join(exp_dir, f"checkpoint_epoch_{epoch:03d}.pth.tar"))
    if is_best:
        shutil.copyfile(latest_path, os.path.join(exp_dir, "checkpoint_best.pth.tar"))


# -----------------------------
# Main
# -----------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Coarse-to-Fine progressive compression training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--opts", nargs="*", default=None,
                        help="Config overrides as dotted-key/value pairs, e.g. --opts train.epochs 50")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    init_distributed(args)
    cfg = load_config(args.config, args.opts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.distributed:
        device = torch.device("cuda", args.local_rank)

    exp_name = make_exp_name(cfg)
    exp_dir = os.path.join(cfg["train"]["save_dir"], exp_name)
    if is_main_process():
        os.makedirs(exp_dir, exist_ok=True)
        setup_logger(os.path.join(exp_dir, "log.txt"))
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        logging.info(f"[info] exp_dir: {exp_dir}")

    set_seed(int(cfg["train"]["seed"]))

    wandb_run = None
    if is_main_process() and cfg["wandb"]["enabled"]:
        import wandb
        wandb_run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=exp_name,
            config=copy.deepcopy(cfg),
        )

    # Data
    train_set, val_set, wnid_list = build_datasets(cfg)
    num_workers = int(cfg["data"]["num_workers"])
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=num_workers,
        sampler=DistributedSampler(train_set) if args.distributed else None,
        shuffle=(not args.distributed),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["train"]["test_batch_size"]),
        num_workers=num_workers,
        sampler=DistributedSampler(val_set, shuffle=False) if args.distributed else None,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    meter = Hier3ProbSumMeter(
        hierarchy_dir=cfg["data"]["hierarchy_dir"],
        wnid_list_in_fine_index_order=wnid_list,
        enable_wup=False,
    )

    # Model
    m = cfg["model"]
    dn = m["delta_network"]
    net = CoarseToFineCodec(
        chunks=m["chunks"],
        quality=int(m["quality"]),
        backbone=m["backbone"],
        delta_enabled=bool(dn["enabled"]),
        delta_hidden=int(dn["hidden"]),
        delta_depth=int(dn["depth"]),
        delta_attn=bool(dn["attn"]),
        clamp_delta_scale=float(dn["clamp_delta_scale"]),
    ).to(device)

    # Resume / warm start
    last_epoch, global_step = 0, 0
    ckpt_dict = {}
    ckpt_path = cfg["train"]["checkpoint"]
    if ckpt_path:
        ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt_dict["state_dict"] if "state_dict" in ckpt_dict else ckpt_dict
        ret = net.load_checkpoint_state(state, strict=False)
        if is_main_process():
            missing = len(getattr(ret, "missing_keys", []))
            unexpected = len(getattr(ret, "unexpected_keys", []))
            logging.info(f"[info] Loaded checkpoint: {ckpt_path} (missing={missing}, unexpected={unexpected})")
    elif is_main_process():
        logging.info("[info] No checkpoint provided. Training from scratch.")

    if is_main_process():
        nparams = sum(p.numel() for p in net.parameters()) / 1e6
        logging.info(f"[info] Model parameters: {nparams:.2f}M | chunks={net.chunk_config}")

    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    classifier = build_classifier(device)

    optimizer, aux_optimizer = configure_optimizers(net, cfg)
    epochs = int(cfg["train"]["epochs"])
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.7), int(epochs * 0.9)], gamma=0.2
    )

    if cfg["train"]["resume_optim"] and isinstance(ckpt_dict, dict) and len(ckpt_dict) > 0:
        for name, obj in (("optimizer", optimizer), ("aux_optimizer", aux_optimizer), ("lr_scheduler", lr_scheduler)):
            if name in ckpt_dict:
                try:
                    obj.load_state_dict(ckpt_dict[name])
                except Exception as e:
                    if is_main_process():
                        logging.warning(f"[resume] {name}.load_state_dict failed (ignored): {e}")
        if "epoch" in ckpt_dict:
            last_epoch = int(ckpt_dict["epoch"]) + 1
        if "global_step" in ckpt_dict:
            global_step = int(ckpt_dict["global_step"])
        if is_main_process():
            logging.info(f"[resume] last_epoch={last_epoch}, global_step={global_step}")

    criterion = CoarseToFineLoss(
        classifier=classifier,
        meter=meter,
        lmbda=cfg["loss"]["lmbda"],
        ce_weight=cfg["loss"]["ce_weight"],
        mse_weight=cfg["loss"].get("mse_weight"),
        label_smoothing=float(cfg["loss"]["label_smoothing"]),
        log_acc=bool(cfg["train"]["log_acc"]),
    )

    best_loss = float("inf")
    for epoch in trange(last_epoch, epochs, desc="Epochs", disable=not is_main_process()):
        if is_main_process():
            logging.info(f"== Epoch {epoch}/{epochs - 1} ==")

        global_step = train_one_epoch(
            net, criterion, train_loader, optimizer, aux_optimizer, epoch, cfg, wandb_run, global_step
        )
        lr_scheduler.step()

        val_loss = test_epoch(epoch, val_loader, net, criterion, wandb_run, cfg, global_step)

        if is_main_process():
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unwrap_model(net).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "config": cfg,
                    "val_loss": val_loss,
                },
                exp_dir,
                epoch,
                is_best,
            )
        barrier()

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main(sys.argv[1:])
