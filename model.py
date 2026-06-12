#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coarse-to-Fine progressive codec.

The 320 latent channels of a TIC backbone are split into three prefix blocks
(coarse / intermediate / fine). Each prefix is decoded through the shared
synthesis transform, and entropy parameters are recomputed per prefix with a
prefix-masked context. Delta networks refine the entropy parameters of the
intermediate and fine blocks conditioned on the lower-level context.
"""

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.zoo import image_models

TOTAL_LATENT_CHANNELS = 320

LEVELS = ("coarse", "inter", "fine")


# -----------------------------
# Chunk configuration
# -----------------------------
def build_chunk_config(coarse_chunk: int, inter_chunk: int, fine_chunk: int) -> Dict[str, int]:
    coarse_chunk = int(coarse_chunk)
    inter_chunk = int(inter_chunk)
    fine_chunk = int(fine_chunk)

    if min(coarse_chunk, inter_chunk, fine_chunk) <= 0:
        raise ValueError("All chunk widths must be positive.")
    if coarse_chunk + inter_chunk + fine_chunk != TOTAL_LATENT_CHANNELS:
        raise ValueError(
            f"coarse_chunk + inter_chunk + fine_chunk must equal {TOTAL_LATENT_CHANNELS}."
        )

    return {
        "coarse_chunk": coarse_chunk,
        "inter_chunk": inter_chunk,
        "fine_chunk": fine_chunk,
        "coarse_end": coarse_chunk,
        "inter_end": coarse_chunk + inter_chunk,
        "fine_end": TOTAL_LATENT_CHANNELS,
    }


# -----------------------------
# Checkpoint key handling
# -----------------------------
def strip_and_upgrade_codec_state_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Strip DDP prefixes and upgrade legacy level-AR module names."""
    upgraded: Dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("level_ar_224."):
            key = "level_ar_inter." + key[len("level_ar_224."):]
        elif key.startswith("level_ar_320."):
            key = "level_ar_fine." + key[len("level_ar_320."):]
        upgraded[key] = value
    return upgraded


def state_has_level_ar(state: Dict[str, Any]) -> bool:
    return any(
        key.startswith(("level_ar_inter.", "level_ar_fine.", "codec.level_ar_inter.", "codec.level_ar_fine."))
        for key in state
    )


def infer_chunk_config_from_state(state: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Infer chunk widths from the output projections of the delta networks."""
    def _out_ch(prefixes) -> Optional[int]:
        for p in prefixes:
            for key in (f"{p}.out_proj.2.weight", f"codec.{p}.out_proj.2.weight"):
                if key in state and hasattr(state[key], "shape"):
                    return int(state[key].shape[0])
        return None

    inter_out = _out_ch(["level_ar_inter"])
    fine_out = _out_ch(["level_ar_fine"])
    if inter_out is None or fine_out is None:
        return None
    if inter_out % 2 != 0 or fine_out % 2 != 0:
        return None

    inter_chunk = inter_out // 2
    fine_chunk = fine_out // 2
    coarse_chunk = TOTAL_LATENT_CHANNELS - inter_chunk - fine_chunk
    if coarse_chunk <= 0:
        return None
    return build_chunk_config(coarse_chunk, inter_chunk, fine_chunk)


def infer_delta_config_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Infer delta-network hyperparameters (hidden/depth/attn) from state_dict shapes."""
    prefix = None
    for p in ("level_ar_inter", "codec.level_ar_inter"):
        if f"{p}.in_proj.0.weight" in state:
            prefix = p
            break
    if prefix is None:
        return None

    hidden = int(state[f"{prefix}.in_proj.0.weight"].shape[0])
    depth = len({k.split(".")[-3] for k in state if k.startswith(f"{prefix}.blocks.") and k.endswith(".pw1.weight")})
    attn = any(k.startswith(f"{prefix}.attn.") for k in state)
    return {"hidden": hidden, "depth": max(depth, 1), "attn": attn}


def load_sidecar_config(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load a config/args JSON saved next to the checkpoint, if present."""
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    for name in ("config.json", "args.json"):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def resolve_chunk_config(
    *,
    checkpoint_path: str = "",
    ckpt: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    coarse_chunk: Optional[int] = None,
    inter_chunk: Optional[int] = None,
    fine_chunk: Optional[int] = None,
) -> Tuple[Dict[str, int], str]:
    """Resolve chunk widths: CLI > checkpoint config > sidecar JSON > state_dict shapes > default."""
    cli = (coarse_chunk, inter_chunk, fine_chunk)
    if any(v is not None for v in cli):
        if not all(v is not None for v in cli):
            raise ValueError("Provide coarse_chunk, inter_chunk, and fine_chunk together.")
        return build_chunk_config(*cli), "cli"

    if isinstance(ckpt, dict):
        cfg = ckpt.get("config")
        if isinstance(cfg, dict):
            chunks = cfg.get("model", {}).get("chunks")
            if isinstance(chunks, dict) and all(k in chunks for k in LEVELS):
                return build_chunk_config(chunks["coarse"], chunks["inter"], chunks["fine"]), "checkpoint_config"

    if checkpoint_path:
        sidecar = load_sidecar_config(checkpoint_path)
        if sidecar is not None:
            if all(k in sidecar for k in ("coarse_chunk", "inter_chunk", "fine_chunk")):
                return (
                    build_chunk_config(
                        sidecar["coarse_chunk"], sidecar["inter_chunk"], sidecar["fine_chunk"]
                    ),
                    "sidecar_json",
                )

    if state:
        inferred = infer_chunk_config_from_state(state)
        if inferred is not None:
            return inferred, "state_dict"

    return build_chunk_config(128, 96, 96), "default"


# -----------------------------
# Delta network (Fig. 3b)
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(ch // r, 4), 1)
        self.fc2 = nn.Conv2d(max(ch // r, 4), ch, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class DWResBlock(nn.Module):
    def __init__(self, ch: int, dw_ks: int = 3):
        super().__init__()
        pad = dw_ks // 2
        self.pw1 = nn.Conv2d(ch, ch, 1)
        self.dw = nn.Conv2d(ch, ch, dw_ks, padding=pad, groups=ch)
        self.pw2 = nn.Conv2d(ch, ch, 1)
        self.se = SEBlock(ch, r=8)

    def forward(self, x):
        y = F.gelu(self.pw1(x))
        y = F.gelu(self.dw(y))
        y = self.pw2(y)
        y = self.se(y)
        return x + y


class TinySpatialAttention(nn.Module):
    """Lightweight MHSA on spatial tokens; intended for small latent maps."""

    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(embed_dim=ch, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        t = self.norm(t)
        t2, _ = self.attn(t, t, t, need_weights=False)
        t = t + t2
        return t.transpose(1, 2).view(B, C, H, W)


class DeltaNetwork(nn.Module):
    """
    Predicts (delta_scales, delta_means) for one prefix block.

    Input:  (B, Cin, H, W)   concat(hyperprior params, prefix-masked context params)
    Output: (B, 2*block_ch, H, W)
    """

    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = 384, depth: int = 3, attn: bool = True):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[DWResBlock(hidden_ch, dw_ks=3) for _ in range(int(depth))])
        self.attn = TinySpatialAttention(hidden_ch, num_heads=4) if bool(attn) else None
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, out_ch, 1),
        )

        # Zero-init only the final conv so deltas start at zero but gradients
        # still reach the rest of the network. (Zero-initializing every conv in
        # out_proj kills gradient flow to everything except the final bias.)
        final = self.out_proj[-1]
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        if self.attn is not None:
            x = self.attn(x)
        return self.out_proj(x)


def _refine_scales_softplus(scales: torch.Tensor, delta: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Refine positive scales in inverse-softplus space: s' = softplus(softplus^-1(s) + delta) + eps."""
    s = scales.clamp_min(eps)
    inv = torch.log(torch.expm1(s))
    return F.softplus(inv + delta) + eps


# -----------------------------
# Codec
# -----------------------------
class CoarseToFineCodec(nn.Module):
    """
    TIC backbone + prefix-masked context modeling + delta refinement networks.

    forward(x, mode) returns a dict:
      {
        "y_hat": full quantized latent,
        "z_likelihoods": ...,
        "coarse": {"x_hat", "y_likelihoods", "bpp"},
        "inter":  {...},
        "fine":   {...},
      }
    BPP per level is cumulative: y likelihoods of the first k channels plus a
    proportional (k / 320) share of the z bitrate.
    """

    def __init__(
        self,
        chunks: Dict[str, int],
        quality: int = 8,
        backbone: str = "tic",
        delta_enabled: bool = True,
        delta_hidden: int = 384,
        delta_depth: int = 3,
        delta_attn: bool = True,
        clamp_delta_scale: float = 0.5,
    ):
        super().__init__()
        self.chunk_config = build_chunk_config(chunks["coarse"], chunks["inter"], chunks["fine"])
        self.delta_enabled = bool(delta_enabled)
        self.clamp_delta_scale = float(clamp_delta_scale)

        self.codec = image_models[backbone](quality=quality)

        if self.delta_enabled:
            first = self.codec.entropy_parameters[0]
            if not isinstance(first, nn.Conv2d):
                raise RuntimeError("Could not infer delta-network input channels from entropy_parameters.")
            entropy_in_ch = int(first.in_channels)

            self.codec.level_ar_inter = DeltaNetwork(
                in_ch=entropy_in_ch,
                out_ch=2 * self.chunk_config["inter_chunk"],
                hidden_ch=delta_hidden,
                depth=delta_depth,
                attn=delta_attn,
            )
            self.codec.level_ar_fine = DeltaNetwork(
                in_ch=entropy_in_ch,
                out_ch=2 * self.chunk_config["fine_chunk"],
                hidden_ch=delta_hidden,
                depth=delta_depth,
                attn=delta_attn,
            )

    # ---- checkpoint I/O ----
    def load_checkpoint_state(self, state: Dict[str, torch.Tensor], strict: bool = False):
        """
        Load either a new-format ("codec."-prefixed) or legacy (flat TIC) state dict.

        Always delegates to the backbone's load_state_dict so its CDF-buffer
        resizing logic runs. Returns an object with missing_keys/unexpected_keys
        (the backbone's override returns None).
        """
        state = strip_and_upgrade_codec_state_keys(state)
        state = {
            (k[len("codec."):] if k.startswith("codec.") else k): v
            for k, v in state.items()
        }

        own_keys = set(self.codec.state_dict().keys())
        missing = sorted(own_keys - set(state.keys()))
        unexpected = sorted(set(state.keys()) - own_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Strict load failed: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

        state = {k: v for k, v in state.items() if k in own_keys}
        self.codec.load_state_dict(state, strict=False)
        self.update(force=True)

        class _LoadResult:
            def __init__(self, missing_keys, unexpected_keys):
                self.missing_keys = missing_keys
                self.unexpected_keys = unexpected_keys

        return _LoadResult(missing, unexpected)

    def update(self, force: bool = False):
        if hasattr(self.codec, "update"):
            self.codec.update(force=force)

    def aux_loss(self):
        return self.codec.aux_loss()

    # ---- forward ----
    def _prefix_mask(self, y_hat_full: torch.Tensor, k: int) -> torch.Tensor:
        y_mask = torch.zeros_like(y_hat_full)
        y_mask[:, :k, ...] = y_hat_full[:, :k, ...]
        return y_mask

    def _apply_delta(
        self,
        level: str,
        feat: torch.Tensor,
        scales_hat: torch.Tensor,
        means_hat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.chunk_config
        if level == "inter":
            net_delta = self.codec.level_ar_inter
            a, b = cfg["coarse_end"], cfg["inter_end"]
        elif level == "fine":
            net_delta = self.codec.level_ar_fine
            a, b = cfg["inter_end"], cfg["fine_end"]
        else:
            return scales_hat, means_hat

        delta = net_delta(feat)
        delta_s, delta_m = delta.chunk(2, dim=1)

        if self.clamp_delta_scale > 0:
            delta_s = self.clamp_delta_scale * torch.tanh(delta_s)

        scales_sub = _refine_scales_softplus(scales_hat[:, a:b, ...], delta_s)
        means_sub = means_hat[:, a:b, ...] + delta_m

        scales_hat = scales_hat.clone()
        means_hat = means_hat.clone()
        scales_hat[:, a:b, ...] = scales_sub
        means_hat[:, a:b, ...] = means_sub
        return scales_hat, means_hat

    @staticmethod
    def _bpp(
        y_likelihoods: torch.Tensor,
        z_likelihoods: torch.Tensor,
        num_pixels: float,
        k: int,
        total_k: int = TOTAL_LATENT_CHANNELS,
    ) -> torch.Tensor:
        sum_log_y = torch.log(y_likelihoods[:, :k, ...]).sum()
        # z bits are shared by all levels; attribute them proportionally to the prefix width
        sum_log_z = torch.log(z_likelihoods).sum() * (k / total_k)
        return (sum_log_y + sum_log_z) / (-math.log(2) * num_pixels)

    def forward(self, x: torch.Tensor, mode: str = "train") -> Dict[str, Any]:
        if mode not in ("train", "eval"):
            raise ValueError(f"Unknown mode: {mode}")
        quant_mode = "noise" if mode == "train" else "dequantize"

        core = self.codec
        x_size = (x.shape[2], x.shape[3])
        num_pixels = float(x.shape[0] * x.shape[2] * x.shape[3])

        y = core.g_a(x, x_size)
        z = core.h_a(y, x_size)
        z_hat, z_likelihoods = core.entropy_bottleneck(z)
        params = core.h_s(z_hat, x_size)

        y_hat_full = core.gaussian_conditional.quantize(y, quant_mode)

        out: Dict[str, Any] = {"y_hat": y_hat_full, "z_likelihoods": z_likelihoods}

        for level in LEVELS:
            k = self.chunk_config[f"{level}_end"]

            y_ctx = self._prefix_mask(y_hat_full, k)
            ctx_params = core.context_prediction(y_ctx)
            feat = torch.cat((params, ctx_params), dim=1)

            gaussian_params = core.entropy_parameters(feat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            if self.delta_enabled and level in ("inter", "fine"):
                scales_hat, means_hat = self._apply_delta(level, feat, scales_hat, means_hat)

            _, y_likelihoods = core.gaussian_conditional(y, scales_hat, means=means_hat)

            x_hat = core.g_s(self._prefix_mask(y_hat_full, k), x_size)

            out[level] = {
                "x_hat": x_hat,
                "y_likelihoods": y_likelihoods,
                "bpp": self._bpp(y_likelihoods, z_likelihoods, num_pixels, k),
            }

        return out


# -----------------------------
# Loss
# -----------------------------
class CoarseToFineLoss(nn.Module):
    """
    L = sum_k [ bpp_k + lambda_k * 255^2 * mse_w_k * MSE_k ] + sum_k gamma_k * Task_k

    Task loss per level (computed through a frozen classifier):
      - coarse: prob-sum NLL over K=10 clusters
      - inter:  prob-sum NLL over K=100 clusters
      - fine:   cross-entropy with label smoothing over 1000 classes
    """

    def __init__(
        self,
        classifier: nn.Module,
        meter,  # Hier3ProbSumMeter
        lmbda: Dict[str, float],
        ce_weight: Dict[str, float],
        mse_weight: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.1,
        log_acc: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()
        from utils import cls_preprocess_batch  # local import to avoid cycle at module load

        self.classifier = classifier
        self.meter = meter
        self.cls_preprocess = cls_preprocess_batch

        self.lmbda = {k: float(lmbda[k]) for k in LEVELS}
        self.ce_weight = {k: float(ce_weight[k]) for k in LEVELS}
        mse_weight = mse_weight or {"coarse": 0.33, "inter": 0.67, "fine": 1.0}
        self.mse_weight = {k: float(mse_weight[k]) for k in LEVELS}

        self.label_smoothing = float(label_smoothing)
        self.log_acc = bool(log_acc)
        self.eps = float(eps)
        self.mse = nn.MSELoss()

        self.f2c10 = meter.f2c10.clone().long().cpu()
        self.f2c100 = meter.f2c100.clone().long().cpu()
        self.K10 = int(meter.K10)
        self.K100 = int(meter.K100)

    def _prob_sum_nll(self, logits_fine, labels_fine, f2c_cpu, K):
        device = logits_fine.device
        B = logits_fine.size(0)

        f2c = f2c_cpu.to(device)
        tgt = f2c[labels_fine]

        probs = F.softmax(logits_fine, dim=1)
        pc = torch.zeros((B, K), device=device, dtype=probs.dtype)
        pc.index_add_(dim=1, index=f2c, source=probs)

        return -torch.log(pc.gather(1, tgt[:, None]).clamp_min(self.eps)).mean()

    def forward(self, out: Dict[str, Any], target: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        loss = 0.0

        for level in LEVELS:
            x_hat = out[level]["x_hat"]
            bpp = out[level]["bpp"]

            mse = self.mse(x_hat, target)
            rd = bpp + self.lmbda[level] * (255.0 ** 2) * mse * self.mse_weight[level]

            logits = self.classifier(self.cls_preprocess(x_hat))
            if level == "coarse":
                task = self._prob_sum_nll(logits, labels, self.f2c10, self.K10)
            elif level == "inter":
                task = self._prob_sum_nll(logits, labels, self.f2c100, self.K100)
            else:
                task = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

            loss = loss + rd + self.ce_weight[level] * task

            result[f"bpp_{level}"] = bpp
            result[f"mse_{level}"] = mse
            result[f"rd_{level}"] = rd
            result[f"ce_{level}"] = task

            if self.log_acc:
                acc = self.meter.compute(logits.detach(), labels.detach())
                for name, key in (
                    ("coarse10", "coarse10_top1"),
                    ("inter100", "inter100_top1"),
                    ("fine", "fine_top1"),
                ):
                    result[f"acc_{level}_{name}_top1"] = torch.tensor(acc[key], device=logits.device)

        result["loss"] = loss
        return result
