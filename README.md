# Coarse-to-Fine: Progressive Image Compression for Semantically Hierarchical Classification

Official implementation of **"Coarse-to-Fine: Progressive Image Compression for Semantically Hierarchical Classification"**.

A progressive learned image codec for machines: the 320 latent channels of a TIC backbone are aligned with a semantic hierarchy so that the first **128** channels recover **coarse** semantics (K=10), the first **224** channels **intermediate** semantics (K=100), and the full **320** channels the **fine-grained** class (K=1000). Entropy parameters are recomputed per prefix with prefix-masked context modeling, and lightweight **Δ-networks** refine the entropy parameters of the intermediate and fine blocks.

```text
bitstream prefix      semantics            example
[0 : 128]      ───►   coarse   (K=10)      "animal"
[0 : 224]      ───►   inter    (K=100)     "dog"
[0 : 320]      ───►   fine     (K=1000)    "golden retriever"
```

## 1. Installation

```bash
conda create -n c2f python=3.10
conda activate c2f
git clone https://github.com/kjungwoo03/Coarse-to-Fine.git
cd Coarse-to-Fine

pip install -r requirements.txt
cd CompressAI && pip install -e . && cd ..
cd pytorch-image-models && pip install -e . && cd ..

# WordNet data for WUP (Wu-Palmer similarity) evaluation
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 2. Data preparation

Prepare ImageNet-1K in `ImageFolder` format:

```text
datasets/ImageNet/
  train/n01440764/*.JPEG ...
  val/n01440764/*.JPEG ...
```

The CLIP-clustered semantic hierarchy (K=10 / K=100 cluster assignments over the 1000 ImageNet classes) is already included in [hierarchy/](hierarchy/):

| file | content |
|---|---|
| `cluster_assignments_clip_K10.csv` | `wnid → cluster_id` mapping for the coarse level (K=10) |
| `cluster_assignments_clip_K100.csv` | `wnid → cluster_id` mapping for the intermediate level (K=100) |

## 3. Training

All hyperparameters live in [config.yaml](config.yaml). The defaults are the paper settings:

| group | key | default | meaning |
|---|---|---|---|
| model | `quality` | 8 | TIC quality (N=192, M=320 latent channels) |
| model | `chunks` | 128 / 96 / 96 | coarse / inter / fine block widths (must sum to 320) |
| model | `delta_network` | hidden 384, depth 3, attn on | Δ-network architecture (Fig. 3b) |
| loss | `lmbda` | 1e-4 / 1e-3 / 1e-2 | distortion weight λ per level |
| loss | `ce_weight` | 0.5 / 1.0 / 2.0 | task weight γ per level |
| loss | `label_smoothing` | 0.1 | label smoothing for the fine-level CE |
| train | `epochs` | 100 | |
| train | `batch_size` | 8 | **per GPU** |
| train | `learning_rate` / `aux_learning_rate` | 1e-4 / 1e-3 | main / entropy-bottleneck-quantiles optimizers |
| data | `train_subset` / `val_subset` | 80000 / 5000 | random subset per run (`<=0` uses the full split) |

The task loss is computed through a **frozen ImageNet-pretrained ResNet-50**: prob-sum NLL over the K=10 / K=100 clusters for the coarse / inter levels, and label-smoothed cross-entropy for the fine level.

```text
L = Σ_k [ bpp_k + λ_k · 255² · w_k · MSE_k ] + Σ_k γ_k · Task_k + aux_loss
```

### 3.1 Multi-GPU (DDP, recommended)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py --config config.yaml
```

### 3.2 Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml
```

### 3.3 Overriding the config from the command line

Any config entry can be overridden with dotted keys via `--opts` (no need to edit the YAML):

```bash
# shorter run, different fine-level lambda
python train.py --config config.yaml --opts train.epochs 50 loss.lmbda.fine 5e-3

# channel-allocation ablation (Table: coarse-heavy vs uniform vs ...)
python train.py --config config.yaml --opts \
  model.chunks.coarse 107 model.chunks.inter 107 model.chunks.fine 106

# Δ-network ablation (base codec, prefix-masked context only)
python train.py --config config.yaml --opts model.delta_network.enabled false

# disable wandb logging
python train.py --config config.yaml --opts wandb.enabled false
```

### 3.4 Resume / warm start

```bash
python train.py --config config.yaml --opts \
  train.checkpoint results/coarse2fine/<exp_name>/checkpoint.pth.tar
```

Optimizer / scheduler / epoch are restored when `train.resume_optim: true` (default). Legacy checkpoints (DDP `module.` prefixes, old `level_ar_224/320` module names) are upgraded automatically on load.

### 3.5 Outputs

Each run creates `results/coarse2fine/<exp_name>/` containing:

```text
config.json                  # resolved config used for this run
log.txt                      # training log
checkpoint.pth.tar           # latest (model + optimizers + scheduler + config)
checkpoint_epoch_XXX.pth.tar # per-epoch snapshots
checkpoint_best.pth.tar      # best validation loss
```

## 4. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --checkpoint results/coarse2fine/<exp_name>/checkpoint_best.pth.tar \
  --dataset datasets/ImageNet/val \
  --hierarchy_dir hierarchy \
  --save_dir results/eval
```

Reports, per semantic level (coarse / inter / fine):

- cumulative **bpp** of the prefix (z bits attributed proportionally)
- **PSNR / SSIM / MS-SSIM**
- hierarchical **top-1** at K=10 / K=100 / K=1000 (prob-sum over fine logits)
- **WUP** (Wu-Palmer) similarity of the fine prediction
- **top-1 / top-5** for multiple downstream classifiers

Useful options:

```bash
--classifiers resnet50.a1_in1k,convnext_base.fb_in1k,mobilenetv3_large_100
                      # any timm model names; the first is used for hierarchy/WUP
--num_images 1000     # quick evaluation on a subset
--disable_wup         # skip WUP (no NLTK needed)
--coarse_chunk 107 --inter_chunk 107 --fine_chunk 106   # manual chunk override
```

Chunk widths and Δ-network hyperparameters are resolved automatically from the checkpoint (embedded config → sidecar `config.json`/`args.json` → state-dict shapes), so a plain `--checkpoint` is usually all you need. Results are saved to `<save_dir>/results.csv`.

## 5. Repository layout

```text
model.py        CoarseToFineCodec (prefix-masked context + Δ-refinement),
                DeltaNetwork, CoarseToFineLoss, checkpoint/chunk-config utilities
train.py        Config-driven DDP training loop
eval.py         Per-level rate / accuracy / WUP evaluation
utils.py        Metrics, Hier3ProbSumMeter (prob-sum + WUP), DDP helpers
config.yaml     Paper hyperparameters
hierarchy/      CLIP K-means cluster assignments (K=10, K=100)
CompressAI/     Custom CompressAI fork (TIC backbone)
pytorch-image-models/  Local timm fork (downstream classifiers)
```

## 6. Implementation note: Δ-network initialization

`DeltaNetwork` zero-initializes **only the final** convolution of its output projection. Deltas therefore start at exactly zero (training stability), while gradients still reach every layer after the first step. Zero-initializing the entire output projection — an easy mistake — permanently blocks gradient flow to everything except the final bias, degenerating the Δ-network into a constant per-channel offset.

## License

See [LICENSE](LICENSE).
