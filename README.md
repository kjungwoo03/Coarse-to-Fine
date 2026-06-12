# Coarse-to-Fine: Progressive Image Compression for Semantically Hierarchical Classification

Official repository of Coarse-to-Fine | [Paper Link](https://arxiv.org/abs/2605.08266).

[Jungwoo Kim](https://kjungwoo03.github.io), [Jun-Hyuk Kim](https://cai.cau.ac.kr/), [Jong-Seok Lee](https://mcml.yonsei.ac.kr/)  

## Summary
A progressive learned image codec for machines: the latent channels are aligned with a semantic hierarchy so that the first recovers **coarse** semantics (K=10), the second **intermediate** semantics (K=100), and the full channels the **fine-grained** class (K=1000). Entropy parameters are recomputed per prefix with prefix-masked context modeling, and lightweight **Δ-networks** refine the entropy parameters of the intermediate and fine blocks.

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


## 2. Training

All hyperparameters live in [config.yaml](config.yaml). 

### 3.1 Multi-GPU (DDP, recommended)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py --config config.yaml
```

### 3.2 Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml
```

## 4. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --checkpoint results/coarse2fine/<exp_name>/checkpoint_best.pth.tar \
  --dataset datasets/ImageNet/val \
  --hierarchy_dir hierarchy \
  --save_dir results/eval
```

## Citation

```
@inproceedings{kim2026coarse,
  title={Coarse-to-Fine: Progressive Image Compression for Semantically Hierarchical Classification},
  author={Kim, Jungwoo and Kim, Jun-Hyuk and Lee, Jong-Seok},
  pages={},
  booktitle={2026 IEEE International Conference on Image Processing (ICIP)}, 
  year={2026}
}
```

As a reference work, you can only cite:
```
@article{kim2025progressive,
  title={Progressive Learned Image Compression for Machine Perception},
  author={Kim, Jungwoo and Kim, Jun-Hyuk and Lee, Jong-Seok},
  journal={arXiv preprint arXiv:2512.20070},
  year={2025}
}
```
