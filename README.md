# Mini‑CLEVR VQA – Baseline Experiments

This repo contains **two lightweight baselines** for the synthetic Mini‑CLEVR
dataset I generated for the lab take‑home project.

| Tag | Vision encoder | Text encoder | Fusion formula | Trainable params | 10 ep Val Acc | 15 ep Val Acc |
|-----|----------------|--------------|----------------|------------------|---------------|---------------|
| **clip‑lora‑auto** | ViT‑B/32 (CLIP) + LoRA *r = 8* | CLIP B/32 text | `cat ⊕ mix ⊕ |diff|` | **≈ 1 M** | 0.87 | **0.91** |
| **v2‑mix‑diff‑layer4** | ResNet‑18 (*layer4 optional*) | SBERT (all‑mpnet‑base‑v2) | `cat ⊕ mix ⊕ |diff|` | 2 M | 0.80 | **0.84** |

---

## Quick Start

```bash
# 1) create & activate env (see *.yml at bottom)
conda env create -f env_mini-clevr.yml
conda activate mini-clevr

# 2) dataset already generated under ./mini_clevr

# 3‑A) ResNet‑SBERT baseline (15 epochs, layer4 frozen)
python baseline_resnet_sbert_v2.py ^
  --data_dir ./mini_clevr ^
  --batch 128 --epochs 15 ^
  --fusion cat_mix_diff ^
  --wandb_run r18-sbert

# 3‑B) ResNet‑SBERT (+layer4 fine‑tune, lr 1e‑4)
python baseline_resnet_sbert_v2.py ^
  --data_dir ./mini_clevr ^
  --batch 128 --epochs 15 ^
  --fusion cat_mix_diff ^
  --unfreeze_layer4 ^
  --wandb_run r18-sbert-layer4

# 3‑C) CLIP + LoRA baseline
python baseline_clip_lora.py ^
  --data_dir ./mini_clevr ^
  --batch 128 --epochs 15 ^
  --fusion cat_mix_diff ^
  --wandb_run clip-lora-auto

# File Tree
.
├── baseline_resnet_sbert_v2.py      # ResNet‑18 + SBERT baseline
├── baseline_clip_lora.py            # ViT‑B/32 + LoRA baseline
├── data_gen/                        # Mini‑CLEVR image + JSONL generator
├── mini_clevr/                      # train/val images & *.jsonl & answer2idx.json
├── report/                          # project report (PDF / PPTX)
├── checkpoints/                     # saved *.pt weights
├── env_mini-clevr.yml               # env for dataset + ResNet‑SBERT baselines
└── env_clip.yml                     # env for CLIP + LoRA baseline

# Dataset Format
{"image": "train/img_00001.png",
 "question": "What colour is the pentagon?",
 "answer": "yellow"}




