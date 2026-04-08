# Art Restoration with Deep Learning

> Can AI help restore damaged artwork? We benchmark three deep learning architectures on a paired dataset of damaged and undamaged paintings to find out.

---

## Overview

Many historical paintings suffer from cracks, stains, discoloration, and missing sections. Traditional physical restoration is time-consuming, expensive, and risky. This project explores **virtual restoration using deep learning** as a non-invasive alternative.

We implement and compare three model architectures — **U-Net**, **Pix2Pix**, and **SMA-GAN** — trained on paired damaged/undamaged artwork tiles, and evaluate them using both quantitative metrics and qualitative visual inspection.

## Dataset

We use the [Damaged and Undamaged Artworks](https://www.kaggle.com/datasets/pes1ug22am047/damaged-and-undamaged-artworks) dataset from Kaggle (116 paired images).

**Preprocessing pipeline:**
- Spatial alignment of damaged/undamaged pairs via content-based cropping
- Padding to ensure dimensions are multiples of the tile size
- Decomposition into non-overlapping square tiles (64×64, 128×128, 256×256)
- Deterministic pairing using artwork ID + tile coordinates

## Models

| Model | Architecture | Key Feature |
|-------|-------------|-------------|
| **U-Net** | Encoder-decoder with skip connections | Preserves spatial structure; uses L1 loss |
| **Pix2Pix** | GAN with U-Net Generator + PatchGAN Discriminator | Adversarial training produces sharper textures |
| **SMA-GAN** | GAN with Self-Attention blocks + ResBlocks | Captures long-range dependencies across the image |

## Results

| Model | PSNR (dB) | SSIM | Quality |
|-------|-----------|------|---------|
| **Pix2Pix (128×128)** | **28.63** | **0.8921** | ✅ Best — acceptable to near-good |
| U-Net | 22.40 | 0.7354 | Structurally accurate but blurry |
| Pix2Pix (256×256) | 17.51 | 0.6230 | Degrades at higher resolution |
| SMA-GAN | 12.19 | 0.31 | Insufficient data for attention to converge |

**Key findings:**
- **Pix2Pix (128×128)** delivered the best balance of structural fidelity and texture realism
- **U-Net** produces smooth, geometrically accurate restorations but lacks high-frequency detail due to pixel-wise loss
- **SMA-GAN** underperformed — self-attention mechanisms need significantly more data to learn spatial relationships from scratch, unlike CNNs which have a built-in inductive bias for local structure

## Evaluation Metrics

| Metric | What it measures | Weak | Acceptable | Good |
|--------|-----------------|------|------------|------|
| **PSNR** | Pixel-level fidelity | < 25 | 25–30 | > 30 |
| **SSIM** | Structural similarity (luminance, contrast, structure) | < 0.80 | 0.80–0.90 | > 0.90 |

## Tech Stack

`Python` · `PyTorch` · `NumPy` · `Matplotlib`

## Team

- **Marta Shkreli**
- Naël Arnoux
- Matteo Couchoud

MSc in Data Science & Business Analytics — ESSEC & CentraleSupélec (2025–2026)
