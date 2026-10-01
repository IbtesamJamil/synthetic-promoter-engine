# Deep Generative Synthetic Promoter Engine (WGAN-GP)

An end-to-end deep learning pipeline designed to generate biologically plausible human promoter sequences (600 bp, [-499, +100] relative to Transcription Start Site) using Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP) in PyTorch.

---

## Key Features & Architecture

* **Generative Architecture**: 1D Transposed Convolutional Generator mapping a 100-D latent noise vector to (4, 600) one-hot sequence tensors.
* **Critic Architecture**: 1D Convolutional Critic with Group Normalization optimizing W1 Wasserstein Distance with Lipschitz continuity enforced via Gradient Penalty (\lambda = 10.0).
* **Dual Representation Tokenizer**: Converts clean FASTA sequences into both 4-channel One-Hot matrices for CNN/GAN training and 6-mer token tensors for Transformer downstream tasks.
* **Biological Validation Suite**: Evaluates synthetic outputs against real human baselines across GC-content bias, TATA-box density, and GC-box distribution.

---

## Project Structure

```text
synthetic-promoter-engine/
ÃÄÄ .gitignore
ÃÄÄ README.md
ÃÄÄ data/
³   ÃÄÄ clean_promoters.txt
³   ÃÄÄ generated_promoters.fasta
³   ÀÄÄ wgan_loss_curve.png
ÀÄÄ src/
    ÃÄÄ download_epd.py      # Dataset generation/ingestion
    ÃÄÄ parse_data.py        # FASTA parsing & quality control
    ÃÄÄ tokenize_data.py     # One-hot & 6-mer PyTorch tokenization
    ÃÄÄ model.py             # Generator & Critic architectures
    ÃÄÄ train.py             # WGAN-GP training loop & loss plotting
    ÀÄÄ evaluate.py          # Biological motif & GC metrics evaluation
```

---

## Biological Evaluation Summary

| Metric | Real Human Promoters (EPDnew) | Synthetic Promoters (Generated) |
| :--- | :--- | :--- |
| **Dataset Size** | 20,000 sequences | Sample Outputs |
| **Average GC%%** | 54.02%% | 54.62%% |
| **GC-Box Density** | 0.589 / seq | 0.500 / seq |
| **TATA-Box Density** | 0.341 / seq | Captured at scale |

---

## Quick Start

```bash
# 1. Activate Virtual Environment
env\Scripts\activate

# 2. Navigate to Source Directory
cd src

# 3. Execute Complete Pipeline
python download_epd.py
python parse_data.py
python tokenize_data.py
python train.py
python evaluate.py
```
