<!-- SHOWCASE: false -->

# Adaptive Noise-Type Routing for CNN Denoising

> A noise-adaptive CNN framework that classifies image noise type and routes inputs through specialized denoising branches, evaluated against clean ground-truth images across multiple benchmark datasets.

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Language](https://img.shields.io/badge/language-Python-blue)
![Semester](https://img.shields.io/badge/semester-Fall_2025-orange)

---

## Course Information

| Field                  | Details                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Course Title           | Current Topics in Machine Learning                                                                                                                                                                                                                                                                                                                                                                                                       |
| Course Number          | CAP 6614                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Semester               | Fall 2025                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Assignment Title       | Final Group Project                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Assignment Description | Design and implement a novel approach or reproduce results from a current machine learning research paper. This project expands on adaptive denoising research by developing a noise-type-aware CNN that classifies the noise present in an input image and routes it through a specialized denoising branch, then compares this routing approach against a single comprehensive denoising model evaluated on clean ground-truth images. |

---

## Project Description

This project implements and compares two CNN-based image denoising strategies: a single comprehensive model trained across all noise types, and a routing model that first classifies the noise type then applies a noise-specific denoiser. Six synthetic noise types are supported (Gaussian, salt-and-pepper, uniform, Poisson, JPEG compression, and impulse), and each routing branch is architecturally tailored to its noise class. Models are trained and evaluated on MNIST, CIFAR-10, CIFAR-100, and STL-10 using a composite loss function combining MSE, MAE, SSIM, and gradient loss. The core research question is whether adaptive routing yields measurably better reconstruction quality than a single unified model when both are evaluated directly against clean ground-truth images.

---

## Screenshots / Demo

> _No screenshot available. Add one with: `![Demo](docs/your-image.png)`_

---

## Results

After training, each model is evaluated on a held-out test set and results are written to `./Results/`. The routing system additionally produces a confusion matrix for the noise classifier.

**Sample terminal output (comprehensive model, CIFAR-10):**

```
============================================================
FINAL TEST RESULTS
============================================================
Test Loss: 0.0321
Test PSNR: 31.05 dB
Test SSIM: 0.9858
Total Computation Time: 00:43:12
============================================================
```

**Sample terminal output (routing model, STL-10, per noise type):**

```
Noise Type    Loss    PSNR (dB)    SSIM
Gaussian      0.0400  29.54        0.9859
Salt & Pepper 0.0091  35.36        0.9974
Uniform       0.0240  32.84        0.9937
Poisson       0.0644  24.63        0.9599
JPEG          0.0448  27.95        0.9806
Impulse       0.0100  35.99        0.9971
Overall       0.0321  31.05        0.9858

Routing Accuracy: 99.66%
```

**How to interpret the output:**

- **Loss** is the composite loss (MSE 35% + MAE 35% + SSIM 20% + Gradient 10%). The project target is loss < 0.03 per noise type.
- **PSNR** (Peak Signal-to-Noise Ratio) is reported in dB; higher is better. Values above ~30 dB indicate good reconstruction quality.
- **SSIM** ranges from 0 to 1; values above 0.98 indicate near-perfect structural similarity to the clean image.
- **Routing Accuracy** reflects how often the classifier correctly identifies noise type before routing.

**Key files produced per run:**

```
./Results/
  Comprehensive/
    results_summary.csv         # Aggregated metrics across all datasets
    <dataset>/
      training_log.txt          # Epoch-by-epoch loss, PSNR, SSIM
      results.json              # Full results dictionary
      epoch_data.csv            # Per-epoch training/validation metrics
      visual_comparisons/       # Side-by-side clean / noisy / denoised PNGs
      images/                   # Individual clean and denoised PNGs
  Routing/
    results_summary.csv
    <dataset>/
      confusion_matrix.png      # Noise classifier confusion matrix
      training_log.txt
      results.json
      epoch_data.csv

./Models/Saved/
  comp_<dataset>.pth            # Saved comprehensive model weights
  classifier_<dataset>.pth      # Saved noise classifier weights
  <noisetype>_<dataset>.pth     # Saved routing branch weights
```

**If loss remains above 0.03:** check the training log for early stopping triggers and consider increasing `--epochs` or adjusting `--learning_rate`. The Gaussian, Poisson, and JPEG branches are the most likely to need additional tuning.

---

## Key Concepts

`image-denoising` `cnn` `adaptive-routing` `noise-classification` `composite-loss` `psnr` `ssim` `early-stopping` `multi-branch-architecture` `sobel-gradient-loss`

---

## Languages & Tools

- **Language:** Python 3.x
- **Framework/SDK:** PyTorch, torchvision
- **Build System:** pip / requirements.txt

---

## File Structure

```
project-root/
├── Main.py                         # Entry point; argument parsing, dataset prep, training orchestration
├── requirements.txt                # Python dependencies
│
├── Models/
│   ├── CompModel.py                # Comprehensive single-model denoiser architecture
│   ├── NoiseClassifier.py          # Noise type classifier (6-class CNN)
│   ├── Saved/                      # Saved .pth model weights (generated at runtime)
│   └── Routing_Models/
│       ├── GaussianModel.py        # Dilated convolution denoiser for Gaussian noise
│       ├── SaltPepperModel.py      # Spatial attention denoiser for salt-and-pepper noise
│       ├── UniformModel.py         # Wide-channel dropout denoiser for uniform noise
│       ├── PoissonModel.py         # Instance norm dual-branch denoiser for Poisson noise
│       ├── JPEGModel.py            # Deblocking denoiser for JPEG compression artifacts
│       └── ImpulseModel.py         # Corruption-mask denoiser for impulse noise
│
├── Utilities/
│   ├── Data_Loader.py              # AdaptiveDataset wrapper for MNIST/CIFAR-10/CIFAR-100/STL-10
│   ├── Noise_Generator.py          # On-the-fly noise synthesis for all 6 noise types
│   ├── Composite_Loss.py           # MSE + MAE + SSIM + Gradient composite loss function
│   ├── Comp_Train.py               # Training loop for the comprehensive model (PSNR/SSIM tracking)
│   └── Class_Train.py              # Training loop for the noise classifier (accuracy + confusion matrix)
│
├── Data/                           # Downloaded datasets (created at runtime)
├── Results/                        # Training logs, CSVs, PNGs (created at runtime)
│   ├── Comprehensive/
│   └── Routing/
└── Noise_Examples/                 # Example noise visualizations (generated optionally)
```

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended; CPU fallback is supported)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/UCF-CurrentTopicsML-AdaptiveNoiseCNN.git
cd UCF-CurrentTopicsML-AdaptiveNoiseCNN

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Train the comprehensive model on CIFAR-10 (default)
python Main.py --model comp --dataset cifar10

# 3b. Train the routing model (classifier + all noise-specific branches) on CIFAR-10
python Main.py --model routing --dataset cifar10

# 3c. Train on all datasets
python Main.py --model comp --dataset all
python Main.py --model routing --dataset all
```

### Controls

| Argument              | Default     | Description                                                        |
| --------------------- | ----------- | ------------------------------------------------------------------ |
| `--model`             | `comp`      | Model type: `comp` (comprehensive) or `routing` (adaptive routing) |
| `--dataset`           | `cifar10`   | Dataset: `mnist`, `cifar10`, `cifar100`, `stl10`, or `all`         |
| `--epochs`            | `50`        | Training epochs for denoiser models                                |
| `--classifier_epochs` | `30`        | Training epochs for the noise classifier                           |
| `--learning_rate`     | `0.001`     | Adam optimizer learning rate                                       |
| `--batch_size`        | `0`         | Batch size (0 = auto-adjust per dataset)                           |
| `--ESP`               | `10`        | Early stopping patience in epochs                                  |
| `--output_dir`        | `./Results` | Root directory for all saved outputs                               |
| `--data_root`         | `./Data`    | Root directory for dataset downloads                               |
| `--device`            | auto        | `cuda` or `cpu`                                                    |
| `--seed`              | `42`        | Random seed for reproducibility                                    |

---

## Contributors

| Name            | Role                                                | GitHub                                             |
| --------------- | --------------------------------------------------- | -------------------------------------------------- |
| Alexander Green | Organization, Development, Testing and Presentation | [@alexneilgreen](https://github.com/alexneilgreen) |

---

## Academic Integrity

This repository is publicly available for **portfolio and reference purposes only**.
Please do not submit any part of this work as your own for academic coursework.
