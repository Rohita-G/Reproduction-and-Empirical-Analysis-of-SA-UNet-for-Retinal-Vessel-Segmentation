# Reproduction and Empirical Analysis of SA-UNet for Retinal Vessel Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-DRIVE-orange.svg)](https://drive.grand-challenge.org/)
[![Paper DOI](https://img.shields.io/badge/IEEE%20Xplore-10.1109%2FICPR48806.2021.9413346-00629B.svg)](https://doi.org/10.1109/ICPR48806.2021.9413346)

---

## Project Overview

This repository contains the independent reproduction and empirical analysis of **SA-UNet (Spatial Attention U-Net)** for retinal vessel segmentation on the benchmark **DRIVE (Digital Retinal Images for Vessel Extraction)** dataset, developed for the **Machine Learning** course.

* **Target Paper:** C. Guo, M. Szemenyei, Y. Yi, W. Wang, B. Chen and C. Fan, *"SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation,"* **2020 25th International Conference on Pattern Recognition (ICPR)**, Milan, Italy, 2021, pp. 1236–1242.  
  * **DOI:** [10.1109/ICPR48806.2021.9413346](https://doi.org/10.1109/ICPR48806.2021.9413346)
  * **Original Author GitHub & Data:** [clguo/SA-UNet](https://github.com/clguo/SA-UNet)
* **Course:** Machine Learning
* **Authors & Team Contribution:**
  * **Rohita** (Team Member 1) — *50% Contribution*
  * **Samora** (Team Member 2) — *50% Contribution*

---

## Repository Structure

```text
├── README.md                                          # Reproduction guide, dependencies, and execution instructions
├── SA_UNet_Retinal_Vessel_Segmentation(drive).ipynb   # Fully annotated reproduction notebook (Cells 1–42)
└── SA_UNet_Retinal_Vessel_Segmentation_IEEE_Report.pdf# 3-page IEEE-format conference reproduction paper
```

---

## Environment Setup and Installing Dependencies

### 1. Prerequisites
* **Python:** $\ge 3.8$ (Python 3.10 or 3.11 recommended)
* **Hardware:** NVIDIA GPU with CUDA support (e.g., Tesla T4, V100, A100, or RTX 3060+) or Google Colab T4 GPU runtime.

### 2. Clone the Repository
```bash
git clone https://github.com/Rohita-G/Reproduction-and-Empirical-Analysis-of-SA-UNet-for-Retinal-Vessel-Segmentation.git
cd Reproduction-and-Empirical-Analysis-of-SA-UNet-for-Retinal-Vessel-Segmentation
```

### 3. Create a Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
Run the following command to install the required libraries:

```bash
# Core Deep Learning Framework
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Image Processing, Machine Learning & Visualization
pip install numpy matplotlib opencv-python pillow scikit-image scikit-learn albumentations tqdm jupyter
```

#### Key Package Requirements:
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `torch` | $\ge 1.10.0$ | Core neural network computation, tensor ops, backpropagation |
| `torchvision` | $\ge 0.11.0$ | Vision utilities and dataset transformations |
| `numpy` | $\ge 1.21.0$ | Numerical and array operations |
| `opencv-python` | $\ge 4.5.0$ | Image preprocessing (CLAHE, channel extraction) |
| `scikit-image` | $\ge 0.19.0$ | SSIM and morphology evaluations |
| `scikit-learn` | $\ge 1.0.0$ | ROC-AUC, confusion matrix, MCC calculation |
| `matplotlib` | $\ge 3.4.0$ | Loss curves, bar charts, and segmentation overlays |
| `albumentations`| $\ge 1.1.0$ | Spatial and affine data augmentations |

---

## Dataset Preparation (DRIVE)

The **DRIVE** dataset contains 40 color fundus photography images ($584 \times 565$ pixels), divided into 20 training images and 20 test images.

1. **Download:** The raw dataset can be downloaded directly from the author's repository:  
   [https://github.com/clguo/SA-UNet](https://github.com/clguo/SA-UNet)
2. **Directory Layout:** Place the downloaded dataset into the project root directory under `./DRIVE`:

```text
./DRIVE/
├── training/
│   ├── images/          # 20 training images (e.g., 21_training.tif ... 40_training.tif)
│   ├── 1st_manual/      # 20 ground-truth binary masks (e.g., 21_manual1.gif)
│   └── mask/            # 20 FOV masks (e.g., 21_training_mask.gif)
└── test/
    ├── images/          # 20 test images (e.g., 01_test.tif ... 20_test.tif)
    ├── 1st_manual/      # 20 ground-truth binary masks (e.g., 01_manual1.gif)
    └── mask/            # 20 FOV masks (e.g., 01_test_mask.gif)
```

3. **Preprocessing Pipeline (Automated in Notebook):**
   * **Green Channel Extraction:** Isolates the green spectral channel ($G$) offering maximum vascular contrast against the fundus background.
   * **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Normalizes illumination gradients and boosts fine capillary visibility.
   * **Symmetric Zero-Padding:** Pads native $584 \times 565$ images to canonical $592 \times 592$ pixels, ensuring exact divisibility by 16 across 4 pooling layers without fractional artifacts.
   * **Data Augmentation:** Generates 260 augmented patches via rotations ($90^\circ, 180^\circ, 270^\circ$), horizontal/vertical flips, and affine jitter.

---

## Running the Code

### Option A: Running on Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File** $\rightarrow$ **Upload notebook** and upload `SA_UNet_Retinal_Vessel_Segmentation(drive).ipynb`.
3. Switch runtime to GPU: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **Hardware accelerator** $\rightarrow$ **T4 GPU**.
4. Run all cells sequentially (**Runtime** $\rightarrow$ **Run all** or `Ctrl+F9`).

### Option B: Running Locally via Jupyter Notebook
Launch Jupyter Lab or Notebook in your terminal:
```bash
jupyter notebook "SA_UNet_Retinal_Vessel_Segmentation(drive).ipynb"
```

### Notebook Execution Flow (42 Cells):
* **Cells 1–8 (Initialization):** System verification, CUDA device mapping, random seeds (`seed=42`), directory creation, and dataset unzipping.
* **Cells 9–15 (Data Pipeline):** Green-channel isolation, CLAHE enhancement, $592 \times 592$ symmetric padding, and augmentation generator ($234$ train / $26$ val).
* **Cells 16–22 (Model Architecture):** Implementation of `DropBlock2D` ($7 \times 7$ block size, $\gamma=0.15$), `SpatialAttentionBlock` (bottleneck SAM), full `SA-UNet` ($537,299$ parameters), and `Plain U-Net` baseline ($537,201$ parameters).
* **Cells 23–27 (Loss & Optimizer):** Compound Binary Cross-Entropy + Dice loss function ($\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$) and Adam optimizer ($\text{lr} = 10^{-3}$).
* **Cells 28–33 (Training Engine):** 100-epoch training loop with automatic checkpoint serialization tracking peak validation Dice score.
* **Cells 34–38 (Quantitative Evaluation):** Metric computation over the 20 DRIVE test images (Accuracy, Sensitivity, Specificity, Dice, IoU, SSIM, AUC-ROC, MCC).
* **Cells 39–42 (Qualitative & Comparative Output):** Generation of publication-quality loss convergence plots, metric comparison bar charts, and full-resolution segmentation overlays.

---

## Quantitative Reproduction Results

Our reproduced implementation on the DRIVE benchmark validates the findings published by Guo et al. (ICPR 2020):

| Model / Source | Accuracy (ACC) | Sensitivity (SE) | Specificity (SP) | Dice / F1 | AUC-ROC | MCC | Total Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plain U-Net (Guo et al. 2020)** | 0.9666 | 0.7677 | **0.9857** | 0.8012 | 0.9789 | 0.7839 | 2,158,705 |
| **SA-UNet (Guo et al. 2020)** | **0.9698** | **0.8212** | 0.9840 | **0.8263** | **0.9864** | **0.8097** | 537,299 |
| **Plain U-Net (Our Baseline)** | 0.9668 | 0.7832 | 0.9833 | 0.7957 | 0.9783 | 0.7778 | **537,201** |
| **SA-UNet (Our Reproduction)** | **0.9703** | **0.8311** | 0.9828 | **0.8218** | **0.9866** | **0.8057** | **537,299** |

### Key Findings:
1. **Accuracy & Sensitivity Gains:** Our reproduced SA-UNet achieves **0.9703 Accuracy** (exceeding paper: $0.9698$) and **0.8311 Sensitivity** (exceeding paper: $0.8212$ by $+0.99\%$), confirming high recall on low-contrast micro-capillaries.
2. **Dice & MCC Consistency:** Dice score reaches **0.8218** (within $0.45\%$ of author's $0.8263$), and MCC reaches **0.8057** (within $0.40\%$ of author's $0.8097$).
3. **DropBlock & Attention Efficacy:** Compared to our plain U-Net baseline, SA-UNet delivers $\mathbf{+4.79\%}$ Sensitivity, $\mathbf{+2.61\%}$ Dice, and $\mathbf{+3.67\%}$ IoU improvement with negligible parameter addition (+98 weights).

---

## Citation

If you use this reproduction codebase or report, please cite the original ICPR 2020 paper:

```bibtex
@inproceedings{guo2020saunet,
  author    = {Changlu Guo and Márton Szemenyei and Yugen Yi and Wei Wang and Bailing Chen and Chuanbo Fan},
  title     = {SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation},
  booktitle = {2020 25th International Conference on Pattern Recognition (ICPR)},
  address   = {Milan, Italy},
  year      = {2021},
  pages     = {1236--1242},
  doi       = {10.1109/ICPR48806.2021.9413346}
}
```
