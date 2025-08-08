# LncLocMamba: Mamba-based Long Non-coding RNA Subcellular Localization Prediction Model

## 📖 Project Introduction

LncLocMamba is a deep learning model based on the Mamba architecture, specifically designed for predicting the subcellular localization of long non-coding RNAs (lncRNAs). This project combines the latest Mamba state space model with deep pseudo-label attention mechanisms, achieving excellent performance in lncRNA subcellular localization prediction tasks.

## 🚀 Core Features

- **Mamba Architecture**: Adopts the latest state space model with linear complexity and long sequence modeling capabilities
- **Multi-label Classification**: Supports simultaneous localization of one lncRNA to multiple subcellular locations
- **Attention Visualization**: Provides attention heatmaps and sequence-level attention analysis
- **Data Augmentation**: Integrates multiple data augmentation strategies to improve model generalization
- **Cross-validation**: Supports 5-fold cross-validation to ensure model stability
- **GPU Acceleration**: Supports CUDA acceleration for improved training and inference efficiency

## 🏗️ Project Architecture

### Core Modules

```
LncLocMamba/
├── dataset/                             # Dataset directory
│   ├── data.csv                        # Full dataset (curated from RNALocate)
│   ├── train_data.csv                  # Training split
│   ├── test_data.csv                   # Test split
│   └── filtered_data.csv               # Filtered subset (if generated)
├── DL_ClassifierModel.py               # Core model, losses, training helpers
├── nnLayer.py                          # Neural network layers / attention blocks
├── metrics.py                          # Metric utilities (AUC/F1/Precision/Recall/ACC ...)
├── utils.py                            # Data loading, k-mer tokenization, helpers
├── train.py                            # Standard training (cross-validation supported)
├── train_new_dataset.py                # Training pipeline for new datasets
├── train_fixed_enhancement.py          # (Optional) fixed enhancement factor experiments
├── prepare_new_dataset.py              # Convert/prepare custom dataset to required format
├── quick_evaluation.py                 # Quick evaluation & basic visualizations
├── evaluate_test_set.py                # Detailed evaluation on test set
├── evaluate_per_class.py               # Per-class metrics and plots (standard dataset)
├── evaluate_new_dataset_per_class.py   # Per-class metrics and plots (new dataset)
├── visualize_attention_heatmap.py      # Attention heatmap visualization
├── visualize_nucleus_attention.py      # Nucleus-specific attention visualization
├── demo_learnable_enhancement.py       # Demo for learnable attention enhancement factor
├── requirements.txt                    # Python dependencies
├── LICENSE                             # Project license (MIT)
├── README.md                           # Documentation
└── motif analysis                      # Motif analysis resources (if applicable)
```

### Model Architecture

1. **Embedding Layer**: Converts RNA sequences to vector representations
2. **Mamba Layer**: Uses state space model to process sequence information
3. **Deep Pseudo-label Attention**: Learns important positional information in sequences
4. **Classification Head**: Outputs multi-label prediction results

## 📦 Requirements

### Python Version
- Python 3.8+

### Installation
Use the provided requirements file (GPU environment recommended):
```bash
pip install -r requirements.txt
```
Key packages:
- PyTorch/torchvision, NumPy, pandas, scikit-learn
- Visualization: matplotlib, seaborn
- Training utilities: tqdm, einops, pytorch-lamb
- Mamba SSM ops: `causal-conv1d==1.0.0`, `mamba-ssm==1.0.1` (ensure CUDA compatibility)

## 🎯 Quick Start

### 1. Data Preparation

Ensure your data format is as follows.
Data source note: the default dataset `dataset/data.csv` is curated from RNALocate. If you use it, please cite RNALocate. See [RNALocate Download](http://www.rnalocate.org/download).
```csv
Gene_ID,Sequence,Subcellular_Localization
gene1,ATCGATCGATCG...,nucleus;cytoplasm
gene2,GCTAGCTAGCTA...,mitochondria
```

### 2. Data Preprocessing

```bash
python prepare_new_dataset.py
```

### 3. Model Training

```bash
# Train with new dataset
python train_new_dataset.py

# Or use standard training script
python train.py
```

### 4. Model Evaluation

```bash
# Quick evaluation
python quick_evaluation.py

# Detailed evaluation
python evaluate_test_set.py

# Evaluation by class
python evaluate_per_class.py
```

### 5. Attention Visualization

```bash
python visualize_attention_heatmap.py
```

## 🔧 Detailed Usage Guide

### Training Parameter Configuration

You can adjust the following parameters in `train_new_dataset.py`:

```python
# Model parameters
embSize = 256        # Embedding dimension
L = 1               # Number of Mamba layers
H = 256             # Hidden layer dimension
A = 8               # Number of attention heads
embDropout = 0.2    # Embedding layer dropout
hdnDropout = 0.1    # Hidden layer dropout

# Training parameters
batchSize = 16      # Batch size
epoch = 128         # Maximum training epochs
earlyStop = 64      # Early stopping epochs
lr = 3e-4          # Learning rate
```

### Data Format Description

- **Gene_ID**: Gene identifier
- **Sequence**: RNA sequence (bases like A, T, G, C)
- **Subcellular_Localization**: Subcellular localization labels (multiple labels separated by semicolons)

### Model Output

Model output includes the following information:
- Predicted subcellular localization probabilities
- Attention weights (for visualization)
- Various evaluation metrics (AUC, F1, precision, recall, etc.)

## 📊 Performance Metrics

Typical model performance on test set:
- **Macro Average AUC (MaAUC)**: >0.85
- **Micro Average AUC (MiAUC)**: >0.90
- **F1 Score**: >0.80
- **Precision**: >0.85
- **Recall**: >0.80

## 🎨 Visualization Features

### Attention Heatmaps
- Displays the model's attention to different sequence positions
- Supports visualization by category
- Provides comprehensive heatmap views

### Sequence-level Attention
- Shows attention distribution for individual sequences
- Highlights important regions
- Supports multi-category comparative analysis

## 🔍 Model Interpretation

### Attention Mechanism
The model uses deep pseudo-label attention mechanism to learn important positional information in sequences:
- Automatically identifies key sequence fragments
- Provides interpretable prediction results
- Supports biological significance analysis

### Mamba Architecture Advantages
- **Linear Complexity**: Compared to Transformer's quadratic complexity, Mamba has linear complexity
- **Long Sequence Modeling**: Can effectively process long RNA sequences
- **State Space Model**: Better captures long-term dependencies in sequences

## 🛠️ Advanced Features

### Data Augmentation
- Sequence truncation and padding
- Random masking
- Sequence transformation

### Adversarial Training
- FGM adversarial training
- Improves model robustness

### Model Ensemble
- Supports multi-model ensemble
- Cross-validation result fusion

## 📝 File Description

### Core Files
- `DL_ClassifierModel.py`: Contains all model definitions and training logic
- `utils.py`: Data processing and utility functions
- `train.py`: Standard training pipeline
- `train_new_dataset.py`: New dataset training pipeline

### Evaluation Files
- `quick_evaluation.py`: Quick model evaluation
- `evaluate_test_set.py`: Detailed test set evaluation
- `evaluate_per_class.py`: Performance evaluation by class

### Visualization Files
- `visualize_attention_heatmap.py`: Attention heatmap generation
- `visualize_nucleus_attention.py`: Nuclear localization attention analysis

## 🔁 Reproducibility

To reproduce main results:
```bash
pip install -r requirements.txt

# (Optional) set random seeds for determinism
export PYTHONHASHSEED=0
python - <<'PY'
import torch, random, numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
PY

# Prepare data (or use existing dataset/data.csv)
python prepare_new_dataset.py

# Train (standard dataset)
python train.py

# Or train with a new dataset
python train_new_dataset.py

# Evaluate
python quick_evaluation.py
python evaluate_test_set.py
python evaluate_per_class.py
```

### Using local Mamba source (optional)

This project depends on `mamba-ssm`. In most cases, installing from PyPI is sufficient and you do NOT need to version-control any local source tree.

If you have a local Mamba implementation at `/root/mamba` and want to use it (e.g., custom kernels or modifications), install it in editable mode:
```bash
pip install -e /root/mamba
```
Alternatively, ensure its Python path is visible:
```bash
export PYTHONPATH="/root/mamba:${PYTHONPATH}"
```
Only consider committing it as a Git submodule if you rely on your own fork and want to lock a specific revision. Otherwise, prefer the PyPI package `mamba-ssm` for simplicity.

## 📚 Dataset


- The default dataset file `dataset/data.csv` was curated from RNALocate downloads. Please refer to RNALocate's official download page for raw resources: [RNALocate Download](http://www.rnalocate.org/download).
- If you replace the dataset, ensure the columns follow the expected format described in the Quick Start section.

RNALocate provides data for non-commercial use conditioned on proper citation; see their Download & API page for details: [RNALocate Download](http://www.rnalocate.org/download).

## 🙏 Acknowledgements

- RNALocate database for subcellular localization data. Please cite RNALocate when using the curated dataset: [RNALocate Download](http://www.rnalocate.org/download).
- Mamba SSM implementation (`mamba-ssm`) used for state space modeling.

## 🤝 Contributing Guidelines

Welcome to submit Issues and Pull Requests to improve the project:

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Note**: This project is for academic research purposes only. Please do not use for commercial purposes. 