import os, sys, torch, tqdm, time
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import *
from DL_ClassifierModel import *
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from functools import reduce
import json

# Set random seed
SEED = 388014
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Data path
data_path = '/root/LncLocFormer/dataset/data.csv'

# Load dataset
print("Loading dataset...")
totalDS = lncRNA_loc_dataset(data_path, k=3, mode='csv')
tokenizer = Tokenizer(totalDS.sequences, totalDS.labels, seqMaxLen=8196, sequences_=totalDS.sequences_)

print(f"Dataset size: {len(totalDS)}")
print(f"Num labels: {tokenizer.labNum}")
print(f"Label classes: {list(tokenizer.lab2id.keys())}")

# Get label names
label_names = []
for i in range(tokenizer.labNum):
    for label, idx in tokenizer.lab2id.items():
        if idx == i:
            label_names.append(label)
            break

print(f"Label names: {label_names}")

# Transform the binary label vector into decimal int and use it to do the stratified split
tknedLabs = []
for lab in totalDS.labels:
    tmp = np.zeros((tokenizer.labNum))
    tmp[[tokenizer.lab2id[i] for i in lab]] = 1
    tknedLabs.append(reduce(lambda x, y: 2 * x + y, tmp) // 4)

# Get the test set (same split as during training)
for i, j in StratifiedShuffleSplit(test_size=0.1, random_state=SEED).split(range(len(tknedLabs)), tknedLabs):
    testIdx = j
    break

testIdx_ = set(testIdx)
restIdx = np.array([i for i in range(len(totalDS)) if i not in testIdx_])

print(f"Test size: {len(testIdx)}")
print(f"Train size: {len(restIdx)}")

# Cache the subsequence one-hot
totalDS.cache_tokenizedKgpSeqArr(tokenizer, groups=512)

# Build test dataset
testDS = torch.utils.data.Subset(totalDS, testIdx)

# Model with fixed enhancement factor
class KGPDPLAM_alpha_Mamba2_FixedEnhancement(KGPDPLAM_alpha_Mamba2):
    def __init__(self, classNum, tknEmbedding=None, tknNum=None,
                 embSize=64, dkEnhance=1, freeze=False,
                 L=4, H=256, A=4, maxRelativeDist=7,
                 embDropout=0.2, hdnDropout=0.15, paddingIdx=-100, 
                 tokenizer=None, fixed_enhancement_factor=2.6344):
        super().__init__(classNum, tknEmbedding, tknNum, embSize, dkEnhance, 
                       freeze, L, H, A, maxRelativeDist, embDropout, hdnDropout, 
                       paddingIdx, tokenizer)
        
        # Re-initialize attention module using fixed enhancement factor
        self.improved_deepPseudoLabelwiseAttn = Improved_DeepPseudoLabelwiseAttention(
            embSize, classNum, L=-1, hdnDropout=hdnDropout, dkEnhance=1,
            tokenizer=tokenizer, sequences=["TTT"], 
            enhanceFactor=fixed_enhancement_factor
        )
        
        # Freeze enhancement factor parameter to make it non-trainable
        self.improved_deepPseudoLabelwiseAttn.enhanceFactor.requires_grad = False
        
        # Compute corresponding raw parameter value
        raw_param_value = np.log((fixed_enhancement_factor - 1.0) / (3.0 - fixed_enhancement_factor))
        with torch.no_grad():
            self.improved_deepPseudoLabelwiseAttn.enhanceFactor.fill_(raw_param_value)

# Load trained model
print("\nLoading trained model...")
model_files = [f for f in os.listdir('./models/') if f.startswith('LncLocMamba_fixed_enhancement') and f.endswith('.pkl')]
if not model_files:
    print("Error: trained model file not found")
    sys.exit(1)

# Use the latest model file
model_file = sorted(model_files)[-1]
model_path = f"./models/{model_file}"
print(f"Using model file: {model_path}")

# Build model
backbone = KGPDPLAM_alpha_Mamba2_FixedEnhancement(
    tokenizer.labNum,
    tknEmbedding=None,
    tknNum=tokenizer.tknNum,
    embSize=256,
    dkEnhance=1,
    freeze=False,
    L=1,
    H=256,
    A=8,
    maxRelativeDist=25,
    embDropout=0.2,
    hdnDropout=0.1,
    paddingIdx=tokenizer.tkn2id['[PAD]'],
    tokenizer=tokenizer,
    fixed_enhancement_factor=2.6344
).to(device)

model = SequenceMultiLabelClassifier(backbone, collateFunc=PadAndTknizeCollateFunc(tokenizer), mode=0)

# Load model weights
try:
    import pickle
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # 尝试不同的加载方式
    if hasattr(model_data, 'state_dict'):
        model.model.load_state_dict(model_data.state_dict())
    elif isinstance(model_data, dict) and 'model' in model_data:
        if hasattr(model_data['model'], 'state_dict'):
            model.model.load_state_dict(model_data['model'].state_dict())
        else:
            model.model.load_state_dict(model_data['model'])
    else:
        model.model.load_state_dict(model_data)
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Trying to retrain the model...")
    sys.exit(1)

# Set model to eval mode
model.model.eval()

# Per-class detailed metrics
def calculate_per_class_metrics(y_true, y_prob, y_pred, label_names):
    """Compute per-class detailed metrics."""
    per_class_metrics = {}
    
    for i, label_name in enumerate(label_names):
        class_metrics = {}
        
        # Ground truth and predictions for this class
        y_true_class = y_true[:, i]
        y_prob_class = y_prob[:, i]
        y_pred_class = y_pred[:, i]
        
        # Metrics
        class_metrics['F1'] = f1_score(y_true_class, y_pred_class, zero_division=0)
        class_metrics['Precision'] = precision_score(y_true_class, y_pred_class, zero_division=0)
        class_metrics['Recall'] = recall_score(y_true_class, y_pred_class, zero_division=0)
        class_metrics['Accuracy'] = accuracy_score(y_true_class, y_pred_class)
        
        # AUC if both classes present
        if len(np.unique(y_true_class)) > 1:
            class_metrics['AUC'] = roc_auc_score(y_true_class, y_prob_class)
        else:
            class_metrics['AUC'] = 0.0
            
        per_class_metrics[label_name] = class_metrics
    
    return per_class_metrics

# Evaluate on test set
print("\nEvaluating model on test set...")
test_loader = torch.utils.data.DataLoader(
    testDS, 
    batch_size=16, 
    shuffle=False,
    collate_fn=model.collateFunc,
    num_workers=4,
    pin_memory=True
)

# 获取预测结果
results = model.calculate_y_prob_by_iterator(test_loader)
y_true = results['y_true']
y_prob = results['y_prob']
y_pred = (y_prob > 0.5).astype(int)

print(f"Num test samples: {y_true.shape[0]}")
print(f"Num classes: {y_true.shape[1]}")

# Overall metrics
from metrics import Metrictor
metrictor = Metrictor()
metrictor.set_data(results)

overall_metrics = metrictor([
    "AvgF1", 'MiF', 'MaF', "MaAUC", 'MiAUC', 
    'MiP', 'MaP', 'MiR', 'MaR', 'ACC'
])

print("\n=== Overall metrics on test set ===")
for key, value in overall_metrics.items():
    print(f"{key}: {value:.4f}")

# Per-class metrics
per_class_metrics = calculate_per_class_metrics(y_true, y_prob, y_pred, label_names)

print("\n=== Detailed metrics by localization class ===")
for label, metrics in per_class_metrics.items():
    print(f"\n{label}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# Results directory
results_dir = './test_set_results'
os.makedirs(results_dir, exist_ok=True)

# Viz 1: per-class metrics comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
metrics_to_plot = ['F1', 'Precision', 'Recall', 'Accuracy', 'AUC']

for idx, metric in enumerate(metrics_to_plot):
    row = idx // 3
    col = idx % 3
    
    values = [per_class_metrics[label][metric] for label in label_names]
    bars = axes[row, col].bar(range(len(label_names)), values, color='skyblue', alpha=0.7)
    axes[row, col].set_title(f'{metric} by Localization', fontsize=14)
    axes[row, col].set_xlabel('Localization')
    axes[row, col].set_ylabel(metric)
    axes[row, col].set_xticks(range(len(label_names)))
    axes[row, col].set_xticklabels(label_names, rotation=45, ha='right')
    axes[row, col].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)

# 隐藏最后一个子图
axes[1, 2].set_visible(False)

plt.suptitle('Per-Class Metrics on Test Set\n(Enhancement Factor: 2.6344)', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(f'{results_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Per-class metrics plot saved: {results_dir}/per_class_metrics.png")

# Viz 2: confusion matrices heatmaps
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

fig, axes = plt.subplots(1, len(label_names), figsize=(4*len(label_names), 4))
if len(label_names) == 1:
    axes = [axes]

cm_list = multilabel_confusion_matrix(y_true, y_pred)

for i, (label, cm) in enumerate(zip(label_names, cm_list)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{label}\nConfusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.suptitle('Confusion Matrices for Each Localization', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{results_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Confusion matrices saved: {results_dir}/confusion_matrices.png")

# Viz 3: prediction probability distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, label in enumerate(label_names[:4]):  # show up to 4 classes
    idx = label_names.index(label)
    
    # Positive and negative prediction probabilities
    positive_probs = y_prob[y_true[:, idx] == 1, idx]
    negative_probs = y_prob[y_true[:, idx] == 0, idx]
    
    axes[i].hist(positive_probs, bins=30, alpha=0.7, label=f'True {label}', color='red', density=True)
    axes[i].hist(negative_probs, bins=30, alpha=0.7, label=f'Non-{label}', color='blue', density=True)
    axes[i].set_title(f'Prediction Probability Distribution - {label}')
    axes[i].set_xlabel('Prediction Probability')
    axes[i].set_ylabel('Density')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Prediction Probability Distributions', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(f'{results_dir}/probability_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Probability distribution plots saved: {results_dir}/probability_distributions.png")

# Viz 4: sample analysis (a few representative samples)
print("\nGenerating sample analysis plots...")
sample_indices = np.random.choice(len(testDS), min(5, len(testDS)), replace=False)

for sample_idx, dataset_idx in enumerate(sample_indices):
    try:
        sample = testDS[dataset_idx]
        
        # Use DataLoader to build a proper batch
        sample_loader = torch.utils.data.DataLoader([sample], batch_size=1, collate_fn=model.collateFunc)
        batch_data = next(iter(sample_loader))
        
        # Move to device
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(device)
        
        with torch.no_grad():
            # Get predictions
            output = model.calculate_y_prob(batch_data)
            sample_y_prob = output['y_prob'][0].cpu().numpy()
            sample_y_true = batch_data['tokenizedLabArr'][0].cpu().numpy()
            
            # Visualization
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Prediction probability vs ground truth
            x_pos = np.arange(len(label_names))
            bars = axes[0].bar(x_pos, sample_y_prob, alpha=0.7, color='lightblue')
            
            # Mark true labels
            for i, (bar, true_label) in enumerate(zip(bars, sample_y_true)):
                if true_label == 1:
                    bar.set_color('red')
                    bar.set_alpha(0.8)
                    axes[0].text(i, bar.get_height() + 0.02, '✓', ha='center', va='bottom', 
                               fontsize=16, fontweight='bold', color='red')
            
            axes[0].set_title(f'Sample {sample_idx+1} - Prediction vs Ground Truth')
            axes[0].set_xlabel('Localization')
            axes[0].set_ylabel('Prediction Probability')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(label_names, rotation=45, ha='right')
            axes[0].set_ylim(0, 1.1)
            axes[0].grid(True, alpha=0.3)
            
            # Add prediction value labels
            for i, prob in enumerate(sample_y_prob):
                axes[0].text(i, prob + 0.05, f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Sequence length and enhancement factor
            seq_tokens = batch_data['tokenizedSeqArr'][0].cpu().numpy()
            valid_length = len(seq_tokens[seq_tokens != tokenizer.tkn2id['[PAD]']])
            enhancement_factor = backbone.improved_deepPseudoLabelwiseAttn.get_enhancement_factor()
            
            # Simulated attention weights (first 100 positions)
            position_weights = np.random.rand(min(100, valid_length))
            position_weights = position_weights / position_weights.sum()
            
            axes[1].plot(range(len(position_weights)), position_weights, marker='o', markersize=3, linewidth=1)
            axes[1].set_title(f'Simulated Attention Pattern (Enhancement Factor: {enhancement_factor:.4f})')
            axes[1].set_xlabel('Sequence Position')
            axes[1].set_ylabel('Attention Weight')
            axes[1].grid(True, alpha=0.3)
            
            # Compose true/pred label names
            true_labels = [label_names[i] for i, val in enumerate(sample_y_true) if val == 1]
            pred_labels = [label_names[i] for i, val in enumerate(sample_y_prob) if val > 0.5]
            
            plt.suptitle(f'Sample {sample_idx+1} Analysis\n' +
                        f'True: {", ".join(true_labels) if true_labels else "None"} | ' +
                        f'Predicted: {", ".join(pred_labels) if pred_labels else "None"}\n' +
                        f'Sequence Length: {valid_length}', 
                        fontsize=14, y=0.98)
            
            plt.tight_layout()
            plt.savefig(f'{results_dir}/sample_analysis_{sample_idx+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Sample {sample_idx+1} analysis plot saved: {results_dir}/sample_analysis_{sample_idx+1}.png")
            
    except Exception as e:
        print(f"Error when processing sample {sample_idx+1}: {e}")

# Save detailed results to JSON
detailed_results = {
    'overall_metrics': overall_metrics,
    'per_class_metrics': per_class_metrics,
    'enhancement_factor': 2.6344,
    'test_set_size': len(testDS),
    'label_names': label_names,
    'model_file': model_file
}

with open(f'{results_dir}/detailed_results.json', 'w') as f:
    json.dump(detailed_results, f, indent=2)

print(f"\nDetailed results saved to: {results_dir}/detailed_results.json")

# Summary report
print("\n=== Test set evaluation summary ===")
print(f"Dataset: {data_path}")
print(f"Test size: {len(testDS)}")
print(f"Model file: {model_file}")
print(f"Enhancement factor: 2.6344")

print("\nF1 scores by class (desc):")
f1_scores = [(label, per_class_metrics[label]['F1']) for label in label_names]
f1_scores.sort(key=lambda x: x[1], reverse=True)
for label, f1 in f1_scores:
    print(f"  {label}: {f1:.4f}")

print("\nAUC by class (desc):")
auc_scores = [(label, per_class_metrics[label]['AUC']) for label in label_names]
auc_scores.sort(key=lambda x: x[1], reverse=True)
for label, auc in auc_scores:
    print(f"  {label}: {auc:.4f}")

print(f"\nAll visualizations and results saved to: {results_dir}/")
print("Includes:")
print("  - per_class_metrics.png: per-class metrics comparison")
print("  - confusion_matrices.png: confusion matrices")
print("  - probability_distributions.png: probability distributions")
print("  - sample_analysis_*.png: sample analysis plots")
print("  - detailed_results.json: detailed results")