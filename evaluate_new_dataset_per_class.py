#!/usr/bin/env python3
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
import os
import numpy as np
import torch
import subprocess
import time
import json
import pickle

# Configure fonts for plots (ensure minus signs render correctly)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set fixed random seed
SEED = 388014
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_detailed_per_class_metrics(y_true, y_prob, y_pred, label_names=None):
    """Compute per-class detailed metrics."""
    if label_names is None:
        label_names = [f'Class_{i}' for i in range(y_true.shape[1])]
    
    per_class_metrics = {}
    
    # Ensure label name count matches prediction classes
    actual_num_classes = y_true.shape[1]
    if label_names and len(label_names) != actual_num_classes:
        print(f"Warning: number of label names ({len(label_names)}) does not match number of predicted classes ({actual_num_classes})")
        print("Using actual number of predicted classes")
        label_names = label_names[:actual_num_classes] if len(label_names) > actual_num_classes else label_names
    
    # Compute metrics for each class
    for i in range(actual_num_classes):
        class_metrics = {}
        
        # Get ground-truth and predictions for this class
        y_true_class = y_true[:, i]
        y_prob_class = y_prob[:, i]
        y_pred_class = y_pred[:, i]
        
        # Compute metrics
        class_metrics['ACC'] = accuracy_score(y_true_class, y_pred_class)
        class_metrics['MiP'] = precision_score(y_true_class, y_pred_class, zero_division=0)
        class_metrics['MiR'] = recall_score(y_true_class, y_pred_class, zero_division=0)
        class_metrics['Ave-F1'] = f1_score(y_true_class, y_pred_class, zero_division=0)
        class_metrics['MiF'] = f1_score(y_true_class, y_pred_class, zero_division=0)
        
        # Compute AUC if both classes present
        if len(np.unique(y_true_class)) > 1:
            class_metrics['MaAUC'] = roc_auc_score(y_true_class, y_prob_class)
        else:
            class_metrics['MaAUC'] = 0.0
        
        # Use provided label name or fallback
        label_name = label_names[i] if label_names and i < len(label_names) else f'Class_{i}'
        per_class_metrics[label_name] = class_metrics
    
    return per_class_metrics

def print_detailed_class_metrics(per_class_results, label_names, dataset_name=""):
    """Print detailed per-class metrics."""
    print(f"\n=== Detailed per-class metrics for {dataset_name} ===")
    print(f"{'Compartment':<25} {'ACC':<8} {'MiP':<8} {'MiR':<8} {'Ave-F1':<8} {'MiF':<8} {'MaAUC':<8}")
    print("-" * 85)
    
    for label in label_names:
        if label in per_class_results:
            metrics = per_class_results[label]
            print(f"{label:<25} "
                  f"{metrics.get('ACC', 0):<8.4f} "
                  f"{metrics.get('MiP', 0):<8.4f} "
                  f"{metrics.get('MiR', 0):<8.4f} "
                  f"{metrics.get('Ave-F1', 0):<8.4f} "
                  f"{metrics.get('MiF', 0):<8.4f} "
                  f"{metrics.get('MaAUC', 0):<8.4f}")

def plot_per_class_metrics(per_class_results, label_names, save_path=None):
    """Plot ACC bar chart for each class."""
    # Extract data
    x_labels = []
    acc_values = []
    
    for label in label_names:
        if label in per_class_results:
            x_labels.append(label)
            acc_values.append(per_class_results[label].get('ACC', 0))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Draw bars
    bars = plt.bar(range(len(x_labels)), acc_values, 
                   color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
    
    # Chart properties
    plt.xlabel('Subcellular Compartments', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Classification Accuracy for Each Subcellular Compartment', fontsize=16, fontweight='bold')
    
    # X ticks
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
    
    # Y axis limit
    plt.ylim(0, 1.05)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, acc_values)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Layout
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ACC bar chart saved to: {save_path}")
    
    plt.show()

def plot_detailed_comparison(per_class_results, label_names, save_dir=None):
    """Plot detailed comparison (bar charts and heatmap)."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1) Bar charts comparison
    metrics_to_plot = ['ACC', 'MiP', 'MiR', 'Ave-F1', 'MiF']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed performance analysis across subcellular compartments', fontsize=16, fontweight='bold')
    
    # Prepare data
    data_matrix = []
    for label in label_names:
        if label in per_class_results:
            row = [per_class_results[label].get(metric, 0) for metric in metrics_to_plot]
            data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//3, i%3]
        bars = ax.bar(range(len(label_names)), data_matrix[:, i], 
                     color=plt.cm.Set3(np.linspace(0, 1, len(label_names))))
        ax.set_title(f'{metric} by compartment', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove extra subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'detailed_bar_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Detailed bar charts saved to: {os.path.join(save_dir, 'detailed_bar_comparison.png')}")
    
    plt.show()
    
    # 2) Heatmap
    plt.figure(figsize=(10, 8))
    
    # Prepare heatmap data
    heatmap_data = pd.DataFrame(data_matrix.T, 
                               index=metrics_to_plot, 
                               columns=label_names)
    
    # Draw heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f',
                cbar_kws={'label': 'Metric value'}, square=True)
    
    plt.title('Heatmap of performance metrics by subcellular compartment', fontsize=14, fontweight='bold')
    plt.xlabel('Subcellular compartment', fontweight='bold')
    plt.ylabel('Performance metric', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'performance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Performance heatmap saved to: {os.path.join(save_dir, 'performance_heatmap.png')}")
    
    plt.show()

def evaluate_model(model, dataset, tokenizer, label_names=None):
    """Evaluate the model on the dataset and return per-class metrics and raw results."""
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False,
        collate_fn=model.collateFunc,
        num_workers=4,
        pin_memory=True
    )
    
    # Get predictions
    results = model.calculate_y_prob_by_iterator(dataloader)
    y_true = results['y_true']
    y_prob = results['y_prob']
    y_pred = (y_prob > 0.5).astype(int)
    
    # Compute per-class metrics
    per_class_metrics = calculate_detailed_per_class_metrics(y_true, y_prob, y_pred, label_names)
    
    return per_class_metrics, results

def lncRNA_loc_dataset_from_df(df, labels, num_classes, unique_locations, k=3):
    """Create a temporary dataset object from a pandas DataFrame."""
    
    # Original sequences
    sequences_ = df['Sequence'].tolist()
    
    # Convert into k-mer sequences
    tmp = ['-'*(k//2)+i+'-'*(k//2) for i in sequences_]
    sequences = [[i[j-k//2:j+k//2+1] for j in range(k//2,len(i)-k//2)] for i in tmp]
    
    # Gene IDs
    if 'Gene_ID' in df.columns:
        ids = df['Gene_ID'].tolist()
    else:
        ids = [f"seq_{i}" for i in range(len(sequences_))]
    
    # Sequence lengths
    sLens = [len(seq) for seq in sequences]
    
    # Convert one-hot labels to label name list
    label_names_list = []
    for label_vector in labels:
        label_names = []
        for i, val in enumerate(label_vector):
            if val == 1:
                label_names.append(unique_locations[i])
        label_names_list.append(label_names if label_names else ['Unknown'])
    
    # Dataset wrapper
    class TempDataset:
        def __init__(self, sequences, sequences_, labels, label_names_list, num_classes, ids, sLens):
            self.sequences = sequences
            self.sequences_ = sequences_  
            self.labels = labels
            self.label_names_list = label_names_list
            self.cla = num_classes
            self.ids = ids
            self.sLens = sLens
            
        def __len__(self):
            return len(self.sequences)
            
        def __getitem__(self, idx):
            return {
                'id': self.ids[idx],
                'sequence': self.sequences[idx],
                'sequence_': self.sequences_[idx],
                'sLen': self.sLens[idx],
                'label': self.label_names_list[idx],
                'tokenizedKgpSeqArr': self.tokenizedKgpSeqArr[idx] if hasattr(self, 'tokenizedKgpSeqArr') else None
            }
            
        def cache_tokenizedKgpSeqArr(self, tokenizer, groups=512):
            self.tokenizedKgpSeqArr = tokenizer.tokenize_sentences_to_k_group(
                self.sequences, groups
            )
    
    return TempDataset(sequences, sequences_, labels, label_names_list, num_classes, ids, sLens)

def main():
    # Check model and label mapping files
    model_dir = './models/new_dataset'
    model_path = os.path.join(model_dir, 'LncLocMamba_new_model_99999999.000.pkl')
    label_mapping_path = os.path.join(model_dir, 'label_mapping.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        print("Please run train_new_dataset.py to train the model first")
        return
    
    if not os.path.exists(label_mapping_path):
        print(f"Error: label mapping file not found: {label_mapping_path}")
        return
    
    # Load label mapping
    with open(label_mapping_path, 'rb') as f:
        label_info = pickle.load(f)
    
    unique_locations = label_info['unique_locations']
    label_to_idx = label_info['label_to_idx']
    num_classes = label_info['num_classes']
    
    print(f"Loaded label mapping: {num_classes} classes")
    for i, loc in enumerate(unique_locations):
        print(f"  {i}: {loc}")
    
    # Load test data
    test_file = "./dataset/test_data.csv"
    if not os.path.exists(test_file):
        print(f"Error: test data file not found: {test_file}")
        return
    
    print(f"\nLoading test data: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"Test set size: {len(test_df)}")
    
    # Prepare test labels
    test_labels = []
    for _, row in test_df.iterrows():
        location = row['Subcellular_Localization']
        if location in label_to_idx:
            label_vector = [0] * len(unique_locations)
            label_vector[label_to_idx[location]] = 1
            test_labels.append(label_vector)
        else:
            # If test set contains unseen classes, use zero vector
            label_vector = [0] * len(unique_locations)
            test_labels.append(label_vector)
    
    # Create test dataset
    print("\nCreating test dataset...")
    testDS = lncRNA_loc_dataset_from_df(test_df, test_labels, num_classes, unique_locations, k=3)
    
    # Create tokenizer (from test data)
    print("Creating tokenizer...")
    tokenizer = Tokenizer(testDS.sequences, testDS.label_names_list, seqMaxLen=8196, sequences_=testDS.sequences_)
    
    # Cache data
    print("Caching test data...")
    testDS.cache_tokenizedKgpSeqArr(tokenizer, groups=512)
    
    # Create model
    print("Creating model...")
    backbone = KGPDPLAM_alpha_Mamba2(
        classNum=testDS.cla,
        tknNum=tokenizer.tknNum,
        embSize=256,
        embDropout=0.1,
        L=1,
        paddingIdx=0,
        tokenizer=tokenizer
    ).to(device)
    
    # Initialize enhance factor if missing
    if not hasattr(backbone.improved_deepPseudoLabelwiseAttn, 'enhanceFactor'):
        backbone.improved_deepPseudoLabelwiseAttn.register_parameter(
            'enhanceFactor', 
            nn.Parameter(torch.tensor(1.5, dtype=torch.float32))
        )
    
    model = SequenceMultiLabelClassifier(backbone, collateFunc=PadAndTknizeCollateFunc(tokenizer), mode=0)
    
    # Load model weights
    if os.path.exists(model_path):
        try:
            model.load(model_path)
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Trying strict loading mode...")
            try:
                # Strict loading (ignore missing params)
                checkpoint = torch.load(model_path, map_location=device)
                if 'model' in checkpoint:
                    # Load only matching params
                    model_dict = model.model.state_dict()
                    pretrained_dict = checkpoint['model']
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    model.model.load_state_dict(model_dict, strict=False)
                    print("Model partially loaded (non-strict)")
                else:
                    print("Invalid model file format")
                    return
            except Exception as e2:
                print(f"Model loading failed completely: {e2}")
                return
    else:
        print(f"Error: model file not found: {model_path}")
        return
    
    # Evaluate model
    print("\nStarting evaluation...")
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    per_class_metrics, results = evaluate_model(model, testDS, tokenizer, unique_locations)
    print_detailed_class_metrics(per_class_metrics, unique_locations, "Test set")
    
    # Results directory
    results_dir = os.path.join(model_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    results_path = os.path.join(results_dir, 'per_class_metrics.json')
    serializable_metrics = {}
    for class_name, metrics in per_class_metrics.items():
        serializable_metrics[class_name] = {k: float(v) for k, v in metrics.items()}
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Use actual label names (aligned with predictions)
    actual_label_names = list(per_class_metrics.keys())
    
    # ACC bar chart
    plot_path = os.path.join(results_dir, 'accuracy_bar_chart.png')
    plot_per_class_metrics(per_class_metrics, actual_label_names, plot_path)
    
    # Report
    print("\n=== Performance Analysis Report ===")
    
    # Find best and worst compartments
    avg_scores = {}
    for location in unique_locations:
        if location in per_class_metrics:
            metrics = per_class_metrics[location]
            avg_score = np.mean([metrics['ACC'], metrics['MiP'], metrics['MiR'], metrics['Ave-F1']])
            avg_scores[location] = avg_score
    
    if avg_scores:
        best_location = max(avg_scores, key=avg_scores.get)
        worst_location = min(avg_scores, key=avg_scores.get)
        
        print(f"Best subcellular compartment: {best_location} (avg score: {avg_scores[best_location]:.4f})")
        print(f"Worst subcellular compartment: {worst_location} (avg score: {avg_scores[worst_location]:.4f})")
        
        # Overall metric analysis
        overall_metrics = {metric: [] for metric in ['ACC', 'MiP', 'MiR', 'Ave-F1', 'MiF']}
        for location in unique_locations:
            if location in per_class_metrics:
                for metric in overall_metrics:
                    overall_metrics[metric].append(per_class_metrics[location][metric])
        
        print(f"\nOverall metric means:")
        for metric, values in overall_metrics.items():
            if values:
                print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

if __name__ == "__main__":
    main() 