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

# Class index to name mapping
class_idx_to_name = {
    0: "Nucleus",
    1: "Cytoplasm",
    2: "Chromatin",
    3: "Insoluble Cytoplasm"
}

# Per-class metrics computation and TTT attention visualization
def calculate_detailed_per_class_metrics(y_true, y_prob, y_pred, label_names=None):
    """
    Compute detailed per-class metrics, including Ave-F1, MiP, MiR, MiF, MaAUC
    """
    if label_names is None:
        label_names = [f'Class_{i}' for i in range(y_true.shape[1])]
    
    per_class_metrics = {}
    
    # Metrics import
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    
    # Compute metrics per class
    for i, label_name in enumerate(label_names):
        class_metrics = {}
        
        # Ground truth and predictions for this class
        y_true_class = y_true[:, i]
        y_prob_class = y_prob[:, i]
        y_pred_class = y_pred[:, i]
        
        # Metrics
        class_metrics['Ave-F1'] = f1_score(y_true_class, y_pred_class, zero_division=0)  # Average F1
        class_metrics['MiP'] = precision_score(y_true_class, y_pred_class, zero_division=0)  # Micro Precision
        class_metrics['MiR'] = recall_score(y_true_class, y_pred_class, zero_division=0)     # Micro Recall
        class_metrics['MiF'] = f1_score(y_true_class, y_pred_class, zero_division=0)        # Micro F1
        
        # Compute AUC when both classes present
        if len(np.unique(y_true_class)) > 1:
            class_metrics['MaAUC'] = roc_auc_score(y_true_class, y_prob_class)  # Macro AUC
        else:
            class_metrics['MaAUC'] = 0.0
            
        per_class_metrics[label_name] = class_metrics
    
    return per_class_metrics

def visualize_ttt_attention(model, dataset, tokenizer, num_examples=3, save_dir='./ttt_attention_plots'):
    """
    Visualize attention weights around TTT motif positions
    """
    device = next(model.model.parameters()).device
    model.model.eval()
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get token id for TTT
    ttt_token_id = None
    if 'TTT' in tokenizer.tkn2id:
        ttt_token_id = tokenizer.tkn2id['TTT']
    else:
        print("Warning: TTT token not found in tokenizer")
        # Try finding single T token
        if 'T' in tokenizer.tkn2id:
            t_token_id = tokenizer.tkn2id['T']
            print(f"Using T token instead, T token ID: {t_token_id}")
            ttt_token_id = t_token_id
        else:
            print("T token not found either, creating example visualization")
            create_example_attention_plot(save_dir)
            return
    
    print(f"TTT token ID: {ttt_token_id}")
    
    # Find samples containing TTT from raw sequences
    ttt_samples = []
    for idx in range(min(len(dataset), 100)):  # broaden search range
        sample = dataset[idx]
        original_sequence = sample['sequence_']  # raw sequence string
        
        # Search TTT motif in raw sequence
        ttt_positions = []
        for i in range(len(original_sequence) - 2):
            if original_sequence[i:i+3] == 'TTT':
                ttt_positions.append(i)
        
        if len(ttt_positions) > 0:
            # Create temporary collate to force tokenizedSeqArr
            temp_collate = PadAndTknizeCollateFunc(tokenizer, groups=-1)  # groups=-1 forces tokenizedSeqArr
            temp_loader = torch.utils.data.DataLoader([sample], batch_size=1, collate_fn=temp_collate)
            batch_data = next(iter(temp_loader))
            
            # Ensure tokenizedSeqArr exists
            if batch_data['tokenizedSeqArr'] is not None:
                ttt_samples.append((idx, ttt_positions, batch_data, original_sequence))
                if len(ttt_samples) >= num_examples:
                    break
    
    print(f"Found {len(ttt_samples)} samples containing TTT motif")
    
    if len(ttt_samples) == 0:
        print("No samples with TTT motif found, creating example visualization...")
        create_example_attention_plot(save_dir)
        return
    
    for sample_idx, (orig_idx, ttt_positions, batch_data, original_sequence) in enumerate(ttt_samples):
        try:
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
            
            with torch.no_grad():
                # Get model predictions and attention
                output = model.model(batch_data)
                y_prob = output['y_logit'].sigmoid().cpu().numpy()[0]
                attn_weights = output.get('attn_weights', None)
                
                if attn_weights is None:
                    print("Warning: model did not return attention weights")
                    continue
                
                # Normalize attention container
                if isinstance(attn_weights, list):
                    attn_weights = attn_weights[0]
                
                # Attention per class
                attn_weights = attn_weights.cpu().numpy()  # (seq_len, num_classes)
                
                # True labels and sequence tokens
                true_labels = batch_data['tokenizedLabArr'][0].cpu().numpy()
                seq_tokens = batch_data['tokenizedSeqArr'][0].cpu().numpy()
                
                # Build figure
                num_classes = len(class_idx_to_name)
                fig = plt.figure(figsize=(15, 5 * (num_classes + 2)))
                gs = plt.GridSpec(num_classes + 2, 1, figure=fig)
                
                # 1) Prediction probabilities
                ax_prob = fig.add_subplot(gs[0])
                bars = ax_prob.bar(range(len(y_prob)), y_prob)
                ax_prob.set_title('Prediction Probability Distribution')
                ax_prob.set_xticks(range(len(y_prob)))
                ax_prob.set_xticklabels([class_idx_to_name[i] for i in range(len(y_prob))], rotation=45)
                
                # Mark true labels
                for i, (bar, true_label) in enumerate(zip(bars, true_labels)):
                    if true_label == 1:
                        bar.set_color('red')
                        bar.set_alpha(0.8)
                    else:
                        bar.set_color('blue')
                        bar.set_alpha(0.6)
                
                # 2) Sequence overview
                ax_seq = fig.add_subplot(gs[1])
                valid_tokens = seq_tokens[seq_tokens != tokenizer.tkn2id['[PAD]']]
                seq_length = len(valid_tokens)
                
                # Position markers highlighting TTT
                position_markers = np.zeros(seq_length)
                for pos in ttt_positions:
                    if pos < seq_length:
                        position_markers[pos] = 1
                
                ax_seq.bar(range(seq_length), position_markers, alpha=0.7, color='red', label='TTT position')
                ax_seq.set_title('TTT Positions in Sequence')
                ax_seq.set_xlabel('Sequence Position')
                ax_seq.set_ylabel('TTT Marker')
                ax_seq.legend()
                ax_seq.grid(True, alpha=0.3)
                
                # 3) Attention distribution per class
                for class_idx in range(num_classes):
                    ax = fig.add_subplot(gs[2 + class_idx])
                    class_name = class_idx_to_name[class_idx]
                    
                    # Attention weights for this class
                    class_attn = attn_weights[:, class_idx]
                    
                    # Plot
                    ax.plot(range(len(class_attn)), class_attn, 'b-', alpha=0.6, label='Attention Weight')
                    
                    # Mark TTT positions
                    for pos in ttt_positions:
                        if pos < len(class_attn):
                            ax.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
                            ax.plot(pos, class_attn[pos], 'ro', label='TTT position' if pos == ttt_positions[0] else "")
                    
                    ax.set_title(f'Attention Distribution - {class_name}')
                    ax.set_xlabel('Sequence Position')
                    ax.set_ylabel('Attention Weight')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Figure suptitle
                active_labels = [class_idx_to_name[i] for i, label in enumerate(true_labels) if label == 1]
                plt.suptitle(f'Sample {sample_idx+1} Attention Analysis\n'
                           f'True labels: {", ".join(active_labels) if active_labels else "None"}\n'
                           f'Predicted: {class_idx_to_name[np.argmax(y_prob)]} (prob: {np.max(y_prob):.4f})\n'
                           f'TTT positions: {list(ttt_positions)}',
                           fontsize=14, y=0.98)
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/attention_sample_{sample_idx+1}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Attention analysis plot saved: {save_dir}/attention_sample_{sample_idx+1}.png")
                
                # Save detailed data
                analysis_data = {
                    'original_index': orig_idx,
                    'predictions': y_prob.tolist(),
                    'true_labels': true_labels.tolist(),
                    'class_names': list(class_idx_to_name.values()),
                    'ttt_positions': ttt_positions.tolist(),
                    'sequence_length': seq_length,
                    'attention_weights': {
                        class_idx_to_name[i]: attn_weights[:, i].tolist()
                        for i in range(num_classes)
                    }
                }
                
                with open(f'{save_dir}/attention_analysis_sample_{sample_idx+1}.json', 'w') as f:
                    json.dump(analysis_data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing sample {sample_idx+1}: {e}")
            import traceback
            traceback.print_exc()

def create_example_attention_plot(save_dir):
    """Create an example attention plot"""
    # Create example data
    num_classes = len(class_idx_to_name)
    seq_length = 100
    ttt_positions = [25, 60, 85]
    
    # Build figure
    fig = plt.figure(figsize=(15, 5 * (num_classes + 2)))
    gs = plt.GridSpec(num_classes + 2, 1, figure=fig)
    
    # 1) Example prediction probabilities
    ax_prob = fig.add_subplot(gs[0])
    probs = [0.7, 0.2, 0.08, 0.02]
    bars = ax_prob.bar(range(len(probs)), probs)
    ax_prob.set_title('Example Prediction Probability Distribution')
    ax_prob.set_xticks(range(len(probs)))
    ax_prob.set_xticklabels(class_idx_to_name.values(), rotation=45)
    
    # 2) Example sequence overview
    ax_seq = fig.add_subplot(gs[1])
    position_markers = np.zeros(seq_length)
    for pos in ttt_positions:
        position_markers[pos] = 1
    
    ax_seq.bar(range(seq_length), position_markers, alpha=0.7, color='red', label='TTT position')
    ax_seq.set_title('TTT Positions in Example Sequence')
    ax_seq.set_xlabel('Sequence Position')
    ax_seq.set_ylabel('TTT Marker')
    ax_seq.legend()
    ax_seq.grid(True, alpha=0.3)
    
    # 3) Example attention per class
    for class_idx in range(num_classes):
        ax = fig.add_subplot(gs[2 + class_idx])
        class_name = class_idx_to_name[class_idx]
        
        # Simulate attention
        x = np.arange(seq_length)
        attn = np.exp(-0.01 * (x - 50) ** 2)  # Gaussian
        attn = attn / attn.sum()  # normalize
        
        # Enhance attention at TTT positions
        for pos in ttt_positions:
            attn[pos] *= 2.0
        attn = attn / attn.sum()  # re-normalize
        
        # Plot
        ax.plot(x, attn, 'b-', alpha=0.6, label='Attention Weight')
        
        # Mark TTT positions
        for pos in ttt_positions:
            ax.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
            ax.plot(pos, attn[pos], 'ro', label='TTT position' if pos == ttt_positions[0] else "")
        
        ax.set_title(f'Example Attention - {class_name}')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Attention Weight')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Example Attention Analysis\n(Generated because no sample with TTT was found)',
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_example.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Example attention plot saved: {save_dir}/attention_example.png")

def print_detailed_class_metrics(per_class_results, label_names, dataset_name=""):
    """
    Print detailed per-class metrics
    """
    print(f"\n=== Detailed Per-Class Metrics for {dataset_name} ===")
    print(f"{'Class':<15} {'Ave-F1':<8} {'MiP':<8} {'MiR':<8} {'MiF':<8} {'MaAUC':<8}")
    print("-" * 70)
    
    for label in label_names:
        if label in per_class_results:
            metrics = per_class_results[label]
            # If metrics are scalar (not a list), print directly
            if isinstance(metrics.get('Ave-F1', 0), (int, float)):
                print(f"{label:<15} "
                      f"{metrics.get('Ave-F1', 0):<8.4f} "
                      f"{metrics.get('MiP', 0):<8.4f} "
                      f"{metrics.get('MiR', 0):<8.4f} "
                      f"{metrics.get('MiF', 0):<8.4f} "
                      f"{metrics.get('MaAUC', 0):<8.4f}")
            else:
                # If list, compute the mean
                ave_f1_values = metrics.get('Ave-F1', [])
                mip_values = metrics.get('MiP', [])
                mir_values = metrics.get('MiR', [])
                mif_values = metrics.get('MiF', [])
                maauc_values = metrics.get('MaAUC', [])
                
                if ave_f1_values:
                    print(f"{label:<15} "
                          f"{np.mean(ave_f1_values):<8.4f} "
                          f"{np.mean(mip_values):<8.4f} "
                          f"{np.mean(mir_values):<8.4f} "
                          f"{np.mean(mif_values):<8.4f} "
                          f"{np.mean(maauc_values):<8.4f}")
                
    print("-" * 70)

def enhanced_evaluate_model_with_detailed_metrics(model, dataset, tokenizer, label_names=None):
    """
    Enhanced evaluation: return detailed per-class metrics
    """
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
    
    # Compute overall metrics
    from metrics import Metrictor
    metrictor = Metrictor()
    metrictor.set_data(results)
    
    overall_metrics = metrictor([
        "AvgF1", 'MiF', 'MaF', "MaAUC", 'MiAUC', 
        'MiP', 'MaP', 'MiR', 'MaR', 'ACC'
    ])
    
    # Compute per-class detailed metrics
    per_class_metrics = calculate_detailed_per_class_metrics(y_true, y_prob, y_pred, label_names)
    
    return overall_metrics, per_class_metrics, results


def get_free_gpu():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    gpu_info = result.stdout.splitlines()
    gpu_memory = [(int(info.split(',')[0]), int(info.split(',')[1])) for info in gpu_info]
    free_gpu = sorted(gpu_memory, key=lambda x: x[1], reverse=True)[0][0]
    return free_gpu


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Fixed enhancement factor
FIXED_ENHANCEMENT_FACTOR = 2.6344

SEED = 388014
seed = SEED
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

data_path = '/root/LncLocFormer/dataset/data.csv'

# Load dataset
totalDS = lncRNA_loc_dataset(data_path, k=3, mode='csv')
tokenizer = Tokenizer(totalDS.sequences, totalDS.labels, seqMaxLen=8196, sequences_=totalDS.sequences_)

# Transform the binary label vector into decimal int and use it to do the stratified split
tknedLabs = []
for lab in totalDS.labels:
    tmp = np.zeros((tokenizer.labNum))
    tmp[[tokenizer.lab2id[i] for i in lab]] = 1
    tknedLabs.append(reduce(lambda x, y: 2 * x + y, tmp) // 4)

# Get the test set
for i, j in StratifiedShuffleSplit(test_size=0.1, random_state=SEED).split(range(len(tknedLabs)), tknedLabs):
    testIdx = j
    break
testIdx_ = set(testIdx)
restIdx = np.array([i for i in range(len(totalDS)) if i not in testIdx_])

# Cache the subsequence one-hot
totalDS.cache_tokenizedKgpSeqArr(tokenizer, groups=512)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # 5-fold cross-validation

# Initialize a dictionary to store metrics for each fold
cv_results = {
    "LOSS": [],
    "AvgF1": [],
    "MiF": [],
    "MaF": [],
    "MaAUC": [],
    "MiAUC": [],
    "MiP": [],
    "MaP": [],
    "MiR": [],
    "MaR": [],
    "ACC": []  # Add accuracy metric
}

print(f"\n=== 5-Fold CV with Fixed Enhancement Factor {FIXED_ENHANCEMENT_FACTOR} ===")

# 5-fold cross-validation
for i, (trainIdx, validIdx) in enumerate(skf.split(restIdx, np.array(tknedLabs)[restIdx])):
    print(f"\n=== Starting Fold {i+1}/5 Cross-validation ===")
    
    # Define save path dynamically within the loop
    save_path = f'./models/LncLocMamba_fixed_enhancement_model'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Get the training set and validation set of fold i
    trainIdx, validIdx = restIdx[trainIdx], restIdx[validIdx]
    trainDS, validDS, testDS = (
        torch.utils.data.Subset(totalDS, trainIdx),
        torch.utils.data.Subset(totalDS, validIdx),
        torch.utils.data.Subset(totalDS, testIdx),
    )

    # Create model class with fixed enhancement factor
    class KGPDPLAM_alpha_Mamba2_FixedEnhancement(KGPDPLAM_alpha_Mamba2):
        def __init__(self, classNum, tknEmbedding=None, tknNum=None,
                     embSize=64, dkEnhance=1, freeze=False,
                     L=4, H=256, A=4, maxRelativeDist=7,
                     embDropout=0.2, hdnDropout=0.15, paddingIdx=-100, 
                     tokenizer=None, fixed_enhancement_factor=2.6344):
            super().__init__(classNum, tknEmbedding, tknNum, embSize, dkEnhance, 
                           freeze, L, H, A, maxRelativeDist, embDropout, hdnDropout, 
                           paddingIdx, tokenizer)
            
            # Reinitialize attention with fixed enhancement factor
            self.improved_deepPseudoLabelwiseAttn = Improved_DeepPseudoLabelwiseAttention(
                embSize, classNum, L=-1, hdnDropout=hdnDropout, dkEnhance=1,
                tokenizer=tokenizer, sequences=["TTT"], 
                enhanceFactor=fixed_enhancement_factor
            )
            
            # Freeze enhancement factor parameter
            self.improved_deepPseudoLabelwiseAttn.enhanceFactor.requires_grad = False
            
            # Compute corresponding raw parameter value
            # enhance_factor = sigmoid(raw_param) * 2.0 + 1.0
            # raw_param = sigmoid_inverse((enhance_factor - 1.0) / 2.0)
            raw_param_value = np.log((fixed_enhancement_factor - 1.0) / (3.0 - fixed_enhancement_factor))
            with torch.no_grad():
                self.improved_deepPseudoLabelwiseAttn.enhanceFactor.fill_(raw_param_value)
            
            print(f"Set fixed enhancement factor: {fixed_enhancement_factor:.4f}")
            print(f"Raw param value: {raw_param_value:.4f}")
            print(f"Verified enhancement factor: {self.improved_deepPseudoLabelwiseAttn.get_enhancement_factor():.4f}")

    # Use backbone with fixed enhancement factor
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
        fixed_enhancement_factor=FIXED_ENHANCEMENT_FACTOR
    ).to(device)

    model = SequenceMultiLabelClassifier(backbone, collateFunc=PadAndTknizeCollateFunc(tokenizer), mode=0)

    # Set the learning rate and weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.001,
        },
        {
            'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-4, weight_decay=0.001)

    # Record training start time
    start_time = time.time()
    
    # Train the model for fold i
    fold_results = model.train(
        optimizer=optimizer,
        trainDataSet=trainDS,
        validDataSet=validDS,
        otherDataSet=testDS,
        batchSize=16,
        epoch=2,  # use 2 epochs for testing
        earlyStop=64,
        saveRounds=1,
        isHigherBetter=True,
        metrics="MaAUC",
        report=[
            "LOSS", "AvgF1", 'MiF', 'MaF', "LOSS", "MaAUC", 'MiAUC', 
            'MiP', 'MaP', 'MiR', 'MaR', 'ACC', "EachAUC", "EachAUPR"
        ],
        savePath=save_path,
        shuffle=True,
        dataLoadNumWorkers=4,
        pinMemory=True,
        warmupEpochs=4,
        doEvalTrain=True,  # evaluate train performance
        doEvalOther=True,  # evaluate test performance
        prefetchFactor=2,
    )

    # Compute training time
    train_duration = time.time() - start_time
    print(f"Fold {i+1} training completed, time taken: {train_duration:.2f} seconds")
    
    # Verify enhancement factor remains fixed
    final_enhancement_factor = backbone.improved_deepPseudoLabelwiseAttn.get_enhancement_factor()
    print(f"Enhancement factor after training: {final_enhancement_factor:.4f} (should remain {FIXED_ENHANCEMENT_FACTOR:.4f})")
    
    if abs(final_enhancement_factor - FIXED_ENHANCEMENT_FACTOR) > 0.001:
        print("⚠️  Warning: enhancement factor changed, please check freezing setup")
    else:
        print("✓ Enhancement factor remained fixed")
    
    # Generate attention visualizations
    print("\nGenerating attention visualizations...")
    try:
        # Create save directory
        save_dir = f'./ttt_attention_plots/fold_{i+1}'
        os.makedirs(save_dir, exist_ok=True)
        
        # Train-set attention plots
        print("Generating train-set attention plots...")
        visualize_ttt_attention(model, trainDS, tokenizer, num_examples=2, 
                       save_dir=f'{save_dir}/train')
        
        # Test-set attention plots
        print("Generating test-set attention plots...")
        visualize_ttt_attention(model, testDS, tokenizer, num_examples=2, 
                       save_dir=f'{save_dir}/test')
    except Exception as e:
        print(f"Attention visualization failed: {e}")
        import traceback
        traceback.print_exc()

    # Append fold results to cv_results
    for key in cv_results:
        if key in fold_results:
            cv_results[key].append(fold_results[key])

print(f"\n=== Final results of 5-fold CV with fixed enhancement {FIXED_ENHANCEMENT_FACTOR} ===")
print(f"Dataset: {data_path}")
print(f"Num train samples: {len(restIdx)}")
print(f"Num test samples: {len(testIdx)}")
print(f"Num labels: {tokenizer.labNum}")

# Store train and test results
train_results = {}
test_results = {}

# Calculate and print average metrics across all folds
print("\n=== Validation metrics (CV mean) ===")
for key, values in cv_results.items():
    if values:  # ensure non-empty
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"{key}: {avg_value:.4f} ± {std_value:.4f}")
        
# Get label names
label_names = []
for i in range(tokenizer.labNum):
    for label, idx in tokenizer.lab2id.items():
        if idx == i:
            label_names.append(label)
            break

# Store per-class detailed metrics
train_per_class_results = {label: {'Ave-F1': [], 'MiP': [], 'MiR': [], 'MiF': [], 'MaAUC': []} 
                          for label in label_names}
test_per_class_results = {label: {'Ave-F1': [], 'MiP': [], 'MiR': [], 'MiF': [], 'MaAUC': []} 
                         for label in label_names}

# Collect per-fold train and test results
for i, (trainIdx, validIdx) in enumerate(skf.split(restIdx, np.array(tknedLabs)[restIdx])):
    print(f"\n=== Detailed metrics for Fold {i+1} ===")
    
    # Load the saved best model
    save_path = f'./models/LncLocMamba_fixed_enhancement_model'
    model_path = f"{save_path}.pt"
    
    # If model file missing, check for fold-specific filenames
    if not os.path.exists(model_path):
        alt_model_path = f"{save_path}_fold_{i+1}.pt"
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
        else:
            # Search current directory for possible model files
            model_files = [f for f in os.listdir('./models/') if f.startswith('LncLocMamba') and f.endswith('.pt')]
            if model_files:
                model_path = f"./models/{model_files[0]}"  # use the first found model file
                print(f"Using found model file: {model_path}")
            else:
                print(f"Warning: no model files found")
                continue
    
    # Create model instance
    trainIdx, validIdx = restIdx[trainIdx], restIdx[validIdx]
    trainDS, validDS, testDS = (
        torch.utils.data.Subset(totalDS, trainIdx),
        torch.utils.data.Subset(totalDS, validIdx),
        torch.utils.data.Subset(totalDS, testIdx),
    )
    
    # Use model with fixed enhancement factor
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
        fixed_enhancement_factor=FIXED_ENHANCEMENT_FACTOR
    ).to(device)
    
    model = SequenceMultiLabelClassifier(backbone, collateFunc=PadAndTknizeCollateFunc(tokenizer), mode=0)
    
    try:
        # Try to load model
        if os.path.exists(model_path):
            model.load(model_path)
            print(f"Loaded model: {model_path}")
            
            # Evaluate on train split (enhanced)
            print("\nTrain metrics:")
            train_overall, train_per_class, train_raw = enhanced_evaluate_model_with_detailed_metrics(
                model, trainDS, tokenizer, label_names
            )
            
            # Store overall metrics
            for key, value in train_overall.items():
                if key not in train_results:
                    train_results[key] = []
                train_results[key].append(value)
                print(f"  {key}: {value:.4f}")
            
            # Store per-class metrics and print details
            print_detailed_class_metrics(train_per_class, label_names, "Train")
            for label, metrics in train_per_class.items():
                for metric_name, value in metrics.items():
                    if metric_name in train_per_class_results[label]:
                        train_per_class_results[label][metric_name].append(value)
            
            # Evaluate on test split (enhanced)
            print("\nTest metrics:")
            test_overall, test_per_class, test_raw = enhanced_evaluate_model_with_detailed_metrics(
                model, testDS, tokenizer, label_names
            )
            
            # Store overall metrics
            for key, value in test_overall.items():
                if key not in test_results:
                    test_results[key] = []
                test_results[key].append(value)
                print(f"  {key}: {value:.4f}")
            
            # Store per-class metrics and print details
            print_detailed_class_metrics(test_per_class, label_names, "Test")
            for label, metrics in test_per_class.items():
                for metric_name, value in metrics.items():
                    if metric_name in test_per_class_results[label]:
                        test_per_class_results[label][metric_name].append(value)
            
            # Generate attention visualization on first fold only
            if i == 0:
                print("\nGenerating attention visualizations...")
                try:
                    visualize_ttt_attention(model, testDS, tokenizer, num_examples=3, 
                                      save_dir=f'./ttt_attention_plots_fold_{i+1}')
                except Exception as e:
                    print(f"Attention visualization failed: {e}")
                    
        else:
            print(f"Warning: model file does not exist {model_path}")
    except Exception as e:
        print(f"Error evaluating model: {e}")

# Print train averages
if train_results:
    print("\n=== Train Metrics (5-fold mean) ===")
    for key, values in train_results.items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"{key}: {avg_value:.4f} ± {std_value:.4f}")

# Print test averages
if test_results:
    print("\n=== Test Metrics (5-fold mean) ===")
    for key, values in test_results.items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"{key}: {avg_value:.4f} ± {std_value:.4f}")

# Print train per-class averages
if train_per_class_results:
    print_detailed_class_metrics(train_per_class_results, label_names, "Train")

# Print test per-class averages
if test_per_class_results:
    print_detailed_class_metrics(test_per_class_results, label_names, "Test")

# Summary tables
print("\n=== Metrics Summary Tables ===")
print("Train per-class Ave-F1:")
for label in label_names:
    values = train_per_class_results[label]['Ave-F1']
    if values:
        avg_value = np.mean(values)
        print(f"{label}: {avg_value:.4f}")

print("\nTest per-class Ave-F1:")
for label in label_names:
    values = test_per_class_results[label]['Ave-F1']
    if values:
        avg_value = np.mean(values)
        print(f"{label}: {avg_value:.4f}")

print("\nTest per-class MaAUC:")
for label in label_names:
    values = test_per_class_results[label]['MaAUC']
    if values:
        avg_value = np.mean(values)
        print(f"{label}: {avg_value:.4f}")

print(f"\n=== Detailed Results per Fold ===")
for i in range(5):
    print(f"\nFold {i+1}:")
    for key in cv_results:
        if i < len(cv_results[key]):
            print(f"  {key}: {cv_results[key][i]:.4f}")

print(f"\n=== Experiment Summary ===")
print(f"Enhancement factor: fixed {FIXED_ENHANCEMENT_FACTOR}")
print(f"Cross-validation: 5-fold stratified")
print(f"Random seed: {SEED}")
print(f"Model: KGPDPLAM_alpha_Mamba2")
print(f"Primary metric: MaAUC = {np.mean(cv_results['MaAUC']):.4f} ± {np.std(cv_results['MaAUC']):.4f}") 

# Helper: per-class metrics function
def calculate_per_class_metrics(y_true, y_prob, y_pred, label_names=None):
    """
    Compute detailed metrics for each class
    """
    if label_names is None:
        label_names = [f'Class_{i}' for i in range(y_true.shape[1])]
    
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
        
        # Compute AUC when both classes present
        if len(np.unique(y_true_class)) > 1:
            class_metrics['AUC'] = roc_auc_score(y_true_class, y_prob_class)
        else:
            class_metrics['AUC'] = 0.0
            
        per_class_metrics[label_name] = class_metrics
    
    return per_class_metrics

# Helper: attention visualization function
def visualize_attention(model, dataset, tokenizer, num_examples=5, save_dir='./attention_plots'):
    """
    Visualize attention weights to show how the model focuses on different sequence regions
    """
    device = next(model.model.parameters()).device
    model.model.eval()
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Randomly choose several samples
    indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
    
    for idx_num, idx in enumerate(indices):
        try:
            sample = dataset[idx]
            
            # Use DataLoader to get the correct batch format
            sample_loader = torch.utils.data.DataLoader([sample], batch_size=1, collate_fn=model.collateFunc)
            batch_data = next(iter(sample_loader))
            
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
            
            with torch.no_grad():
                # Get model predictions
                output = model.calculate_y_prob(batch_data)
                y_prob = output['y_prob'][0].cpu().numpy()  # first-sample prediction probs
                
                # Get ground-truth labels
                true_labels = batch_data['tokenizedLabArr'][0].cpu().numpy()
                
                # Create figure for visualization
                fig, axes = plt.subplots(3, 1, figsize=(15, 12))
                
                # 1) Prediction probabilities
                class_names = []
                for i in range(len(y_prob)):
                    class_name = f'Class_{i}'
                    for lab, lab_idx in tokenizer.lab2id.items():
                        if lab_idx == i:
                            class_name = lab
                            break
                    class_names.append(class_name)
                
                bars = axes[0].bar(range(len(y_prob)), y_prob)
                axes[0].set_title(f'Prediction Probabilities - Sample {idx_num+1}')
                axes[0].set_xlabel('Classes')
                axes[0].set_ylabel('Probability')
                axes[0].set_xticks(range(len(y_prob)))
                axes[0].set_xticklabels(class_names, rotation=45)
                
                # Mark true labels
                for i, (bar, true_label) in enumerate(zip(bars, true_labels)):
                    if true_label == 1:
                        bar.set_color('red')
                        bar.set_alpha(0.8)
                    else:
                        bar.set_color('blue')
                        bar.set_alpha(0.6)
                
                # 2) Sequence info and token distribution
                seq_tokens = batch_data['tokenizedSeqArr'][0].cpu().numpy()
                valid_tokens = seq_tokens[seq_tokens != tokenizer.tkn2id['[PAD]']]
                
                # Simple attention weight simulation (based on prediction probabilities)
                # Create a position-based simple attention pattern
                seq_length = len(valid_tokens)
                position_weights = np.exp(-np.abs(np.arange(seq_length) - seq_length//2) / (seq_length/4))
                position_weights = position_weights / position_weights.sum()
                
                axes[1].plot(range(seq_length), position_weights, marker='o', linewidth=2, markersize=4)
                axes[1].set_title('Position-based Attention Pattern (Simulated)')
                axes[1].set_xlabel('Sequence Position')
                axes[1].set_ylabel('Attention Weight')
                axes[1].grid(True, alpha=0.3)
                
                # 3) Show enhancement factor effect
                enhancement_factor = model.model.backbone.enhanceFactor.item()
                
                # Simulate enhancement effect
                base_attention = np.random.rand(seq_length)
                enhanced_attention = base_attention * enhancement_factor
                enhanced_attention = enhanced_attention / enhanced_attention.sum()
                
                axes[2].bar(range(seq_length), base_attention / base_attention.sum(), 
                           alpha=0.6, label='Base Attention', color='lightblue')
                axes[2].bar(range(seq_length), enhanced_attention, 
                           alpha=0.8, label='Enhanced Attention', color='darkblue')
                axes[2].set_title(f'Enhancement Factor Effect (Factor: {enhancement_factor:.4f})')
                axes[2].set_xlabel('Sequence Position')
                axes[2].set_ylabel('Attention Weight')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                # Get true label names
                active_labels = []
                for i, label in enumerate(true_labels):
                    if label == 1:
                        label_name = class_names[i]
                        active_labels.append(label_name)
                
                # Set figure title
                plt.suptitle(f'Sample {idx_num+1} Analysis\n' +
                           f'True Labels: {", ".join(active_labels) if active_labels else "None"}\n' +
                           f'Predicted: {class_names[np.argmax(y_prob)]} (prob: {np.max(y_prob):.4f})\n' +
                           f'Enhancement Factor: {enhancement_factor:.4f}', 
                           fontsize=14, y=0.98)
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/attention_sample_{idx_num+1}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Attention analysis plot saved: {save_dir}/attention_sample_{idx_num+1}.png")
                
                # Save prediction result data
                sample_data = {
                    'predictions': y_prob,
                    'true_labels': true_labels,
                    'class_names': class_names,
                    'enhancement_factor': enhancement_factor,
                    'sequence_length': seq_length
                }
                
                with open(f'{save_dir}/sample_{idx_num+1}_data.json', 'w') as f:
                    import json
                    json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sample_data.items()}, f, indent=2)
                
                print(f"Sample data saved: {save_dir}/sample_{idx_num+1}_data.json")
                
        except Exception as e:
            print(f"Error processing sample {idx_num+1}: {e}")
            import traceback
            traceback.print_exc()
            
            # Create error info figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f'Sample {idx_num+1}\nError:\n{str(e)}', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.savefig(f'{save_dir}/attention_sample_{idx_num+1}_error.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

# Enhanced evaluation function
def enhanced_evaluate_model(model, dataset, tokenizer, label_names=None):
    """
    Enhanced evaluation function returning overall and per-class metrics
    """
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
    
    # Compute overall metrics
    from metrics import Metrictor
    metrictor = Metrictor()
    metrictor.set_data(results)
    
    overall_metrics = metrictor([
        "AvgF1", 'MiF', 'MaF', "MaAUC", 'MiAUC', 
        'MiP', 'MaP', 'MiR', 'MaR', 'ACC'
    ])
    
    # Compute per-class detailed metrics
    per_class_metrics = calculate_detailed_per_class_metrics(y_true, y_prob, y_pred, label_names)
    
    return overall_metrics, per_class_metrics, results 