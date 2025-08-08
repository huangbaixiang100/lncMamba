import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
from utils import *
from DL_ClassifierModel import *

# Configure fonts for plots (ensure minus signs render correctly)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set random seed
SEED = 388014
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def load_model_and_tokenizer(model_path, label_mapping_path):
    """Load model and tokenizer."""
    # Load label mapping
    with open(label_mapping_path, 'rb') as f:
        label_mapping = pickle.load(f)
    
    # Load dataset to build tokenizer
    # Note: the column name in test CSV is 'Subcellular_Localization', not 'SubCellular_Localization'
    totalDS = lncRNA_loc_dataset('./dataset/test_data.csv', k=3, mode='csv')
    tokenizer = Tokenizer(totalDS.sequences, totalDS.labels, seqMaxLen=8196, sequences_=totalDS.sequences_)
    
    # Build model – use the same number of classes as in the saved checkpoint
    # Load checkpoint to infer the correct number of classes
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model_dict = checkpoint['model']
    else:
        model_dict = checkpoint
    
    # Infer number of classes from checkpoint
    for key, value in model_dict.items():
        if 'outLWA.weight' in key:
            saved_num_classes = value.shape[0]
            print(f"Inferred number of classes from checkpoint: {saved_num_classes}")
            break
    else:
        saved_num_classes = tokenizer.labNum
        print(f"Unable to infer class count from checkpoint, fallback to tokenizer class count: {saved_num_classes}")
    
    backbone = KGPDPLAM_alpha_Mamba2(
        saved_num_classes,  # use class count from saved model
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
    ).to(device)
    
    # Create classifier model
    model = SequenceMultiLabelClassifier(backbone, collateFunc=PadAndTknizeCollateFunc(tokenizer), mode=0)
    model.model = model.model.to(device)
    
    # Load model weights
    try:
        model.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Try non-strict loading
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            model_dict = checkpoint['model']
        else:
            model_dict = checkpoint
        
        # Load only matching keys
        model_dict_filtered = {}
        for key in model_dict.keys():
            if key in model.model.state_dict():
                model_dict_filtered[key] = model_dict[key]
        
        model.model.load_state_dict(model_dict_filtered, strict=False)
        print("Model partially loaded (non-strict)")
    
    return model, tokenizer, label_mapping

def extract_attention_weights(model, dataset, tokenizer, num_samples=50):
    """Extract attention weights."""
    model.model.eval()
    attention_data = []
    
    # Randomly sample indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        selected_data = dataset[idx]
        inputs = selected_data['tokenizedKgpSeqArr'].unsqueeze(0).to(device)
        
        # Forward pass to obtain attention weights
        with torch.no_grad():
            output = model.model({'tokenizedKgpSeqArr': inputs})
            attn_weights = output.get('attn_weights')
        
        if attn_weights is not None:
            # Normalize attention weights container
            if isinstance(attn_weights, list):
                attn_weights = attn_weights[0]
            
            # Adjust dims: (batch_size, emb_size, class_num) -> (class_num, emb_size)
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.permute(2, 1, 0).mean(dim=2).cpu().numpy()
            
            # Retrieve token sequence
            rna_seq = inputs[0].cpu().numpy()
            rna_seq = np.argmax(rna_seq, axis=-1)
            rna_tokens = [tokenizer.id2tkn[idx] for idx in rna_seq]
            
            attention_data.append({
                'tokens': rna_tokens,
                'attention': attn_weights,
                'sequence_length': len(rna_tokens)
            })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(indices)} samples")
    
    return attention_data

def create_attention_heatmap(attention_data, label_names, save_dir, max_length=100):
    """Create attention heatmaps for each class."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create heatmap per class
    num_classes = len(label_names)
    
    for class_idx in range(num_classes):
        print(f"Creating heatmap for class {label_names[class_idx]} ...")
        
        # Collect attention data for this class
        class_attention_matrix = []
        token_sequences = []
        
        for data in attention_data:
            attention = data['attention'][class_idx]
            tokens = data['tokens']
            
            # Truncate or pad to fixed length
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                attention = attention[:max_length]
            else:
                # Pad to fixed length
                padding_length = max_length - len(tokens)
                tokens.extend(['[PAD]'] * padding_length)
                attention = np.pad(attention, (0, padding_length), mode='constant', constant_values=0)
            
            class_attention_matrix.append(attention)
            token_sequences.append(tokens)
        
        # Convert to numpy array
        attention_matrix = np.array(class_attention_matrix)
        
        # Create heatmap
        plt.figure(figsize=(20, 12))
        
        # Use seaborn to draw the heatmap
        sns.heatmap(attention_matrix, 
                   cmap='viridis', 
                   cbar_kws={'label': 'Attention Weight'},
                   xticklabels=False,  # hide x labels (too many tokens)
                   yticklabels=False)  # hide y labels
        
        plt.title(f'Attention Heatmap for {label_names[class_idx]}', fontsize=16, fontweight='bold')
        plt.xlabel('Sequence Position', fontsize=14)
        plt.ylabel('Sample Index', fontsize=14)
        
        # Save figure
        save_path = os.path.join(save_dir, f'attention_heatmap_{label_names[class_idx]}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to: {save_path}")
    
    # Create combined heatmap across all classes
    print("Creating combined heatmap...")
    create_combined_heatmap(attention_data, label_names, save_dir, max_length)

def create_combined_heatmap(attention_data, label_names, save_dir, max_length=100):
    """Create a combined heatmap for all classes."""
    num_classes = len(label_names)
    num_samples = len(attention_data)
    
    # Prepare large matrix to store attention of all classes
    combined_matrix = np.zeros((num_classes * num_samples, max_length))
    
    for class_idx in range(num_classes):
        for sample_idx, data in enumerate(attention_data):
            attention = data['attention'][class_idx]
            
            # Truncate or pad
            if len(attention) > max_length:
                attention = attention[:max_length]
            else:
                padding_length = max_length - len(attention)
                attention = np.pad(attention, (0, padding_length), mode='constant', constant_values=0)
            
            row_idx = class_idx * num_samples + sample_idx
            combined_matrix[row_idx] = attention
    
    # Create combined heatmap
    plt.figure(figsize=(24, 16))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    axes = axes.flatten()
    
    # Create subplot for each class (up to 4)
    for class_idx in range(min(4, num_classes)):  # show at most 4 classes
        start_row = class_idx * num_samples
        end_row = (class_idx + 1) * num_samples
        
        sns.heatmap(combined_matrix[start_row:end_row], 
                   cmap='viridis',
                   ax=axes[class_idx],
                   cbar_kws={'label': 'Attention Weight'},
                   xticklabels=False,
                   yticklabels=False)
        
        axes[class_idx].set_title(f'{label_names[class_idx]}', fontsize=14, fontweight='bold')
        axes[class_idx].set_xlabel('Sequence Position', fontsize=12)
        axes[class_idx].set_ylabel('Sample Index', fontsize=12)
    
    # Hide extra subplots
    for i in range(num_classes, 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'combined_attention_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined heatmap saved to: {save_path}")

def create_sequence_attention_plot(attention_data, label_names, save_dir, num_sequences=5):
    """Create sequence-level attention plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select several representative sequences
    selected_data = attention_data[:num_sequences]
    
    for seq_idx, data in enumerate(selected_data):
        tokens = data['tokens']
        attention = data['attention']
        
        # Create subplots
        num_classes = len(label_names)
        fig, axes = plt.subplots(num_classes, 1, figsize=(20, 4 * num_classes))
        if num_classes == 1:
            axes = [axes]
        
        for class_idx in range(num_classes):
            class_attention = attention[class_idx]
            
            # Create bar chart
            positions = range(len(class_attention))
            bars = axes[class_idx].bar(positions, class_attention, alpha=0.7, color='skyblue')
            
            # Highlight high-attention regions
            threshold = np.percentile(class_attention, 90)  # top 10% attention
            high_attention_positions = np.where(class_attention >= threshold)[0]
            
            for pos in high_attention_positions:
                if pos < len(bars):
                    bars[pos].set_color('red')
                    bars[pos].set_alpha(0.9)
            
            axes[class_idx].set_title(f'{label_names[class_idx]} - Sequence {seq_idx + 1}', fontsize=12, fontweight='bold')
            axes[class_idx].set_ylabel('Attention Weight', fontsize=10)
            axes[class_idx].set_xlabel('Sequence Position', fontsize=10)
            axes[class_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sequence_attention_{seq_idx + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sequence attention plot saved to: {save_path}")

def main():
    """Main entry point."""
    # Paths for model and label mapping
    model_dir = './models/new_dataset'
    model_path = os.path.join(model_dir, '')
    label_mapping_path = os.path.join(model_dir, 'label_mapping.pkl')
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, label_mapping = load_model_and_tokenizer(model_path, label_mapping_path)
    
    # Load test data
    print("Loading test data...")
    testDS = lncRNA_loc_dataset('./dataset/test_data.csv', k=3, mode='csv')
    testDS.cache_tokenizedKgpSeqArr(tokenizer, groups=512)
    
    # Extract attention weights
    print("Extracting attention weights...")
    attention_data = extract_attention_weights(model, testDS, tokenizer, num_samples=30)
    
    # Get label names
    label_names = list(label_mapping.values())
    
    # Directory for visualizations
    save_dir = './models/new_dataset/attention_visualization'
    
    # Create heatmaps
    print("Creating attention heatmaps...")
    create_attention_heatmap(attention_data, label_names, save_dir)
    
    # Create sequence-level plots
    print("Creating sequence-level attention plots...")
    create_sequence_attention_plot(attention_data, label_names, save_dir)
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main() 