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
    totalDS = lncRNA_loc_dataset('./dataset/test_data.csv', k=3, mode='csv')
    tokenizer = Tokenizer(totalDS.sequences, totalDS.labels, seqMaxLen=8196, sequences_=totalDS.sequences_)
    
    # Build model – use the same number of classes as in the saved checkpoint
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
        saved_num_classes,
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

def extract_nucleus_attention_weights(model, dataset, tokenizer, num_samples=20):
    """Extract attention weights for the Nucleus class."""
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
            
            # Retrieve the original raw sequence string
            original_sequence = selected_data['sequence_']
            
            attention_data.append({
                'tokens': rna_tokens,
                'attention': attn_weights,
                'sequence_length': len(rna_tokens),
                'original_sequence': original_sequence,
                'sample_idx': idx
            })
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(indices)} samples")
    
    return attention_data

def find_nucleus_class_index(label_mapping):
    """Find the index for the Nucleus class."""
    # Inspect label_mapping structure
    print(f"Label mapping type: {type(label_mapping)}")
    print(f"Label mapping content: {label_mapping}")
    
    # If label_mapping is a dict, check key/value pairs
    if isinstance(label_mapping, dict):
        for key, value in label_mapping.items():
            print(f"Checking key: {key}, value: {value}")
            if isinstance(value, str) and 'Nucleus' in value:
                return key
            elif isinstance(key, str) and 'Nucleus' in key:
                return key
    
    # If still not found, try a direct lookup/fallback
    for key in label_mapping.keys():
        if isinstance(key, int):
            return key
    
    print("Nucleus class not found, using default index 8")
    return 8  # assume Nucleus is the 9th class (0-based index)

def create_nucleus_attention_plot(attention_data, label_mapping, save_dir, num_sequences=5):
    """Create sequence attention plots for the Nucleus class, showing ±10 bases around the max-attention base."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Find index of Nucleus class
    nucleus_idx = find_nucleus_class_index(label_mapping)
    if nucleus_idx is None:
        print("Nucleus class not found")
        return
    
    print(f"Nucleus class index: {nucleus_idx}")
    
    # Select several representative sequences
    selected_data = attention_data[:num_sequences]
    
    for seq_idx, data in enumerate(selected_data):
        tokens = data['tokens']
        attention = data['attention']
        original_sequence = data['original_sequence']
        sample_idx = data['sample_idx']
        
        # Get attention weights for the Nucleus class
        nucleus_attention = attention[nucleus_idx]
        
        # Find the position with the maximum attention
        max_attention_pos = np.argmax(nucleus_attention)
        max_attention_val = nucleus_attention[max_attention_pos]
        
        # Positions within ±10 around the maximum
        start_pos = max(0, max_attention_pos - 10)
        end_pos = min(len(nucleus_attention), max_attention_pos + 11)
        
        # Extract attention values and corresponding bases
        context_positions = range(start_pos, end_pos)
        context_attention = nucleus_attention[start_pos:end_pos]
        
        # Get the corresponding base sequence
        context_bases = []
        for pos in context_positions:
            if pos < len(tokens):
                token = tokens[pos]
                # Extract base from the token
                if len(token) >= 3:  # 3-mer token
                    # Use the middle base for 3-mers
                    base = token[1] if len(token) == 3 else token[0]
                else:
                    base = token
                context_bases.append(base)
            else:
                context_bases.append('N')
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Create bar chart
        bars = ax.bar(range(len(context_bases)), context_attention, 
                     alpha=0.7, color='lightblue', edgecolor='navy', linewidth=1)
        
        # Highlight the maximum-attention position
        max_pos_in_context = max_attention_pos - start_pos
        if 0 <= max_pos_in_context < len(bars):
            bars[max_pos_in_context].set_color('red')
            bars[max_pos_in_context].set_alpha(0.9)
        
        # Set x-axis labels to bases
        ax.set_xticks(range(len(context_bases)))
        ax.set_xticklabels(context_bases, fontsize=12, fontweight='bold')
        
        # Titles and labels
        ax.set_title(f'Nucleus Attention: ±10 Bases Around Max Attention Position {max_attention_pos}\n'
                    f'Sequence {seq_idx + 1} (Sample {sample_idx})', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax.set_xlabel('Nucleotide Bases', fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Do not place numeric labels on bars
        
        # Legend
        ax.bar([], [], color='red', alpha=0.9, label=f'Max Attention (Pos {max_attention_pos}, Val {max_attention_val:.4f})')
        ax.bar([], [], color='lightblue', alpha=0.7, label='Other Positions')
        ax.legend()
        
        # Add sequence context info
        if max_attention_pos < len(original_sequence):
            context_start = max(0, max_attention_pos - 5)
            context_end = min(len(original_sequence), max_attention_pos + 6)
            context_seq = original_sequence[context_start:context_end]
            ax.text(0.02, 0.98, f'Context Sequence: {context_seq}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'nucleus_attention_sequence_{seq_idx + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Nucleus attention plot saved to: {save_path}")
        
        # Print details
        print(f"\nSequence {seq_idx + 1} (Sample {sample_idx}) details:")
        print(f"  Max attention position: {max_attention_pos}, value: {max_attention_val:.4f}")
        print(f"  Bases and attention values within ±10 positions:")
        for i, (pos, base, attn_val) in enumerate(zip(context_positions, context_bases, context_attention)):
            marker = "***" if pos == max_attention_pos else "   "
            print(f"    Pos {pos:3d}: Base {base} -> Attention {attn_val:.4f} {marker}")
        print()

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
    print("Extracting Nucleus attention weights...")
    attention_data = extract_nucleus_attention_weights(model, testDS, tokenizer, num_samples=10)
    
    # Directory for visualizations
    save_dir = './models/new_dataset/nucleus_attention_visualization'
    
    # Create Nucleus attention plots
    print("Creating Nucleus sequence attention plots...")
    create_nucleus_attention_plot(attention_data, label_mapping, save_dir, num_sequences=5)
    
    print("Nucleus attention visualization completed!")

if __name__ == "__main__":
    main() 