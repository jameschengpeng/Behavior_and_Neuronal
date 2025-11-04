"""
Example showing how to use the new modular structure.

This script demonstrates the complete workflow:
1. Compute dataset statistics
2. Load data with proper normalization
3. Train a model
4. Evaluate the model
5. Visualize results
"""

import torch
from torch.utils.data import DataLoader, random_split

# ============================================================================
# 1. COMPUTE DATASET STATISTICS (Run once)
# ============================================================================
print("=" * 60)
print("Step 1: Computing Dataset Statistics")
print("=" * 60)

from data.preprocessing import compute_dataset_statistics, save_normalization_stats
from utils.paths import get_data_path

# Compute statistics from your data
# Example: Use get_data_path("D21/input_output_data_downsample_444.h5")
# This will automatically use the correct path for Windows or Linux
H5_PATH = get_data_path("D21/input_output_data_downsample_444.h5")  # UPDATE the relative path as needed
mean, std = compute_dataset_statistics(H5_PATH, max_videos=10, frames_per_video=100)

# Save for future use (to data directory, same as H5 file)
save_normalization_stats(mean, std, get_data_path("D21/normalization_stats.json"))

# ============================================================================
# 2. LOAD DATA
# ============================================================================
print("\n" + "=" * 60)
print("Step 2: Loading Data")
print("=" * 60)

from data import H5VideoDataset, pad_collate_fn

# Create dataset
dataset = H5VideoDataset(
    h5_path=H5_PATH,
    merge_groups=[[0, 1], [2, 3]],  # Example: merge behaviors 0+1, 2+3
    drop_indices=[4]                 # Example: drop behavior 4
)

# Split into train/val/test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda b: pad_collate_fn(b, training=True, max_T=500)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=pad_collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=pad_collate_fn
)

print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# ============================================================================
# 3. TRAIN MODEL
# ============================================================================
print("\n" + "=" * 60)
print("Step 3: Training Model")
print("=" * 60)

from training import train_model

# Get number of labels from dataset
num_labels = dataset[0][1].shape[1]  # y shape is (T, num_labels)

model = train_model(
    train_loader=train_loader,
    num_labels=num_labels,
    num_epochs=50,
    lr=3e-4,
    save_path='behavior_model.pth',
    use_asl=True,                    # Use Asymmetric Loss
    asl_gamma_neg=4.0,
    use_bias_prior_init=True,
    stats_loader=train_loader,       # Use train_loader for computing statistics
)

print("\nTraining complete!")

# ============================================================================
# 4. EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 60)
print("Step 4: Evaluating Model")
print("=" * 60)

from evaluation import test_model

# Test with per-class threshold tuning
per_class_thresholds = test_model(
    test_loader=test_loader,
    model_path='behavior_model.pth',
    num_labels=num_labels,
    threshold=0.5,
    use_hysteresis=False,            # Can enable for temporal smoothing
    val_loader=val_loader,           # Provide val_loader for threshold tuning
    tune_per_class=True,             # Tune thresholds on validation set
    presence_mode='probs',           # Use probability-based presence detection
    presence_threshold=0.5,
)

print("\nEvaluation complete!")

# ============================================================================
# 5. VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("Step 5: Visualizing Results")
print("=" * 60)

from evaluation import plot_ethogram_pair
from models import BehaviorPredictor
import torch.nn as nn

# Load model for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BehaviorPredictor(num_labels=num_labels).to(device)
model.load_state_dict(torch.load('behavior_model.pth', map_location=device))
model.eval()

# Get a test sample
with torch.no_grad():
    for batch in test_loader:
        xb, yb, lengths, stim_type, stim_loc, stim_onset, stim_offset = batch[:7]
        xb = xb.to(device)
        
        # Forward pass
        logits = model(xb, stim_type.to(device), stim_loc.to(device), 
                      stim_onset.to(device), stim_offset.to(device), lengths.to(device))
        preds = torch.sigmoid(logits)
        
        # Get first sample in batch
        L = int(lengths[0].item())
        gt = yb[0, :L, :]       # (T, C)
        pred = preds[0, :L, :]  # (T, C)
        
        # Plot
        behavior_labels = [f'B{i}' for i in range(num_labels)]
        fig, axes = plot_ethogram_pair(
            gt, pred,
            ethogram_labels=behavior_labels,
            fps=10,
            titles=('Ground Truth', 'Model Prediction'),
            show=True
        )
        
        break  # Just plot first sample

print("\n" + "=" * 60)
print("Complete Workflow Finished!")
print("=" * 60)
print("\nFiles created:")
print("  - normalization_stats.json  (dataset statistics)")
print("  - behavior_model.pth        (trained model)")
print("\nNext steps:")
print("  1. Update NN_model.py to use normalization_stats.json")
print("  2. Experiment with different hyperparameters")
print("  3. Try hysteresis decoding: use_hysteresis=True")
