# Quick Reference: New Import Paths

## Data Operations
```python
from data import (
    H5VideoDataset,                    # Dataset class
    pad_collate_fn,                    # Batch collation
    make_bucket_batch_sampler_for_subset,  # Bucketed sampling
    build_weighted_sampler,            # Weighted sampling
    compute_dataset_statistics,         # Compute mean/std
    load_normalization_stats,          # Load saved stats
    save_normalization_stats,          # Save stats to JSON
    Normalize,                         # Normalization transform
)
```

## Training Operations
```python
from training import (
    train_model,                       # Main training function
    asymmetric_loss_with_mask,         # ASL loss
    embedding_aux_loss,                # Embedding diversity loss
    compute_pos_weight_and_prior,      # Class weight computation
    init_classifier_bias_from_prior,   # Bias initialization
)
```

## Evaluation Operations
```python
from evaluation import (
    test_model,                        # Main testing function
    tune_per_class_thresholds,         # Threshold optimization
    hysteresis_decode,                 # Temporal smoothing
    plot_ethogram_pair,                # Visualization
)
```

## Quick Start Example
```python
# 1. Load data
from data import H5VideoDataset, pad_collate_fn
from torch.utils.data import DataLoader

dataset = H5VideoDataset('data.h5')
loader = DataLoader(dataset, batch_size=4, collate_fn=pad_collate_fn)

# 2. Train
from training import train_model

model = train_model(
    train_loader=loader,
    num_labels=10,
    num_epochs=50,
    save_path='model.pth'
)

# 3. Evaluate
from evaluation import test_model

test_model(
    test_loader=loader,
    model_path='model.pth',
    num_labels=10
)

# 4. Visualize
from evaluation import plot_ethogram_pair
import torch

gt = torch.rand(100, 10)      # Example ground truth
pred = torch.rand(100, 10)    # Example predictions

plot_ethogram_pair(gt, pred, ethogram_labels=['Behavior ' + str(i) for i in range(10)])
```
