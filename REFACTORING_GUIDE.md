# Project Refactoring Guide

## New Structure

The codebase has been reorganized following software engineering best practices:

```
Behavior_and_Neuronal/
├── data/                    # Data-related modules
│   ├── __init__.py
│   ├── dataset.py          # H5VideoDataset (to be moved from utils.py)
│   ├── preprocessing.py    # Normalization, dataset statistics
│   └── collate.py          # Batch collation (to be moved from utils.py)
│
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── predictor.py        # BehaviorPredictor (to be moved from NN_model.py)
│   └── encoders.py         # TextEncoder (to be moved from NN_model.py)
│
├── training/                # Training logic
│   ├── __init__.py
│   ├── train.py            # train_model (to be moved from utils.py)
│   ├── losses.py           # Loss functions (to be moved from utils.py)
│   └── utils.py            # Training helpers
│
├── evaluation/              # Evaluation & visualization
│   ├── __init__.py
│   ├── test.py             # test_model (to be moved from utils.py)
│   ├── metrics.py          # Metrics computation
│   ├── postprocessing.py   # hysteresis_decode (to be moved from utils.py)
│   └── visualization.py    # plot_ethogram_pair (to be moved from utils.py)
│
├── scripts/                 # Executable scripts
│   ├── compute_stats.py    # Compute dataset normalization statistics
│   ├── train.py            # Main training script (to be created)
│   └── evaluate.py         # Main evaluation script (to be created)
│
├── NN_model.py             # [LEGACY - to be split into models/]
├── utils.py                # [LEGACY - to be split into modules above]
└── main.ipynb              # Main workflow notebook
```

## Migration Status

### ✅ Completed
- ✅ Created directory structure
- ✅ `data/preprocessing.py` - Dataset statistics computation and normalization
- ✅ `data/dataset.py` - H5VideoDataset moved from utils.py
- ✅ `data/collate.py` - Collate functions and samplers moved from utils.py
- ✅ `training/train.py` - train_model function moved from utils.py
- ✅ `training/losses.py` - Loss functions moved from utils.py
- ✅ `training/utils.py` - Training helpers moved from utils.py
- ✅ `evaluation/test.py` - test_model function moved from utils.py
- ✅ `evaluation/metrics.py` - Threshold tuning moved from utils.py
- ✅ `evaluation/postprocessing.py` - hysteresis_decode moved from utils.py
- ✅ `evaluation/visualization.py` - plot_ethogram_pair moved from utils.py
- ✅ `scripts/compute_stats.py` - Script to compute normalization statistics

### 📋 Optional Future Tasks
1. **Split NN_model.py**: 
   - `BehaviorPredictor` → `models/predictor.py`
   - `TextEncoder` → `models/encoders.py`
   
   Note: This is optional since NN_model.py is already focused on models only

## Usage

### 1. Compute Dataset Statistics (First Time Setup)

```bash
# Edit scripts/compute_stats.py to set your H5 file path
python scripts/compute_stats.py
```

This generates `normalization_stats.json` with your dataset-specific mean/std.

### 2. Use in Your Code

```python
from data.preprocessing import load_normalization_stats, Normalize

# Load computed statistics
mean, std = load_normalization_stats('normalization_stats.json')

# Use in dataset or model
normalizer = Normalize(mean=mean, std=std)
normalized_video = normalizer(video_tensor)
```

### 3. Update Your Model

In `BehaviorPredictor.__init__()`:
```python
from data.preprocessing import load_normalization_stats

# Load your dataset statistics
mean, std = load_normalization_stats('normalization_stats.json')
self.imagenet_mean = torch.tensor(mean).view(1, 3, 1, 1)
self.imagenet_std = torch.tensor(std).view(1, 3, 1, 1)
```

## Benefits of This Structure

### 1. **Separation of Concerns**
- Each module has a clear, single responsibility
- Easy to find and modify specific functionality

### 2. **Maintainability**
- Changes in one area don't affect others
- Easier to debug and test individual components

### 3. **Reusability**
- Import only what you need
- Share modules across different scripts/notebooks

### 4. **Scalability**
- Easy to add new models, losses, or evaluation metrics
- Team members can work on different modules independently

### 5. **Professional Standards**
- Follows Python/ML project conventions
- Ready for collaboration and version control

## Gradual Migration Strategy

You don't need to migrate everything at once! Here's a safe approach:

1. **Keep old files working** - Don't delete `utils.py` or `NN_model.py` yet
2. **Create new modules** - Copy functions into new structure
3. **Update imports gradually** - Start with new scripts/notebooks
4. **Test thoroughly** - Ensure everything works with new structure
5. **Deprecate old files** - Once everything is migrated and tested

## Example: Using New Structure

```python
# Old way (everything from utils.py)
from utils import H5VideoDataset, train_model, test_model, pad_collate_fn

# New way (organized imports with package-level imports)
from data import H5VideoDataset, pad_collate_fn, load_normalization_stats
from training import train_model
from evaluation import test_model, plot_ethogram_pair

# Or import from specific modules
from data.dataset import H5VideoDataset
from data.collate import pad_collate_fn
from data.preprocessing import load_normalization_stats
from training.train import train_model
from training.losses import asymmetric_loss_with_mask
from evaluation.test import test_model
from evaluation.visualization import plot_ethogram_pair
```

## Next Steps

1. Run `scripts/compute_stats.py` to generate normalization statistics
2. Update `BehaviorPredictor` to use your dataset-specific statistics
3. Gradually migrate functions from `utils.py` to appropriate modules
4. Update notebooks to use new import structure
