# Cross-Platform Path Migration Summary

## Overview

All Python scripts and Jupyter notebooks in this repository have been updated to support cross-platform file paths. The code now automatically detects whether it's running on Windows or Linux and uses the appropriate data paths.

## Changes Made

### 1. New Utility Module Created

**`utils/paths.py`**
- Provides `get_data_path()` for relative path resolution
- Provides `convert_path()` for Windows path conversion
- Automatically detects OS and returns correct paths
- Maps Windows paths: `D:\Mouse_behavior_data\...`
- To Linux paths: `/work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data/...`

**`utils/__init__.py`**
- Exports the path utility functions

**`utils/README.md`**
- Complete documentation on using the path utilities

### 2. Updated Python Scripts

#### `scripts/compute_stats.py`
**Before:**
```python
H5_PATH = "path/to/your/data.h5"  # UPDATE THIS
```

**After:**
```python
from utils.paths import get_data_path
H5_PATH = get_data_path("D21/input_output_data_downsample_444.h5")
```

#### `example_usage.py`
**Before:**
```python
H5_PATH = "path/to/your/data.h5"  # UPDATE THIS PATH
```

**After:**
```python
from utils.paths import get_data_path
H5_PATH = get_data_path("D21/input_output_data_downsample_444.h5")
```

### 3. Updated Jupyter Notebooks

#### `Explore_data.ipynb`
**Before:**
```python
h5_path = r"D:\Mouse_behavior_data\D21\input_output_data_downsample_444.h5"
```

**After:**
```python
from utils.paths import get_data_path
h5_path = get_data_path("D21/input_output_data_downsample_444.h5")
```

#### `main.ipynb`
**Before:**
```python
H5_PATH = r"D:\Mouse_behavior_data\D21\input_output_data_downsample_444.h5"
normalization_stats_path = r"D:\Mouse_behavior_data\D21\normalization_stats.json"
model_save_path = r"D:\Mouse_behavior_data\D21\model.pth"
```

**After:**
```python
from utils.paths import get_data_path
H5_PATH = get_data_path("D21/input_output_data_downsample_444.h5")
normalization_stats_path = get_data_path("D21/normalization_stats.json")
model_save_path = get_data_path("D21/model.pth")
```

## How It Works

The path utility automatically detects the operating system using Python's `platform.system()`:

- **On Windows**: Returns paths like `D:\Mouse_behavior_data\D21\data.h5`
- **On Linux**: Returns paths like `/work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data/D21/data.h5`

## Usage Examples

### For New Code
```python
from utils.paths import get_data_path

# Just provide the relative path from the data root
h5_file = get_data_path("D21/input_output_data_downsample_444.h5")
```

### For Existing Windows Paths
```python
from utils.paths import convert_path

# Convert existing Windows paths
old_path = r"D:\Mouse_behavior_data\D21\data.h5"
new_path = convert_path(old_path)
```

### Validation
```python
from utils.paths import get_data_path, validate_path

data_file = get_data_path("D21/data.h5")
if validate_path(data_file):
    print(f"Found: {data_file}")
```

## Testing

Run the path utility directly to test conversions:
```bash
python utils/paths.py
```

This will display:
- Detected operating system
- Example path conversions
- File existence checks

## Benefits

✅ **No more manual path editing** when switching between Windows and Linux  
✅ **Single source of truth** for data locations  
✅ **Easy to maintain** - change paths in one place  
✅ **Backward compatible** - can convert existing Windows paths  
✅ **Automatic detection** - no configuration needed  

## Migration Checklist

For future Python files that need data access:

- [ ] Import the path utility: `from utils.paths import get_data_path`
- [ ] Replace hardcoded Windows paths with `get_data_path("relative/path")`
- [ ] Test on both Windows (if available) and Linux
- [ ] Update documentation if adding new data directories

## File Structure

```
utils/
├── __init__.py          # Module exports
├── paths.py             # Path conversion utilities
└── README.md            # Detailed usage documentation
```

## Next Steps

1. ✅ All existing Python scripts updated
2. ✅ All Jupyter notebooks updated
3. ✅ Utility module created and documented
4. 📝 When adding new scripts, use `get_data_path()` from the start
5. 📝 If data root changes, update constants in `utils/paths.py`

## Support

For questions or issues:
1. Check `utils/README.md` for detailed documentation
2. Look at examples in updated scripts and notebooks
3. Run `python utils/paths.py` for testing

---

**Date of Migration**: November 3, 2025  
**Author**: GitHub Copilot  
**Status**: ✅ Complete
