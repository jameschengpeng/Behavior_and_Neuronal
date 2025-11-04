# Path Utilities for Cross-Platform Data Access

This directory contains utilities for handling file paths that differ between Windows and Linux systems.

## Overview

The `paths.py` module provides functions to automatically convert between Windows and Linux data paths, ensuring your scripts work seamlessly on both operating systems.

### Path Mappings

- **Windows**: `D:\Mouse_behavior_data\...`
- **Linux**: `/work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data/...`

## Usage

### Method 1: Using `get_data_path()` (Recommended)

The simplest way is to use `get_data_path()` with a relative path:

```python
from utils.paths import get_data_path

# Provide the relative path from the data root
h5_path = get_data_path("D21/input_output_data_downsample_444.h5")
model_path = get_data_path("D21/model.pth")
stats_path = get_data_path("D21/normalization_stats.json")

# The function automatically returns the correct path for your OS
# On Linux: /work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data/D21/input_output_data_downsample_444.h5
# On Windows: D:\Mouse_behavior_data\D21\input_output_data_downsample_444.h5
```

### Method 2: Using `convert_path()` for Existing Windows Paths

If you have existing Windows paths in your code, use `convert_path()`:

```python
from utils.paths import convert_path

# Convert a Windows path
windows_path = r"D:\Mouse_behavior_data\D21\input_output_data_downsample_444.h5"
h5_path = convert_path(windows_path)

# On Linux, this returns: /work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data/D21/input_output_data_downsample_444.h5
# On Windows, this returns the original path unchanged
```

### Method 3: Validating Paths

Check if a path exists:

```python
from utils.paths import get_data_path, validate_path

h5_path = get_data_path("D21/input_output_data_downsample_444.h5")

if validate_path(h5_path):
    print(f"File found: {h5_path}")
else:
    print(f"File not found: {h5_path}")
```

## Examples in the Repository

### Python Scripts

All Python scripts have been updated to use the path utilities:

1. **`scripts/compute_stats.py`**
   ```python
   from utils.paths import get_data_path
   H5_PATH = get_data_path("D21/input_output_data_downsample_444.h5")
   ```

2. **`example_usage.py`**
   ```python
   from utils.paths import get_data_path
   H5_PATH = get_data_path("D21/input_output_data_downsample_444.h5")
   ```

### Jupyter Notebooks

Both notebooks have been updated:

1. **`Explore_data.ipynb`**
   ```python
   from utils.paths import get_data_path
   h5_path = get_data_path("D21/input_output_data_downsample_444.h5")
   ```

2. **`main.ipynb`**
   ```python
   from utils.paths import get_data_path
   H5_PATH = get_data_path("D21/input_output_data_downsample_444.h5")
   normalization_stats_path = get_data_path("D21/normalization_stats.json")
   model_save_path = get_data_path("D21/model.pth")
   ```

## Testing

You can test the path utilities by running the module directly:

```bash
python utils/paths.py
```

This will show you:
- The detected operating system
- Example path conversions
- Whether files exist at the converted paths

## Adding New Data Paths

When you need to access a new data file:

1. Identify the relative path from the data root (e.g., `D21/new_file.h5`)
2. Use `get_data_path()` to get the full path:
   ```python
   from utils.paths import get_data_path
   new_file_path = get_data_path("D21/new_file.h5")
   ```

## Configuration

If your data is stored in a different location, you can modify the constants in `paths.py`:

```python
# Base paths for different operating systems
WINDOWS_DATA_ROOT = r"D:\Mouse_behavior_data"
LINUX_DATA_ROOT = "/work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data"
```

## Benefits

✅ **Cross-platform compatibility**: Code works on both Windows and Linux  
✅ **No manual path editing**: Paths automatically adjust based on OS  
✅ **Centralized configuration**: Change data roots in one place  
✅ **Easy migration**: Simple to adapt to new environments  
✅ **Backward compatible**: Existing Windows paths can be converted  

## Support

For issues or questions about the path utilities, check:
- This README
- The docstrings in `paths.py`
- Run `python utils/paths.py` for testing and examples
