"""
Path utilities for cross-platform data access.

This module provides functions to handle paths that differ between
Windows and Linux systems. On Windows, data is typically stored at:
    D:\\Mouse_behavior_data\\...
    
On Linux, the same data is stored at:
    /work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data/...
"""

import os
import platform
import re
from pathlib import Path


# Base paths for different operating systems
WINDOWS_DATA_ROOT = r"D:\Mouse_behavior_data"
LINUX_DATA_ROOT = "/work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data"


def get_os_type():
    """
    Detect the operating system.
    
    Returns:
        str: 'Windows' or 'Linux' (or 'Darwin' for macOS)
    """
    return platform.system()


def convert_path(windows_path):
    r"""
    Convert a Windows data path to the appropriate path for the current OS.
    
    Args:
        windows_path (str): Windows-style path like 
                           "D:\\Mouse_behavior_data\\D21\\input_output_data_downsample_444.h5"
                           or "D:/Mouse_behavior_data/D21/input_output_data_downsample_444.h5"
    
    Returns:
        str: Appropriate path for the current operating system
        
    Examples:
        On Linux:
        >>> convert_path(r"D:\Mouse_behavior_data\D21\data.h5")
        '/work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data/D21/data.h5'
        
        On Windows:
        >>> convert_path(r"D:\Mouse_behavior_data\D21\data.h5")
        'D:\\Mouse_behavior_data\\D21\\data.h5'
    """
    if not windows_path or windows_path == "path/to/your/data.h5":
        # Return placeholder unchanged
        return windows_path
    
    # Normalize the path to use forward slashes
    normalized = windows_path.replace('\\', '/')
    
    # Extract the relative part after the Windows root
    # Handle both "D:\Mouse_behavior_data" and "D:/Mouse_behavior_data"
    pattern = r'^[A-Za-z]:[/\\]Mouse_behavior_data[/\\]?'
    match = re.match(pattern, windows_path.replace('/', '\\'))
    
    if match:
        # Extract relative path after "D:\Mouse_behavior_data\"
        relative_path = normalized[len("D:/Mouse_behavior_data/"):]
        
        os_type = get_os_type()
        if os_type == "Windows":
            # Keep original Windows format
            return windows_path
        else:
            # Convert to Linux path
            linux_path = os.path.join(LINUX_DATA_ROOT, relative_path)
            return linux_path
    else:
        # Path doesn't match expected pattern, return as-is
        return windows_path


def get_data_path(relative_path):
    """
    Get the full data path for a file relative to the data root.
    
    Args:
        relative_path (str): Path relative to data root, e.g., "D21/input_output_data_downsample_444.h5"
    
    Returns:
        str: Full path appropriate for the current OS
        
    Examples:
        On Linux:
        >>> get_data_path("D21/data.h5")
        '/work/Jamespeng/Astrocyte/Behavior_and_Neuronal_Data/D21/data.h5'
        
        On Windows:
        >>> get_data_path("D21/data.h5")
        'D:\\Mouse_behavior_data\\D21\\data.h5'
    """
    os_type = get_os_type()
    
    if os_type == "Windows":
        return os.path.join(WINDOWS_DATA_ROOT, relative_path)
    else:
        return os.path.join(LINUX_DATA_ROOT, relative_path)


def validate_path(path):
    """
    Check if a path exists.
    
    Args:
        path (str): Path to validate
        
    Returns:
        bool: True if path exists, False otherwise
    """
    return os.path.exists(path)


# Example usage and testing
if __name__ == "__main__":
    print(f"Operating System: {get_os_type()}")
    print()
    
    # Test conversion
    test_paths = [
        r"D:\Mouse_behavior_data\D21\input_output_data_downsample_444.h5",
        r"D:/Mouse_behavior_data/D21/input_output_data_downsample_444.h5",
        r"D:\Mouse_behavior_data\D21\normalization_stats.json",
        r"D:\Mouse_behavior_data\D21\model.pth",
    ]
    
    print("Path Conversion Tests:")
    print("-" * 80)
    for path in test_paths:
        converted = convert_path(path)
        exists = validate_path(converted)
        print(f"Original:  {path}")
        print(f"Converted: {converted}")
        print(f"Exists:    {exists}")
        print()
    
    print("Get Data Path Tests:")
    print("-" * 80)
    test_relative = [
        "D21/input_output_data_downsample_444.h5",
        "D21/normalization_stats.json",
        "D21/model.pth",
    ]
    
    for rel_path in test_relative:
        full_path = get_data_path(rel_path)
        exists = validate_path(full_path)
        print(f"Relative:  {rel_path}")
        print(f"Full:      {full_path}")
        print(f"Exists:    {exists}")
        print()
