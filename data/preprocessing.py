"""
Data preprocessing utilities for calcium imaging videos.

This module handles:
- Computing dataset-specific normalization statistics
- Normalization transforms
"""

import numpy as np
import h5py
import torch
import json
from pathlib import Path


def compute_dataset_statistics(h5_path, max_videos=10, frames_per_video=100, seed=42):
    """
    Compute channel-wise mean and std from your calcium imaging dataset.
    
    Since you're using pseudo-RGB (t-1, t, t+1 stacked frames), this computes
    statistics specific to your temporal imaging data rather than using ImageNet stats.
    
    Args:
        h5_path: Path to HDF5 file with video data
        max_videos: Maximum number of videos to sample (for efficiency)
        frames_per_video: Number of random frames to sample per video
        seed: Random seed for reproducibility
        
    Returns:
        mean: np.array of shape (3,) - per-channel means
        std: np.array of shape (3,) - per-channel stds
    """
    np.random.seed(seed)
    
    with h5py.File(h5_path, 'r') as f:
        video_keys = sorted(list(f.keys()))
        
        # Sample a subset of videos if dataset is large
        if len(video_keys) > max_videos:
            sampled_keys = np.random.choice(video_keys, max_videos, replace=False)
        else:
            sampled_keys = video_keys
            
        all_pixels = []
        
        print(f"Computing statistics from {len(sampled_keys)} videos...")
        
        for i, key in enumerate(sampled_keys):
            X = f[key]['X'][:]  # Load video data
            
            # Handle different data layouts
            if X.ndim == 4:
                if X.shape[-1] == 3:        # (T, H, W, C)
                    X = np.transpose(X, (3, 0, 1, 2))  # -> (C, T, H, W)
                elif X.shape[0] == 3:       # (C, H, W, T)
                    X = np.transpose(X, (0, 3, 1, 2))  # -> (C, T, H, W)
                else:
                    raise ValueError(f"Unexpected X shape {X.shape}")
            else:
                raise ValueError(f"Expected 4D array, got {X.ndim}D")
            
            # Normalize to [0, 1] if needed
            if X.max() > 1.0:
                X = X.astype(np.float32) / 255.0
            
            C, T, H, W = X.shape
            
            # Sample random frames from this video
            n_frames = min(frames_per_video, T)
            sampled_indices = np.random.choice(T, n_frames, replace=False)
            
            # Collect pixels: (C, n_frames, H, W)
            all_pixels.append(X[:, sampled_indices, :, :])
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(sampled_keys)} videos...")
        
        # Concatenate all sampled frames: (C, total_frames, H, W)
        all_pixels = np.concatenate(all_pixels, axis=1)
        
        # Compute per-channel statistics
        mean = all_pixels.mean(axis=(1, 2, 3))  # (3,)
        std = all_pixels.std(axis=(1, 2, 3))    # (3,)
        
        print(f"\nDataset statistics computed from {all_pixels.shape[1]} frames:")
        print(f"  Mean: {mean}")
        print(f"  Std:  {std}")
        
        return mean, std


def save_normalization_stats(mean, std, save_path='normalization_stats.json'):
    """Save normalization statistics to JSON file."""
    stats = {
        'mean': mean.tolist() if isinstance(mean, np.ndarray) else mean,
        'std': std.tolist() if isinstance(std, np.ndarray) else std,
    }
    
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Normalization stats saved to: {save_path}")


def load_normalization_stats(load_path='normalization_stats.json'):
    """Load normalization statistics from JSON file."""
    if not Path(load_path).exists():
        raise FileNotFoundError(
            f"Normalization stats file not found: {load_path}\n"
            f"Run scripts/compute_stats.py first to generate it."
        )
    
    with open(load_path, 'r') as f:
        stats = json.load(f)
    
    mean = np.array(stats['mean'], dtype=np.float32)
    std = np.array(stats['std'], dtype=np.float32)
    
    return mean, std


class Normalize:
    """
    Normalization transform for video data.
    
    Args:
        mean: Channel-wise means (3,) or None to use ImageNet defaults
        std: Channel-wise stds (3,) or None to use ImageNet defaults
        use_imagenet: If True and mean/std are None, use ImageNet statistics
    """
    def __init__(self, mean=None, std=None, use_imagenet=False):
        if mean is None and std is None and use_imagenet:
            # ImageNet defaults (for reference/fallback)
            self.mean = torch.tensor([0.485, 0.456, 0.406])
            self.std = torch.tensor([0.229, 0.224, 0.225])
            print("Warning: Using ImageNet normalization for pseudo-RGB data")
        elif mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)
        else:
            raise ValueError("Must provide either (mean, std) or set use_imagenet=True")
    
    def __call__(self, x):
        """
        Normalize a video tensor.
        
        Args:
            x: Tensor of shape (C, T, H, W) or (B, C, T, H, W)
            
        Returns:
            Normalized tensor of same shape
        """
        if x.ndim == 4:
            # (C, T, H, W)
            mean = self.mean.to(x.device).view(3, 1, 1, 1)
            std = self.std.to(x.device).view(3, 1, 1, 1)
        elif x.ndim == 5:
            # (B, C, T, H, W)
            mean = self.mean.to(x.device).view(1, 3, 1, 1, 1)
            std = self.std.to(x.device).view(1, 3, 1, 1, 1)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.ndim}D")
        
        return (x - mean) / std
