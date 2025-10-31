"""Script to compute and save dataset normalization statistics."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import compute_dataset_statistics, save_normalization_stats


def main():
    # Configuration
    H5_PATH = "path/to/your/data.h5"  # UPDATE THIS
    OUTPUT_PATH = "normalization_stats.json"
    
    MAX_VIDEOS = 10           # Sample from 10 videos
    FRAMES_PER_VIDEO = 100    # 100 frames per video
    SEED = 42
    
    print("="*60)
    print("Computing Dataset Normalization Statistics")
    print("="*60)
    print(f"H5 file: {H5_PATH}")
    print(f"Sampling: {MAX_VIDEOS} videos, {FRAMES_PER_VIDEO} frames each")
    print()
    
    # Compute statistics
    mean, std = compute_dataset_statistics(
        h5_path=H5_PATH,
        max_videos=MAX_VIDEOS,
        frames_per_video=FRAMES_PER_VIDEO,
        seed=SEED
    )
    
    # Save to file
    save_normalization_stats(mean, std, save_path=OUTPUT_PATH)
    
    print()
    print("="*60)
    print("Done! Use these values in your model:")
    print(f"  self.mean = torch.tensor({mean.tolist()})")
    print(f"  self.std  = torch.tensor({std.tolist()})")
    print("="*60)


if __name__ == "__main__":
    main()
