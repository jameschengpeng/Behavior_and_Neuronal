"""Script to compute and save dataset normalization statistics."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import compute_dataset_statistics, save_normalization_stats
from utils.paths import get_data_path


def main():
    # Configuration
    # Example: Use get_data_path("D21/input_output_data_downsample_444.h5")
    # This will automatically use the correct path for Windows or Linux
    H5_PATH = get_data_path("D21/input_output_data_downsample_444.h5")  # UPDATE the relative path as needed
    OUTPUT_PATH = get_data_path("D21/normalization_stats.json")  # Save to data directory
    
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
