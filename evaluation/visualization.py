"""Visualization utilities for ethograms and predictions."""

import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_ethogram_pair(
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    ethogram_labels=None,
    fps: int = 10,
    cmap: str = "viridis",
    titles=("Ground truth", "Prediction"),
    xlabel: str = "Time (in seconds)",
    ylabel: str = "Ethogram label",
    figsize=(10, 6),
    show: bool = True,
    block: bool = False,
    vmin: float = 0.0,
    vmax: float = 1.0,
    # colorbar controls
    cbar_location: str = "right",   # "right" | "left" | "bottom" | "top"
    cbar_pad: float = 0.02,
    cbar_fraction: float = 0.05,
    cbar_shrink: float = 0.95,
):
    """
    Plot two ethogram heatmaps stacked vertically (GT on top, Pred below).

    Colorbar is placed outside the panels (default: right). Use cbar_location
    to move it, e.g., cbar_location='bottom' for a horizontal bar.
    
    Args:
        gt_tensor: Ground truth ethogram (T, K) or (1, T, K)
        pred_tensor: Predicted ethogram (T, K) or (1, T, K)
        ethogram_labels: List of label names (length K)
        fps: Frames per second for x-axis conversion
        cmap: Colormap name
        titles: Tuple of (GT title, Pred title)
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        show: Whether to show the plot
        block: Whether to block execution until window is closed
        vmin, vmax: Colorbar limits
        cbar_location: Colorbar position ("right", "left", "bottom", "top")
        cbar_pad: Padding between axes and colorbar
        cbar_fraction: Fraction of axes for colorbar
        cbar_shrink: Shrink factor for colorbar
        
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    def _to_2d(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError("Inputs must be torch.Tensor")
        a = t.detach().cpu()
        if a.ndim == 3:
            if a.shape[0] != 1:
                raise ValueError(f"If 3D, first dim must be 1 (got {tuple(a.shape)})")
            a = a[0]
        if a.ndim != 2:
            raise ValueError("Each input must be 2D (T, K) or 3D with first dim = 1")
        return a  # (T, K)

    gt = _to_2d(gt_tensor)
    pr = _to_2d(pred_tensor)
    if gt.shape != pr.shape:
        raise ValueError(f"GT and Prediction must have same shape; got {gt.shape} vs {pr.shape}")

    T, K = gt.shape
    if ethogram_labels is not None:
        if len(ethogram_labels) != K:
            raise ValueError("Length of ethogram_labels must match number of columns (K)")
    else:
        ethogram_labels = [str(i) for i in range(K)]

    gt_plot = gt.T  # (K, T)
    pr_plot = pr.T  # (K, T)

    # Use constrained layout so colorbar sits outside without overlap
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize, constrained_layout=True)

    # --- Top (GT) ---
    im_top = axes[0].imshow(gt_plot, cmap=cmap, aspect="auto", origin="lower", 
                            vmin=vmin, vmax=vmax, interpolation="none")
    axes[0].set_yticks(np.arange(K))
    axes[0].set_yticklabels(ethogram_labels)
    axes[0].set_ylabel(ylabel)
    if titles and len(titles) > 0 and titles[0]:
        axes[0].set_title(titles[0])
    for y in range(1, K):
        axes[0].axhline(y - 0.5, color="white", linewidth=0.8)

    # --- Bottom (Prediction) ---
    im_bot = axes[1].imshow(pr_plot, cmap=cmap, aspect="auto", origin="lower", 
                            vmin=vmin, vmax=vmax, interpolation="none")
    
    axes[1].set_yticks(np.arange(K))
    axes[1].set_yticklabels(ethogram_labels)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    if titles and len(titles) > 1 and titles[1]:
        axes[1].set_title(titles[1])
    for y in range(1, K):
        axes[1].axhline(y - 0.5, color="white", linewidth=0.8)

    # x-axis ticks in seconds
    axes[1].set_xticks(np.linspace(0, T - 1, num=6))
    axes[1].set_xticklabels([f"{t / fps:.1f}" for t in np.linspace(0, T - 1, num=6)])

    # One colorbar outside the stack
    cbar = fig.colorbar(
        im_bot, ax=axes, location=cbar_location,
        pad=cbar_pad, fraction=cbar_fraction, shrink=cbar_shrink
    )
    cbar.set_label("Value")

    if show:
        plt.show(block=block)

    return fig, axes
