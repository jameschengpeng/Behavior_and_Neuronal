"""Post-processing functions for model outputs."""

import torch


def hysteresis_decode(probs, t_low=0.3, t_high=0.7, min_len=4):
    """
    Per-class hysteresis decoding for temporal smoothing.
    
    Applies hysteresis thresholding: activates at t_high, deactivates at t_low,
    and only keeps segments >= min_len frames.
    
    Args:
        probs: (B, T, C) torch float in [0,1]
        t_low: Lower threshold (deactivation)
        t_high: Upper threshold (activation)
        min_len: Minimum segment length to keep
        
    Returns:
        Binary (B, T, C) torch float {0,1}
    """
    B, T, C = probs.shape
    out = torch.zeros_like(probs, dtype=torch.float32)
    
    for b in range(B):
        for c in range(C):
            on = False
            start = 0
            for t in range(T):
                p = probs[b, t, c].item()
                if not on and p >= t_high:
                    on = True
                    start = t
                elif on and p <= t_low:
                    if t - start >= min_len:
                        out[b, start:t, c] = 1.0
                    on = False
            # Handle segment extending to end
            if on and T - start >= min_len:
                out[b, start:T, c] = 1.0
                
    return out
