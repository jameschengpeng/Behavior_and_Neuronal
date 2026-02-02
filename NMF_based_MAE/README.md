# NMF-based Temporal MAE

This folder contains a minimal masked autoencoder (MAE) for temporal NMF components exported by `NMF_result_to_h5.m`.

## Expected H5 structure

- `/A` : spatial components (n_pixels × k)
- `/video_XX/C` : temporal components (k × T)
- `/video_XX/S` : stimulus channels (nCond × T)

## Quick usage

Train with C-only (full video as one sample):

- `--include-s` is **off** by default
- Masking is temporal with contiguous spans

Train with S conditioning:

- add `--include-s`
- S is used as conditioning only (not masked or reconstructed)

## Notes

- Patch length controls temporal patch size used for masking.
- The MAE reconstructs only masked patches from C.
- S channels are never masked and never reconstructed.
- Each video is treated as a single sample with shape (T, D).
