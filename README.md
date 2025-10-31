# Predicting Mouse Behaviors from Spinal Cord Calcium Imaging

## 🧠 Project Overview
This project aims to build a **machine learning pipeline** that predicts a mouse’s **behavioral ethogram** (time-resolved behavioral labels) from its **neuronal calcium imaging data** and **stimulation metadata**.

The goal is to identify, from the calcium imaging signal, when and what behaviors occur — with a strong emphasis on **accurate onset timing**.

---

## 🐁 Biological Background
The experiments involve mice that received **partial sciatic nerve ligation (PSNL)** on the **left leg**, which induces **chronic neuropathic pain**.  
We specifically label **CCK neurons** in the spinal cord using **GCaMP6f**, allowing us to record their calcium activity during sensory stimulation.

---

## 📦 Dataset Summary

### Recording Sessions
- **Number of sessions:** 32 (each representing one trial)
- **Duration per session:** ~12 seconds  
- **Frame rate:** 40 frames per second  
- **Spatial resolution:** 850 × 534 pixels  
- **Data format:** time-lapse calcium imaging videos with shape ≈ (H=850, W=534, T≈600)

Each session contains **one stimulation event**.

---

### Stimulation Metadata
Each session includes the following information:
- **Stimulation type:** {von Frey filament, pin prick, brush}  
- **Stimulation location:** {left hindpaw, right hindpaw}  
- **Stimulation onset and offset times:** single pair per session (one stimulus only)

These metadata describe **what stimulus** was applied and **when/where** it occurred. Consider utilizing sentence transformer to embed the stimulation metadata.

---

### Calcium Imaging Preprocessing
We preprocess the calcium imaging videos using **AQuA2** to identify **independent calcium events**.

- A **calcium event** is defined as a group of spatially connected voxels that exhibit a **single temporal peak** in ΔF/F.
- AQuA2 also provides ΔF for each voxel.

Currently, we just use the ΔF map for the preparation of image data as model input. In the similar way as (Ajioka et al., 2024), the original ΔF map (single channel) was converted to pseudo-3-channel by stacking the frames at $[t-1, t, t+1]$ as the new 3-channel frame at time $t$. The pseudo-3-channel images makes it feasible to utilize the pretrained CNN model such as **Efficient-Net B0**. 

---

### Behavioral Ethogram
The behavioral labels are stored as a **binary ethogram matrix**:
- **Shape:** (T × B)
  - **T:** number of time frames (same as imaging, 40 Hz)
  - **B:** number of behavior types
- **Entries:**  
  - 1 → the behavior occurred at that frame  
  - 0 → otherwise
- The ethogram is **sparse**, and positive entries typically form **continuous runs** due to behavior persistence over time.

---

## 🎯 Objective
Train a deep learning model that:
1. **Inputs:**
   - Preprocessed calcium imaging data (e.g., ΔF and/or AQuA2 event maps)
   - Stimulation metadata (type, location, onset/offset time)
2. **Output:**
   - Framewise prediction of the behavioral ethogram matrix (multi-label time series)
3. **Primary focus:**  
   - **Accurate detection of behavior onset times**
   - Robustness to low signal-to-noise calcium signals

---

## ⚙️ Notes for Model Development
- Models may include:
  - **Video encoders** (e.g., 3D CNN, VideoMAE, or hybrid CNN–Transformer)
  - **Temporal sequence modules** (e.g., BiLSTM or Transformer)
  - **Metadata fusion** (embedding or FiLM-style conditioning)
- Evaluation metrics should prioritize:
  - Framewise F1-score
  - PR-AUC
  - Onset-time F1 within a tolerance window (± few frames)
