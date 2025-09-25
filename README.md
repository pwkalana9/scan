# ARDT-4D Radar Cube (Clean Edition)

Pipeline for **A**zimuth × **R**ange × **D**oppler × **T**ime radar cubes:
- Synthetic data generation (+ optional GPS track injection)
- Visualization for Az-Range, Az-Doppler, Range-Doppler, Doppler-Time
- Dataset splits (70:20:10)
- PVT-style 4D segmentation model (PyTorch)
- Train/Test CLI

## Install
```bash
pip install -r requirements.txt
# optional (dev install)
# pip install -e .

