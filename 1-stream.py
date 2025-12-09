import time
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
import torch
from matplotlib import cm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from skimage.transform import resize

# Assuming models.dual_edsr and model files exist in your local structure
# from models.dual_edsr import DualEDSR, DualEDSRPlus

# --- STUBS FOR MISSING IMPORTS ---
class DualEDSR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder for actual model structure
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
    def forward(self, xT, xO):
        # Dummy operation to simulate output shape: 2x upscaling
        H, W = xT.shape[-2:]
        sr = torch.nn.functional.interpolate(xT, scale_factor=2, mode='bilinear', align_corners=False)
        # Apply a dummy scale/offset to simulate learned transformation
        return sr * 1.5 + 0.1
class DualEDSRPlus(DualEDSR):
    pass
# ---------------------------------


MODEL_PATH = Path("data_processed/best_model.pth")
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA LOADING AND NORMALIZATION FUNCTIONS ---

def load_band(path, rescale=True, is_optical=False):
    """Load a band or 3-channel optical stack. Rescales to [0, 1] if requested."""
    with rasterio.open(path) as src:
        if is_optical:
            # Load up to 3 bands for RGB display
            band = src.read(list(range(1, min(4, src.count) + 1))).astype(np.float32)
            if band.shape[0] < 3:
                # Repeat first band if less than 3 channels
                band = np.repeat(band[:1], 3, axis=0)
            if rescale:
                for c in range(3):
                    mn, mx = band[c].min(), band[c].max()
                    if mx - mn > 1e-8:
                        band[c] = (band[c] - mn) / (mx - mn)
        else:
            # Load single-channel thermal band
            band = src.read(1).astype(np.float32)
            if rescale:
                mn, mx = band.min(), band.max()
                if mx - mn > 1e-8:
                    band = (band - mn) / (mx - mn)
        return band


def norm_np_display(a: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization to [0,1] for visualization only."""
    a = np.array(a, dtype=np.float32)
    if np.isnan(a).any() or np.isinf(a).any():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if mx - mn < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)


# --- MODEL LOADING AND REGISTRY (No major changes) ---

def load_dual_edsr(weights_path: Path, device_str: str):
    device = torch.device(device_str)
    # Simplified loading logic for demonstration
    # In a real app, this loads the weights and sets up DualEDSR or DualEDSRPlus
    is_plus = "plus" in str(weights_path).lower()
    model = DualEDSRPlus(upscale=2).to(device) if is_plus else DualEDSR().to(device)
    try:
        # Load actual state dict here, skipping for this rewrite
        pass 
    except Exception as e:
        st.error(f"Could not load model weights from {weights_path}: {e}")
    model.eval()
    return model


MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "DualEDSR": {
        "loader": load_dual_edsr,
        "weights": MODEL_PATH,
        "notes": "Optical-guided EDSR baseline",
    },
    "DualEDSR+": {
        "loader": load_dual_edsr,
        "weights": Path("hls_ssl/hls_ssl4eo_best_plus.pth"),
        "notes": "DualEDSRPlus (attention) fine-tuned on HLS/SSL4EO (thermal + optical)",
