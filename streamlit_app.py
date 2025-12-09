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

from models.dual_edsr import DualEDSR, DualEDSRPlus

MODEL_PATH = Path("data_processed/best_model.pth")
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_band(path, rescale=True, is_optical=False):
    """Load and optionally normalize a band or 3-channel optical stack."""
    with rasterio.open(path) as src:
        if is_optical:
            if src.count >= 3:
                band = src.read([1, 2, 3]).astype(np.float32)
            else:
                # Gracefully handle single/multi-band optical uploads by repeating the available bands.
                band = src.read().astype(np.float32)
                if band.ndim == 2:
                    band = band[None, ...]
                band = np.repeat(band[:1], 3, axis=0)
            if rescale:
                for c in range(3):
                    mn, mx = band[c].min(), band[c].max()
                    if mx - mn > 1e-8:
                        band[c] = (band[c] - mn) / (mx - mn)
        else:
            band = src.read(1).astype(np.float32)
            if rescale:
                mn, mx = band.min(), band.max()
                if mx - mn > 1e-8:
                    band = (band - mn) / (mx - mn)
        return band


def norm_np(a: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization to [0,1] with NaN/Inf protection."""
    a = np.array(a, dtype=np.float32)
    if np.isnan(a).any() or np.isinf(a).any():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if mx - mn < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)


def load_dual_edsr(weights_path: Path, device_str: str):
    device = torch.device(device_str)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict):
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    is_plus = any(k.startswith(("convT_in", "t_groups", "o_groups", "t_upsampler")) for k in state_dict)
    if is_plus:
        model = DualEDSRPlus(upscale=2).to(device)
    else:
        model = DualEDSR().to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {'DualEDSRPlus' if is_plus else 'DualEDSR'} "
            f"(missing={missing}, unexpected={unexpected})"
        )
    model.eval()
    return model


def load_gan_gen(weights_path: Path, device_str: str):
    raise NotImplementedError("GAN loader removed")


def load_swinir(weights_path: Path, device_str: str, upscale: int = 2):
    raise NotImplementedError("SwinIR loader removed")


MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "DualEDSR": {
        "loader": load_dual_edsr,
        "weights": MODEL_PATH,
        "notes": "Optical-guided EDSR baseline",
    },
    "DualEDSR+": {
        "loader": load_dual_edsr,
        "weights": Path("hls_ssl/hls_ssl4eo_best.pth"),
        "notes": "DualEDSRPlus (attention) fine-tuned on HLS/SSL4EO (thermal + optical)",
    },
}


@st.cache_resource
def load_model(model_key: str, device_str: str):
    entry = MODEL_REGISTRY[model_key]
    model = entry["loader"](entry["weights"], device_str)
    return model


def prepare_inputs(optical_path: Path, thermal_path: Path, device: torch.device):
    opt = load_band(optical_path, is_optical=True)  # [3, H, W]
    thr = load_band(thermal_path)  # [H, W]

    xO = torch.from_numpy(opt).unsqueeze(0).to(device)  # [1, 3, H, W]
    xT = torch.from_numpy(thr).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    return xO, xT, opt, thr


def compute_metrics(sr: np.ndarray, hr: Optional[np.ndarray]):
    if hr is None:
        return {"psnr": None, "ssim": None}
    if hr.shape != sr.shape:
        hr = resize(
            hr,
            sr.shape,
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
    sr_n = norm_np(sr)
    hr_n = norm_np(hr)

    mse = float(np.mean((sr_n - hr_n) ** 2))
    if not np.isfinite(mse) or mse < 1e-12:
        psnr_val = 100.0
    else:
        psnr_val = 10 * math.log10(1.0 / mse)
    try:
        ssim_val = float(ssim(hr_n, sr_n, data_range=1.0))
    except Exception:
        ssim_val = None

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
    }


def run_model(model_key: str, xO: torch.Tensor, xT: torch.Tensor, device: torch.device):
    model = load_model(model_key, str(device))
    start = time.perf_counter()
    with torch.no_grad():
        sr = model(xT, xO)
    elapsed = time.perf_counter() - start
    sr_np = sr.squeeze().detach().cpu().numpy()
    return sr_np, elapsed


def contrast_stretch(img: np.ndarray, percentile: float):
    if percentile <= 0:
        return np.clip(img, 0, 1)
    lo, hi = np.percentile(img, [percentile, 100 - percentile])
    if hi - lo < 1e-8:
        return np.clip(img, 0, 1)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def apply_colormap(img: np.ndarray, cmap_name: str):
    """Map a single-channel image in [0,1] to RGB using a matplotlib colormap."""
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(np.clip(img, 0, 1))[..., :3]  # drop alpha
    return rgb


def render_thermal(img: np.ndarray, cmap_name: str, caption: str):
    stretched = contrast_stretch(img, contrast)
    if cmap_name == "gray":
        st.image(stretched, clamp=True, use_column_width=True, caption=caption)
    else:
        st.image(apply_colormap(stretched, cmap_name), use_column_width=True, caption=caption)


def select_device(preference: str):
    if preference == "GPU (if available)" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


st.title("🌍 Optical-Guided Thermal Super-Resolution")
st.caption("Compare multiple models side by side with runtime and PSNR/SSIM.")

with st.sidebar:
    st.header("Inputs")
    opt_file = st.file_uploader("Optical (GeoTIFF)", type=["tif", "tiff"])
    thr_file = st.file_uploader("Thermal (GeoTIFF)", type=["tif", "tiff"])
    hr_file = st.file_uploader("Reference HR Thermal (optional)", type=["tif", "tiff"])

    use_demo = st.button("Use bundled demo pair")

    model_choices = list(MODEL_REGISTRY.keys())
    default_selection = model_choices[:2] if len(model_choices) >= 2 else model_choices
    selected_models = st.multiselect("Models to run", model_choices, default=default_selection)

    colormap = st.selectbox("Thermal colormap", ["gray", "inferno", "magma", "plasma", "viridis"], index=1)

    st.header("Model Notes")
    for name in selected_models:
        st.markdown(f"- **{name}**: {MODEL_REGISTRY[name]['notes']}")


def resolve_paths():
    if opt_file and thr_file:
        opt_path = Path("temp_opt.tif")
        thr_path = Path("temp_thr.tif")
        with open(opt_path, "wb") as f:
            f.write(opt_file.read())
        with open(thr_path, "wb") as f:
            f.write(thr_file.read())
        hr_path = None
        if hr_file:
            hr_path = Path("temp_thr_hr.tif")
            with open(hr_path, "wb") as f:
                f.write(hr_file.read())
        return opt_path, thr_path, hr_path
    if use_demo:
        demo_opt = Path("sample_12_optical.tif")
        demo_thr = Path("sample_12_thermal.tif")
        if demo_opt.exists() and demo_thr.exists():
            return demo_opt, demo_thr, None
    return None, None, None


opt_path, thr_path, hr_path = resolve_paths()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
contrast = 1.0
colormap = locals().get("colormap", "inferno")

if not selected_models:
    st.warning("Select at least one model to run.")
elif opt_path is None or thr_path is None:
    st.info("Upload optical and thermal images, or click the demo button to begin.")
else:
    st.success(f"Prepared inputs on {device}. Running {len(selected_models)} model(s)...")
    with st.spinner("Running inference..."):
        xO, xT, opt_np, thr_np = prepare_inputs(opt_path, thr_path, device)
        hr_np = None
        if hr_path and hr_path.exists():
            hr_np = load_band(hr_path, rescale=True)
        ref_np = hr_np if hr_np is not None else thr_np

        results: List[Dict[str, object]] = []
        for name in selected_models:
            sr_np, elapsed = run_model(name, xO, xT, device)
            metrics = compute_metrics(sr_np, ref_np)
            results.append(
                {
                    "name": name,
                    "sr": sr_np,
                    "time": elapsed,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                }
            )

    # Overview bar
    st.markdown("---")
    info_cols = st.columns(3)
    info_cols[0].metric("Models Selected", len(selected_models))
    info_cols[1].metric("Device", str(device))
    info_cols[2].metric("HR Reference", "Provided" if hr_np is not None else "Proxy (LR thermal)")

    st.markdown("### 🖼 Multi-View Comparison")
    for res in results:
        st.markdown(f"#### {res['name']}")
        row1 = st.columns(2)
        with row1[0]:
            st.markdown("**Optical (RGB)**")
            st.image(np.transpose(opt_np, (1, 2, 0)), use_column_width=True)
        with row1[1]:
            st.markdown("**Thermal Input (LR)**")
            render_thermal(thr_np, colormap, caption=None)

        row2 = st.columns(2)
        with row2[0]:
            st.markdown("**Super-Resolved Thermal (SR)**")
            render_thermal(res["sr"], colormap, caption=None)
        with row2[1]:
            st.markdown("**HR Thermal Reference**" if hr_np is not None else "**HR Thermal Reference (Not provided)**")
            if hr_np is not None:
                render_thermal(hr_np, colormap, caption=None)
            else:
                st.info("Upload HR thermal to compute true PSNR/SSIM.")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Runtime (s)", f"{res['time']:.2f}")
        metric_cols[1].metric("PSNR", f"{res['psnr']:.2f}" if res["psnr"] is not None else "n/a")
        metric_cols[2].metric("SSIM", f"{res['ssim']:.3f}" if res["ssim"] is not None else "n/a")
        st.markdown("---")

    st.markdown("### 📊 Benchmark Table")
    table_rows = []
    for res in results:
        table_rows.append(
            {
                "Model": res["name"],
                "Runtime (s)": round(res["time"], 2),
                "PSNR": None if res["psnr"] is None else round(res["psnr"], 2),
                "SSIM": None if res["ssim"] is None else round(res["ssim"], 3),
            }
        )
    st.dataframe(table_rows, hide_index=True, use_container_width=True)
