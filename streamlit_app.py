import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from skimage.transform import resize

from models.dual_edsr import DualEDSR
from models.gan_sr import build_gan
from models.swinir_guided import SwinIRConfig, build_model as build_swinir

MODEL_PATH = Path("data_processed/best_model.pth")
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_band(path, rescale=True, is_optical=False):
    """Load and optionally normalize a band or 3-channel optical stack."""
    with rasterio.open(path) as src:
        if is_optical:
            band = src.read([1, 2, 3]).astype(np.float32)
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


def load_dual_edsr(weights_path: Path, device_str: str):
    device = torch.device(device_str)
    model = DualEDSR().to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_gan_gen(weights_path: Path, device_str: str):
    device = torch.device(device_str)
    gen, _ = build_gan()
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    gen.load_state_dict(state_dict)
    gen = gen.to(device)
    gen.eval()
    return gen


def load_swinir(weights_path: Path, device_str: str, upscale: int = 2):
    device = torch.device(device_str)
    cfg = SwinIRConfig(weight_path=weights_path, upscale=upscale, device=device_str)
    model = build_swinir(cfg)
    model.eval()
    return model


MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "DualEDSR": {
        "loader": load_dual_edsr,
        "weights": MODEL_PATH,
        "notes": "Optical-guided EDSR baseline",
    },
    "DualEDSR-GAN": {
        "loader": load_gan_gen,
        "weights": Path("checkpoints/gan_ssl4eo/gan_best.pth"),
        "notes": "Newest GAN run (SSL4EO) stored under checkpoints/gan_ssl4eo",
    },
    "SwinIR-Guided": {
        "loader": lambda w, d: load_swinir(w, d, upscale=2),
        "weights": Path("checkpoints/swinir/ssl4eo_best_swinir.pth"),
        "notes": "Guided SwinIR (simplified attention) using thermal+optical; weights in checkpoints/swinir",
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
    return {
        "psnr": float(peak_signal_noise_ratio(hr, sr, data_range=hr.max() - hr.min() + 1e-8)),
        "ssim": float(ssim(hr, sr, data_range=hr.max() - hr.min() + 1e-8)),
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
        # Use proxy reference (input thermal) if no HR provided so PSNR/SSIM still display.
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

    tabs = st.tabs(["Inputs"] + [r["name"] for r in results])

    with tabs[0]:
        st.subheader("Inputs")
        if hr_np is None:
            st.info("No HR thermal provided: PSNR/SSIM use the input thermal as a proxy reference.")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.imshow(np.transpose(opt_np, (1, 2, 0)))
            ax.set_title("Optical (RGB)")
            ax.axis("off")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.imshow(contrast_stretch(thr_np, contrast), cmap="gray")
            ax.set_title("Thermal Input")
            ax.axis("off")
            st.pyplot(fig)

    for idx, res in enumerate(results, start=1):
        with tabs[idx]:
            st.subheader(res["name"])
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**Optical (RGB)**")
                st.image(np.transpose(opt_np, (1, 2, 0)), use_column_width=True)
            with col_b:
                st.markdown("**Thermal Input**")
                st.image(contrast_stretch(thr_np, contrast), clamp=True, use_column_width=True)
            with col_c:
                st.markdown("**Super-Resolved Thermal**")
                st.image(contrast_stretch(res["sr"], contrast), clamp=True, use_column_width=True)

            cols = st.columns(3)
            cols[0].metric("Runtime (s)", f"{res['time']:.2f}")
            cols[1].metric("PSNR", f"{res['psnr']:.2f}" if res["psnr"] is not None else "n/a")
            cols[2].metric("SSIM", f"{res['ssim']:.3f}" if res["ssim"] is not None else "n/a")

    st.subheader("Benchmark Table")
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
    st.dataframe(table_rows, hide_index=True)
