"""Local Streamlit demo. Launch with:  streamlit run demo/app.py   (CPU only, read-only)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import colormaps
from matplotlib.figure import Figure

from asymmetric_autoencoder import demo_support as ds

ROOT = Path(__file__).resolve().parents[1]
# Read-only inputs; the environment overrides exist so tests can point at empty directories.
RESULTS = Path(os.environ.get("AAE_RESULTS_DIR", ROOT / "experiments" / "results"))
ASSETS = Path(os.environ.get("AAE_ASSETS_DIR", ROOT / "demo" / "assets"))
FIGURES = Path(os.environ.get("AAE_FIGURES_DIR", ROOT / "reports" / "figures"))
MODELS = list(ds.MODEL_RUNS)
SCALE = 6  # nearest-neighbour upscaling of 32x32 images for display

st.set_page_config(page_title="Asymmetric autoencoder demo", layout="wide")


@st.cache_resource
def get_models(results_dir: str):
    return ds.load_models(Path(results_dir))


@st.cache_resource
def get_samples(assets_dir: str):
    return ds.load_samples(Path(assets_dir) / ds.SAMPLES_FILE)


@st.cache_resource
def get_pca(assets_dir: str):
    return ds.load_pca_assets(Path(assets_dir) / ds.PCA_FILE)


@st.cache_data
def get_results(results_dir: str):
    return ds.load_saved_results(Path(results_dir))


def need(loader, what: str, hint: str):
    """Return ``loader()`` or show a friendly message and return None."""
    try:
        return loader()
    except FileNotFoundError as exc:
        st.info(f"{what} not available: {exc}\n\n{hint}")
        return None


BUILD_HINT = "Run `python scripts/build_demo_assets.py` once (inference only)."


def models_or_none():
    return need(lambda: get_models(str(RESULTS)), "Checkpoints", "Saved-results pages still work.")


def samples_or_none():
    return need(lambda: get_samples(str(ASSETS)), "Validation samples", BUILD_HINT)


def pca_or_none():
    return need(lambda: get_pca(str(ASSETS)), "PCA assets", BUILD_HINT)


def show(img: np.ndarray, caption: str) -> None:
    st.image(img, caption=caption, width="stretch", clamp=True)


FORMATS = {
    "MSE": "%.6f",
    "mse": "%.6f",
    "MAE": "%.4f",
    "mae": "%.4f",
    "PSNR (dB)": "%.2f",
    "psnr": "%.2f",
    "SSIM": "%.4f",
    "ssim": "%.4f",
}


def table(rows: list[dict], index: str) -> None:
    """Show metric rows with enough decimals to tell small differences apart."""
    df = pd.DataFrame(rows).set_index(index)
    config = {c: st.column_config.NumberColumn(format=f) for c, f in FORMATS.items() if c in df}
    st.dataframe(df, column_config=config, width="stretch")


def metrics_table(rows: list[dict]) -> None:
    table(rows, "image")


def pick_sample(samples, key: str, label: str = "Validation sample") -> int:
    cls = st.selectbox("Class filter", ["all", *samples.class_names], key=f"{key}-cls")
    ids = [
        i
        for i in range(len(samples))
        if cls == "all" or samples.class_names[samples.labels[i]] == cls
    ]
    n = st.slider(label, 1, len(ids), 1, key=f"{key}-n")
    i = ids[n - 1]
    st.caption(
        f"class: **{samples.class_names[samples.labels[i]]}**, "
        f"training-array index {int(samples.val_indices[i])} (validation split)"
    )
    return i


st.title("Asymmetric Auto-Encoder Bottlenecks: interactive demo")
st.info(
    "**Single-seed evidence.** All research results come from one training run per model (seed 0); "
    "variability across seeds was not measured and no difference is statistically assessed. "
    f"Live computations below run on **CPU** (torch {torch.__version__}) on validation images; "
    "saved results were computed on Apple M2/MPS, so tiny numeric differences are expected."
)
tab_res, tab_rec, tab_rob, tab_lat = st.tabs(
    ["Results", "Reconstruct", "Robustness (N1)", "Latent"]
)

# ------------------------------------------------------------------ Results
with tab_res:
    res = get_results(str(RESULTS))
    st.subheader("Reconstruction, validation split (5,000 images)")
    rows = ds.reconstruction_rows(res, "val")
    if rows:
        table(rows, "model")
    else:
        st.info("No saved results found.")
    with st.expander(
        "Saved official-test metrics (one-time final evaluation, not re-evaluated here)"
    ):
        st.caption(
            "Computed once in Phase 5 on the 10,000 held-out images. "
            "This app never loads that split."
        )
        rows = ds.reconstruction_rows(res, "test")
        if rows:
            table(rows, "model")
    st.subheader("Model size and compute")
    st.caption("Latency: median forward pass on Apple M2 (CPU threads=4 and MPS), as saved.")
    rows = ds.efficiency_rows(res)
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("model"))
    st.subheader("Noise robustness (protocol N1, validation)")
    st.caption("Models trained on clean images; corrupted inputs scored against the clean image.")
    kind = st.radio("Corruption", ["gaussian", "salt_pepper"], horizontal=True, key="res-kind")
    metric = st.radio("Metric", ["psnr", "ssim"], horizontal=True, key="res-metric")
    levels, series = ds.robustness_series(res, kind, metric)
    if levels:
        df = pd.DataFrame(series, index=levels)
        df = df.mask(df > 90)  # an uncorrupted input hits the 100 dB PSNR floor
        st.line_chart(df)
        st.dataframe(df)
    st.subheader("Latent analysis (64-d latents; fit on train, scored on validation)")
    cols = st.columns(2)
    if rows := ds.probe_rows(res):
        cols[0].caption("Linear probe (chance = 10%)")
        cols[0].dataframe(pd.DataFrame(rows).set_index("representation"))
    if rows := ds.rank_rows(res):
        cols[1].caption("Effective dimensionality")
        cols[1].dataframe(pd.DataFrame(rows).set_index("model"))
    if trunc := ds.truncation_series(res):
        st.caption(
            "Post-hoc truncation: validation MSE when only k latent principal components are kept"
        )
        st.line_chart(pd.DataFrame(trunc))
    st.subheader("Figures")
    for name, caption in ds.FIGURES.items():
        if (FIGURES / name).is_file():
            st.image(str(FIGURES / name), caption=caption)

# ------------------------------------------------------------------ Reconstruct
with tab_rec:
    models = models_or_none()
    samples = samples_or_none()
    if models and samples:
        i = pick_sample(samples, "rec")
        x = samples.tensor(i)
        recs = ds.reconstruct(models, x)
        cols = st.columns(4)
        with cols[0]:
            show(ds.to_display(x, SCALE), "original")
        for col, name in zip(cols[1:], MODELS, strict=True):
            with col:
                show(ds.to_display(recs[name], SCALE), name)
        st.caption("Per-image metrics, live CPU computation against the original image:")
        metrics_table([{"image": n, **ds.image_metrics(r, x)} for n, r in recs.items()])
        outs = ds.multiscale_outputs(models["asymmetric"], x)
        if outs:
            with st.expander("Asymmetric model: multiscale decoder outputs"):
                st.caption(
                    "Auxiliary heads (trained against area-averaged targets); "
                    "shown without resampling."
                )
                for col, o in zip(st.columns(len(outs)), outs, strict=True):
                    with col:
                        size = o.shape[-1]
                        show(ds.to_display(o, SCALE * 32 // size), f"{size}x{size}")

# ------------------------------------------------------------------ Robustness
with tab_rob:
    st.markdown(
        "**Protocol N1 (evaluation only).** The models were trained on *clean* images. "
        "Here the input is corrupted and every reconstruction is scored against the *clean* "
        "image. This is not evidence of training-based denoising: no model was trained to denoise."
    )
    models = models_or_none()
    samples = samples_or_none()
    if models and samples:
        i = pick_sample(samples, "rob")
        kind = st.radio("Corruption", ["gaussian", "salt_pepper"], horizontal=True, key="rob-kind")
        top = 0.5 if kind == "gaussian" else 0.2
        level = st.slider(
            "Level (sigma, or fraction of pixels)", 0.0, top, 0.1, 0.01, key="rob-level"
        )
        seed = int(st.number_input("Noise seed", 0, 10_000, 0, key="rob-seed"))
        clean = samples.tensor(i)
        noisy = ds.corrupt_image(clean, kind, level, seed)
        recs = ds.reconstruct(models, noisy)
        cols = st.columns(5)
        for col, (img, cap) in zip(
            cols,
            [(clean, "clean"), (noisy, "corrupted input"), *[(recs[m], m) for m in MODELS]],
            strict=True,
        ):
            with col:
                show(ds.to_display(img, SCALE), cap)
        st.caption("Scores against the clean image (live CPU):")
        metrics_table(
            [
                {"image": "corrupted input (no model)", **ds.image_metrics(noisy, clean)},
                *[{"image": m, **ds.image_metrics(r, clean)} for m, r in recs.items()],
            ]
        )

# ------------------------------------------------------------------ Latent
with tab_lat:
    st.markdown(
        "Post-hoc analysis of models **trained with a 64-dimensional bottleneck**. PCA bases were "
        "fitted on **training-set latents only**. Truncation, interpolation and 2-D projections "
        "are exploratory visualisations and are not proof of semantic structure."
    )
    models = models_or_none()
    samples = samples_or_none()
    pca = pca_or_none()
    if models and samples and pca:
        model_name = st.selectbox("Model", MODELS, index=2, key="lat-model")
        model, asset = models[model_name], pca[model_name]
        i = pick_sample(samples, "lat")
        x = samples.tensor(i)
        st.subheader("PCA truncation of the latent (post-hoc)")
        k = st.slider(
            "Kept principal components k (0 = mean latent, 64 = full latent)",
            0,
            64,
            16,
            key="lat-k",
        )
        rec = ds.truncated_reconstruction(model, x, asset.basis, k)
        cols = st.columns(3)
        with cols[0]:
            show(ds.to_display(x, SCALE), "original")
        with cols[1]:
            show(ds.to_display(rec, SCALE), f"{model_name}, k = {k}")
        with cols[2]:
            m = ds.image_metrics(rec, x)
            st.metric("MSE (live CPU)", f"{m['mse']:.5f}")
            st.metric("PSNR (dB)", f"{m['psnr']:.2f}")
        st.subheader("2-D projection of the sampled validation latents (exploratory)")
        fig = Figure(figsize=(6, 5))
        ax = fig.subplots()
        cmap = colormaps["tab10"]
        coords = asset.sample_coords
        for c, cname in enumerate(samples.class_names):
            sel = samples.labels == c
            ax.scatter(coords[sel, 0], coords[sel, 1], s=10, color=cmap(c), label=cname)
        ax.scatter(
            *coords[i], s=140, facecolors="none", edgecolors="k", linewidths=2, label="selected"
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=6, ncol=2)
        st.pyplot(fig)
        st.subheader("Latent interpolation (exploratory)")
        j = pick_sample(samples, "lat-b", "Second validation sample")
        alphas = [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]
        for col, a, img in zip(
            st.columns(len(alphas)),
            alphas,
            ds.interpolate(model, x, samples.tensor(j), alphas),
            strict=True,
        ):
            with col:
                show(ds.to_display(img, SCALE), f"{a:.2f}")
