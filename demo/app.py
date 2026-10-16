"""Streamlit demo (CPU only, read-only). Local launch: scripts/run_demo.sh (localhost only)."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import colormaps
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
# Streamlit Community Cloud runs this file without pip-installing the project, so make the src/
# layout importable (harmless locally, where the package is installed in editable mode).
sys.path.insert(0, str(ROOT / "src"))

from asymmetric_autoencoder import demo_artifacts as da  # noqa: E402
from asymmetric_autoencoder import demo_support as ds  # noqa: E402

# Read-only inputs; the environment overrides exist so tests can point at empty directories.
RESULTS = Path(os.environ.get("AAE_RESULTS_DIR", ROOT / "experiments" / "results"))
ASSETS = Path(os.environ.get("AAE_ASSETS_DIR", ROOT / "demo" / "assets"))
FIGURES = Path(os.environ.get("AAE_FIGURES_DIR", ROOT / "reports" / "figures"))
MANIFEST = Path(os.environ.get("AAE_ARTIFACT_MANIFEST", ROOT / "demo" / "artifacts.json"))
RETRY_SECONDS = 60  # after a failed download, wait this long before trying again
# The live-demo files (checkpoints, demo assets) are fetched automatically only for the default
# locations: not when a test or user points the app at custom directories, and never when
# AAE_DISABLE_DOWNLOAD=1. Existing local files are used as they are and never overwritten.
AUTO_FETCH = not (
    os.environ.get("AAE_DISABLE_DOWNLOAD") == "1"
    or "AAE_RESULTS_DIR" in os.environ
    or "AAE_ASSETS_DIR" in os.environ
)
MODELS = list(ds.MODEL_RUNS)
SCALE = 6  # nearest-neighbour upscaling of 32x32 images for display

st.set_page_config(page_title="Asymmetric autoencoder demo", layout="wide")


@st.cache_resource
def runtime_root(override: str) -> Path:
    """Project root when writable (also local development), else a temporary mirror."""
    return da.prepare_runtime_root(ROOT, override or None)


LIVE_ROOT = runtime_root(os.environ.get("AAE_RUNTIME_ROOT", "")) if AUTO_FETCH else None
LIVE_RESULTS = LIVE_ROOT / "experiments" / "results" if LIVE_ROOT else RESULTS
LIVE_ASSETS = LIVE_ROOT / "demo" / "assets" if LIVE_ROOT else ASSETS


@st.cache_resource(show_spinner="Downloading the live-demo files (about 24 MB, once)...")
def fetch_live_files(root: str, manifest: str) -> list[str]:
    """Download missing artifacts (checked by size and SHA-256); raises ArtifactError on failure."""
    return [str(p) for p in da.ensure_artifacts(da.load_manifest(manifest), Path(root))]


def live_ready() -> bool:
    """True when the live tabs can run; otherwise explain why (once per tab) and return False."""
    if not AUTO_FETCH:
        return True  # nothing to fetch; the loaders show their own notices if files are missing
    error = None
    if time.time() < st.session_state.get("_live_error_until", 0.0):
        error = st.session_state["_live_error"]
    else:
        try:
            manifest = da.load_manifest(MANIFEST)
            if da.missing(manifest, LIVE_ROOT):  # local development: files present, no network
                fetch_live_files(str(LIVE_ROOT), str(MANIFEST))
        except da.ArtifactError as exc:
            error = str(exc)
            st.session_state["_live_error"] = error
            st.session_state["_live_error_until"] = time.time() + RETRY_SECONDS
    if error:
        st.warning(
            f"The live-demo files could not be provided: {error}\n\n"
            "This is usually a temporary network problem; the app tries again about every minute. "
            "The Results tab is unaffected."
        )
        return False
    return True


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
        detail = str(exc.filename or exc).replace(str(ROOT) + os.sep, "")  # no absolute paths
        st.info(f"{what} not available ({detail}).\n\n{hint}")
        return None


BUILD_HINT = (
    "Live tabs need the demo assets: the hosted app downloads them automatically; locally, build "
    "them once (inference only) with `python scripts/build_demo_assets.py`."
)
CHECKPOINT_HINT = (
    "The Results tab still works. Live inference needs the three trained checkpoints: the hosted "
    "app downloads them automatically; locally they come from your own training runs (see the "
    "README)."
)


def models_or_none():
    return need(lambda: get_models(str(LIVE_RESULTS)), "Checkpoints", CHECKPOINT_HINT)


def samples_or_none():
    return need(lambda: get_samples(str(LIVE_ASSETS)), "Validation samples", BUILD_HINT)


def pca_or_none():
    return need(lambda: get_pca(str(LIVE_ASSETS)), "PCA assets", BUILD_HINT)


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


HOW_TO_READ = "Lower is better: MSE, MAE. Higher is better: PSNR (dB), SSIM."
UNITS = {"psnr": "PSNR (dB)", "ssim": "SSIM"}


def line_chart(series: dict, xlabel: str, ylabel: str, logy: bool = False) -> None:
    """Labelled line chart with deterministic axes; ``series[name] = (x values, y values)``."""
    fig = Figure(figsize=(6.5, 3.6))
    ax = fig.subplots()
    for name, (xs, ys) in series.items():
        style = (
            {"color": "k", "linestyle": "--", "marker": "x"}
            if name == "corrupted input"
            else {"marker": "o", "markersize": 3}
        )
        ax.plot(xs, ys, label=name, **style)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)


def pick_sample(samples, key: str, label: str = "Validation sample", default: int = 0) -> int:
    cls = st.selectbox("Class filter", ["all", *samples.class_names], key=f"{key}-cls")
    ids = [
        i
        for i in range(len(samples))
        if cls == "all" or samples.class_names[samples.labels[i]] == cls
    ]
    start = ids.index(default) + 1 if default in ids else 1  # default applies to "all" only
    n = st.slider(label, 1, len(ids), start, key=f"{key}-n")
    i = ids[n - 1]
    st.caption(
        f"class: **{samples.class_names[samples.labels[i]]}**, "
        f"training-array index {int(samples.val_indices[i])} (validation split)"
    )
    return i


st.title("Asymmetric Autoencoder Bottlenecks: Interactive Research Demo")
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
    st.caption(
        "Saved results from the research runs (seed 0, Apple M2/MPS); checkpoints were chosen on "
        f"this split. {HOW_TO_READ}"
    )
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
    st.caption(
        "Saved measurements on an Apple M2 (CPU with 4 threads, and MPS): median forward-pass "
        "time. MACs = multiply-accumulates per image (convolutions and linear layers only)."
    )
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
        xlabel = "Gaussian noise sigma" if kind == "gaussian" else "fraction of pixels corrupted"
        line_chart(
            {n: (levels, df[n].tolist()) for n in df}, xlabel, UNITS[metric] + " vs clean image"
        )
        st.dataframe(df.rename_axis(xlabel))
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
        line_chart(
            {n: (sorted(v), [v[k] for k in sorted(v)]) for n, v in trunc.items()},
            "kept latent principal components k (64 = full latent)",
            "validation MSE (log scale)",
            logy=True,
        )
    st.subheader("Figures")
    for name, caption in ds.FIGURES.items():
        if (FIGURES / name).is_file():
            st.image(str(FIGURES / name), caption=caption)

# ------------------------------------------------------------------ Reconstruct
with tab_rec:
    ready = live_ready()
    models = models_or_none() if ready else None
    samples = samples_or_none() if ready else None
    if models and samples:
        st.caption(
            "**baseline**: small symmetric convolutional autoencoder. **baseline_matched**: the "
            "same design widened to the asymmetric model's parameter count. **asymmetric**: heavy "
            "residual encoder with a light multiscale decoder. All three use a 64-dimensional "
            "latent."
        )
        i = pick_sample(samples, "rec")
        x = samples.tensor(i)
        recs = ds.reconstruct(models, x)
        cols = st.columns(4)
        with cols[0]:
            show(ds.to_display(x, SCALE), "original")
        for col, name in zip(cols[1:], MODELS, strict=True):
            with col:
                show(ds.to_display(recs[name], SCALE), name)
        st.caption(
            f"Per-image metrics, live CPU computation against the original image. {HOW_TO_READ}"
        )
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
    ready = live_ready()
    models = models_or_none() if ready else None
    samples = samples_or_none() if ready else None
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
        st.caption(
            "Scores against the clean image (live CPU). The first row is the noisy image itself, "
            "i.e. what a model that changed nothing would score; the models are not denoisers. "
            f"{HOW_TO_READ}"
        )
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
    ready = live_ready()
    models = models_or_none() if ready else None
    samples = samples_or_none() if ready else None
    pca = pca_or_none() if ready else None
    if models and samples and pca:
        model_name = st.selectbox("Model", MODELS, index=2, key="lat-model")
        model, asset = models[model_name], pca[model_name]
        a_default, b_default = ds.default_interpolation_endpoints(samples.labels)
        i = pick_sample(
            samples, "lat", "Validation sample (also interpolation endpoint A)", a_default
        )
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
        j = pick_sample(samples, "lat-b", "Interpolation endpoint B", b_default)
        name = samples.class_names
        st.caption(
            f"Endpoint A = the sample selected above (**{name[samples.labels[i]]}**); endpoint B = "
            f"**{name[samples.labels[j]]}**. Each frame decodes the latent mix (1 - alpha) * z_A + "
            f"alpha * z_B with the **{model_name}** model; alpha = 0 is A's reconstruction and "
            "alpha = 1 is B's."
        )
        if i == j:
            st.info("Endpoints A and B are the same image, so every frame is identical.")
        alphas = [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]
        x_b = samples.tensor(j)
        ends = st.columns(len(alphas))
        with ends[0]:
            show(ds.to_display(x, SCALE), f"A (original): {name[samples.labels[i]]}")
        with ends[-1]:
            show(ds.to_display(x_b, SCALE), f"B (original): {name[samples.labels[j]]}")
        frames = ds.interpolate(model, x, x_b, alphas)
        for col, a, img in zip(st.columns(len(alphas)), alphas, frames, strict=True):
            with col:
                tag = " (A)" if a == 0 else " (B)" if a == 1 else ""
                show(ds.to_display(img, SCALE), f"alpha = {a:.2f}{tag}")
