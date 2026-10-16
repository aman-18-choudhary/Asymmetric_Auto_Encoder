import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from asymmetric_autoencoder import demo_assets as da
from asymmetric_autoencoder import demo_support as ds
from asymmetric_autoencoder.config import config_from_dict, save_config
from asymmetric_autoencoder.evaluation.latent_analysis import PCABasis, fit_pca
from asymmetric_autoencoder.evaluation.metrics import batch_metrics
from asymmetric_autoencoder.models.factory import build_model
from asymmetric_autoencoder.training.checkpoints import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SMALL = {
    "baseline": {"type": "baseline", "latent_dim": 8, "channels": [8, 16, 16]},
    "baseline_matched": {"type": "baseline", "latent_dim": 8, "channels": [8, 16, 16]},
    "asymmetric": {
        "type": "asymmetric",
        "latent_dim": 8,
        "enc_channels": [8, 16, 16, 16],
        "dec_channels": [8, 8, 8, 8],
    },
}


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    """A tiny results tree with real (untrained, small) checkpoints in the repo's layout."""
    root = tmp_path_factory.mktemp("results")
    torch.manual_seed(0)
    for name, model_cfg in SMALL.items():
        run = ds.run_dir(root, name)
        cfg = config_from_dict({"model": model_cfg})
        save_config(cfg, run / "config.yaml") if run.mkdir(parents=True) is None else None
        save_checkpoint(run / "checkpoints" / "best.pt", build_model(cfg.model, 3, 32))
    return root


@pytest.fixture(scope="module")
def models(results_dir):
    return ds.load_models(results_dir)


def _x(n=1, seed=0):
    return torch.rand(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- models
def test_missing_checkpoints_are_reported(tmp_path):
    assert len(ds.missing_checkpoints(tmp_path)) == 3
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        ds.load_model(tmp_path, "baseline")
    with pytest.raises(FileNotFoundError, match="checkpoints not found"):
        ds.load_models(tmp_path)


def test_loaded_models_are_eval_and_frozen(models):
    assert set(models) == set(ds.MODEL_RUNS)
    for m in models.values():
        assert not m.training and not any(p.requires_grad for p in m.parameters())


def test_reconstruct_shapes_and_multiscale(models):
    x = _x()
    recs = ds.reconstruct(models, x)
    assert all(r.shape == x.shape for r in recs.values())
    assert ds.multiscale_outputs(models["baseline"], x) is None
    outs = ds.multiscale_outputs(models["asymmetric"], x)
    assert [o.shape[-1] for o in outs] == [8, 16, 32]
    assert torch.equal(outs[-1], recs["asymmetric"])


# ---------------------------------------------------------------- samples / images
def _write_samples(path: Path, n=6):
    np.savez_compressed(
        path,
        images=np.random.default_rng(0).integers(0, 256, (n, 3, 32, 32), dtype=np.uint8),
        labels=np.arange(n) % 3,
        val_indices=np.arange(n) * 7,
        class_names=np.array(["a", "b", "c"]),
    )


def test_samples_load_and_convert(tmp_path):
    _write_samples(tmp_path / "s.npz")
    s = ds.load_samples(tmp_path / "s.npz")
    t = s.tensor(2)
    assert len(s) == 6 and t.shape == (1, 3, 32, 32) and 0 <= t.min() and t.max() <= 1
    assert torch.allclose(t[0], torch.from_numpy(s.images[2]).float() / 255)
    assert ds.to_display(t).shape == (32, 32, 3)
    assert ds.to_display(t, 4).shape == (128, 128, 3)
    np.savez(
        tmp_path / "bad.npz",
        images=np.zeros((2, 3, 8, 8), np.uint8),
        labels=np.zeros(2),
        val_indices=np.zeros(2),
        class_names=np.array(["a"]),
    )
    with pytest.raises(ValueError):
        ds.load_samples(tmp_path / "bad.npz")


def test_metrics_and_corruption_use_existing_apis():
    x = _x(seed=1)
    m = ds.image_metrics(x, x)
    assert m["mse"] == 0 and m["ssim"] == pytest.approx(1.0, abs=1e-5)
    y = _x(seed=2)
    ref = batch_metrics(y, x)
    assert ds.image_metrics(y, x) == {k: float(v[0]) for k, v in ref.items()}
    a = ds.corrupt_image(x, "gaussian", 0.2, seed=3)
    assert torch.equal(a, ds.corrupt_image(x, "gaussian", 0.2, seed=3))
    assert not torch.equal(a, ds.corrupt_image(x, "gaussian", 0.2, seed=4))
    assert torch.equal(ds.corrupt_image(x, "salt_pepper", 0.0, seed=3), x)


# ---------------------------------------------------------------- latent helpers
def test_truncation_and_interpolation(models):
    m, x = models["asymmetric"], _x()
    z = ds.torch.no_grad()(m.encode)(x)
    basis = fit_pca(torch.randn(200, 8, generator=torch.Generator().manual_seed(0)))
    full = ds.truncated_reconstruction(m, x, basis, 8)
    assert torch.allclose(full, m(x), atol=1e-4)  # keeping every component changes nothing
    zero = ds.truncated_reconstruction(m, x, basis, 0)
    assert torch.allclose(zero, m.decode(basis.mean.expand_as(z)), atol=1e-6)
    a, b = _x(seed=5), _x(seed=6)
    outs = ds.interpolate(m, a, b, [0.0, 0.5, 1.0])
    assert torch.allclose(outs[0], m(a), atol=1e-5) and torch.allclose(outs[2], m(b), atol=1e-5)


def test_pca_asset_roundtrip_uses_safe_loading(tmp_path, models):
    imgs = np.random.default_rng(0).integers(0, 256, (40, 3, 32, 32), dtype=np.uint8)
    asset = da.build_pca_asset(models["baseline"], imgs[:30], imgs[30:], torch.device("cpu"))
    assert asset["components"].shape == (8, 8) and asset["sample_coords"].shape == (10, 2)
    assert torch.allclose(asset["components"].T @ asset["components"], torch.eye(8), atol=1e-4)
    torch.save(
        {"meta": {"fit_split": "train", "n_train": 30}, "models": {"baseline": asset}},
        tmp_path / "p.pt",
    )
    loaded = ds.load_pca_assets(tmp_path / "p.pt")["baseline"]
    assert isinstance(loaded.basis, PCABasis) and loaded.sample_coords.shape == (10, 2)


# ---------------------------------------------------------------- interpolation defaults
def _class_ordered_labels(n_classes=10, per_class=30):
    return np.repeat(np.arange(n_classes), per_class)  # how the demo sample file is ordered


def test_default_endpoints_are_different_images_of_different_classes():
    labels = _class_ordered_labels()
    a, b = ds.default_interpolation_endpoints(labels)
    assert (a, b) == (0, 150)  # first airplane, first dog (class 0 -> class 5)
    assert a != b and labels[a] != labels[b]
    assert ds.default_interpolation_endpoints(labels) == (a, b)  # deterministic


def test_default_endpoints_with_shuffled_few_and_single_class_labels():
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(_class_ordered_labels())
    a, b = ds.default_interpolation_endpoints(shuffled)
    assert a != b and shuffled[a] != shuffled[b]
    assert ds.default_interpolation_endpoints(np.array([3, 3, 7, 7])) == (0, 2)  # two classes
    assert ds.default_interpolation_endpoints(np.array([2, 2, 2])) == (0, 1)  # one class
    with pytest.raises(ValueError):
        ds.default_interpolation_endpoints(np.array([1]))


def test_interpolation_frames_are_decoded_from_the_endpoint_latents(models):
    m, a, b = models["asymmetric"], _x(seed=11), _x(seed=12)
    alphas = [0.0, 1 / 6, 0.5, 5 / 6, 1.0]
    frames = ds.interpolate(m, a, b, alphas)
    assert len(frames) == 5 and all(f.shape == a.shape for f in frames)
    with torch.no_grad():
        za, zb = m.encode(a), m.encode(b)
        for f, al in zip(frames, alphas, strict=True):
            assert torch.allclose(f, m.decode((1 - al) * za + al * zb), atol=1e-6)
        assert torch.allclose(frames[0], m(a), atol=1e-5) and torch.allclose(
            frames[-1], m(b), atol=1e-5
        )
    assert not torch.equal(frames[0], frames[-1])  # different endpoints give different frames
    same = ds.interpolate(m, a, a, alphas)
    assert all(
        torch.allclose(f, same[0], atol=1e-6) for f in same
    )  # identical endpoints -> identical frames


def _latent_endpoint_captions(at):
    import re

    pat = re.compile(r"class: \*\*(\w+)\*\*, training-array index (\d+)")
    found = [pat.search(c.value) for c in at.caption if pat.search(c.value)]
    return [(m.group(1), int(m.group(2))) for m in found][-2:]  # Latent tab: endpoint A, then B


@pytest.fixture(scope="module")
def full_app_available():
    pytest.importorskip("streamlit")
    needed = [ROOT / "demo" / "assets" / ds.SAMPLES_FILE, ROOT / "demo" / "assets" / ds.PCA_FILE]
    needed += [ds.checkpoint_path(ROOT / "experiments" / "results", m) for m in ds.MODEL_RUNS]
    if not all(p.is_file() for p in needed):
        pytest.skip("local checkpoints / demo assets not present")


def test_app_starts_interpolation_from_two_different_validation_images(full_app_available):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=120).run()
    assert not at.exception
    (class_a, idx_a), (class_b, idx_b) = _latent_endpoint_captions(at)
    assert idx_a != idx_b and class_a != class_b
    assert any(c.value.startswith("Endpoint A = the sample selected above") for c in at.caption)
    assert not any("same image" in i.value for i in at.info)


def test_app_interpolation_endpoints_are_independently_adjustable(full_app_available):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=120).run()
    (_, a0), (_, b0) = _latent_endpoint_captions(at)
    at.slider(key="lat-b-n").set_value(2).run()  # move only B
    assert not at.exception
    (_, a1), (_, b1) = _latent_endpoint_captions(at)
    assert a1 == a0 and b1 != b0
    at.slider(key="lat-n").set_value(5).run()  # move only A
    (_, a2), (_, b2) = _latent_endpoint_captions(at)
    assert a2 != a1 and b2 == b1
    # forcing both endpoints onto the same image is allowed and is flagged to the user
    at.selectbox(key="lat-cls").set_value("airplane").run()
    at.selectbox(key="lat-b-cls").set_value("airplane").run()
    at.slider(key="lat-n").set_value(1).run()
    at.slider(key="lat-b-n").set_value(1).run()
    assert any("same image" in i.value for i in at.info)


def test_efficiency_columns_carry_units(tmp_path):
    results = ROOT / "experiments" / "results"
    if not (results / "latent_analysis_val.json").is_file():
        pytest.skip("saved results not present")
    row = ds.efficiency_rows(ds.load_saved_results(results))[0]
    assert "MACs per image (millions)" in row and "FP32 size (MB)" in row
    assert (
        all(k.endswith("(ms)") for k in row if "latency" in k)
        and len([k for k in row if "latency" in k]) == 4
    )


# ---------------------------------------------------------------- saved results
def test_saved_results_tolerate_missing_files(tmp_path):
    res = ds.load_saved_results(tmp_path)
    assert ds.reconstruction_rows(res, "val") == [] and ds.efficiency_rows(res) == []
    assert ds.robustness_series(res, "gaussian") == ([], {})
    assert ds.probe_rows(res) == [] and ds.truncation_series(res) == {}


def test_saved_results_from_repo_if_present():
    results = ROOT / "experiments" / "results"
    if not (results / "latent_analysis_val.json").is_file():
        pytest.skip("saved results not present")
    res = ds.load_saved_results(results)
    assert [r["model"] for r in ds.reconstruction_rows(res, "val")] == list(ds.MODEL_RUNS)
    levels, series = ds.robustness_series(res, "gaussian")
    assert levels[0] == 0.0 and set(series) == {"corrupted input", *ds.MODEL_RUNS}
    assert {r["representation"] for r in ds.probe_rows(res)} >= {"raw_pixels PCA-64"}
    assert set(ds.truncation_series(res)) == set(ds.MODEL_RUNS)


def test_figure_allowlist_has_no_test_images():
    assert "reconstructions_test.png" not in ds.FIGURES
    assert not any(name.startswith("test_") for name in ds.FIGURES)
    assert all((ROOT / "reports" / "figures" / n).is_file() for n in ds.FIGURES)


# ---------------------------------------------------------------- asset builder
@pytest.fixture
def fake_cifar(tmp_path):
    d = tmp_path / "cifar"
    d.mkdir()
    rng = np.random.default_rng(1)
    for i in range(1, 6):
        batch = {
            b"data": rng.integers(0, 256, (20, 3072), dtype=np.uint8),
            b"labels": [int(v) for v in rng.integers(0, 10, 20)],
        }
        (d / f"data_batch_{i}").write_bytes(pickle.dumps(batch))
    (d / "batches.meta").write_bytes(
        pickle.dumps({b"label_names": [f"c{i}".encode() for i in range(10)]})
    )
    (d / "test_batch").write_bytes(b"not a pickle: must never be parsed")
    return d


def test_loader_reads_only_training_batches(fake_cifar):
    images, labels, names = da.load_train_arrays(fake_cifar)  # would raise if it parsed the decoy
    assert images.shape == (100, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (100,) and names == [f"c{i}" for i in range(10)]
    first = pickle.loads((fake_cifar / "data_batch_1").read_bytes())[b"data"]
    assert np.array_equal(images[:20], first.reshape(-1, 3, 32, 32))  # torchvision's order/layout


def test_stratified_sample_is_deterministic_and_balanced():
    labels = np.repeat(np.arange(4), 50)
    cand = np.arange(0, 200, 2)
    a = da.stratified_sample(labels, cand, 5, seed=1)
    assert np.array_equal(a, da.stratified_sample(labels, cand, 5, seed=1))
    assert set(a) <= set(cand) and len(a) == 20
    assert sorted(np.bincount(labels[a])) == [5, 5, 5, 5]
    with pytest.raises(ValueError):
        da.stratified_sample(labels, cand, 99)


@pytest.fixture
def project(tmp_path):
    """A throw-away project tree with the directories the guard must protect."""
    for d in ("experiments/results", "reports/figures", "data", "demo", "src"):
        (tmp_path / d).mkdir(parents=True)
    return tmp_path


def test_assets_dir_accepts_only_the_project_demo_assets(project, monkeypatch):
    expected = (project / "demo" / "assets").resolve()
    assert da.ensure_assets_dir(project / "demo" / "assets", project) == expected
    assert (
        da.ensure_assets_dir(project / "demo" / "." / "assets" / ".." / "assets", project)
        == expected
    )
    monkeypatch.chdir(project)
    assert da.ensure_assets_dir("demo/assets", project) == expected  # relative path, same target
    assert expected.is_dir()


@pytest.mark.parametrize(
    "bad",
    [
        "experiments/results",
        "experiments/results/demo/assets",
        "reports/figures",
        "reports/demo/assets",
        "data",
        "data/demo/assets",
        "demo",
        "assets",
        "demo/assets/sub",
        "demo/assets/../../experiments/results",
        "demo/assets/../../reports/figures",
        ".",
    ],
)
def test_assets_dir_rejects_other_locations_and_creates_nothing(project, bad):
    before = sorted(p.relative_to(project).as_posix() for p in project.rglob("*"))
    with pytest.raises(ValueError):
        da.ensure_assets_dir(project / bad, project)
    assert sorted(p.relative_to(project).as_posix() for p in project.rglob("*")) == before


def test_assets_dir_rejects_other_projects_and_absolute_escapes(project, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    with pytest.raises(ValueError):
        da.ensure_assets_dir(other / "demo" / "assets", project)  # right shape, wrong project
    with pytest.raises(ValueError):
        da.ensure_assets_dir("/tmp/demo/assets", project)
    assert not (other / "demo").exists()


def test_assets_dir_rejects_symlinks_that_redirect_writes(project):
    (project / "demo" / "assets").symlink_to(project / "experiments" / "results")
    with pytest.raises(ValueError):
        da.ensure_assets_dir(project / "demo" / "assets", project)
    (project / "demo" / "assets").unlink()
    (project / "demo").rmdir()
    (project / "demo").symlink_to(project / "experiments")
    with pytest.raises(ValueError):
        da.ensure_assets_dir(project / "demo" / "assets", project)
    assert [p.name for p in (project / "experiments").iterdir()] == ["results"]


def test_asset_script_uses_the_hardened_guard_with_its_own_project_root():
    text = (ROOT / "scripts" / "build_demo_assets.py").read_text()
    assert "ensure_assets_dir(args.out_dir, PROJECT_ROOT)" in text
    assert "Path(__file__).resolve().parents[1]" in text


# ---------------------------------------------------------------- safety (static)
def test_demo_code_never_touches_held_out_data_or_writes_results():
    files = [
        ROOT / "demo" / "app.py",
        ROOT / "scripts" / "build_demo_assets.py",
        ROOT / "src" / "asymmetric_autoencoder" / "demo_support.py",
        ROOT / "src" / "asymmetric_autoencoder" / "demo_assets.py",
    ]
    for f in files:
        text = f.read_text()
        for token in (
            "test_batch",
            "CIFAR10(",
            "build_test_set",
            "train=False",
            "allow-test",
            "Trainer",
            "training.trainer",
            "save_checkpoint",
            "save_config",
        ):
            assert token not in text, f"{token!r} found in {f.name}"
    for f in (files[0], files[2]):  # the app and its helpers are strictly read-only
        text = f.read_text()
        for token in ("write_text", "torch.save", "np.save", "savez", ".mkdir("):
            assert token not in text, f"{token!r} found in {f.name}"
    for f in (files[0], files[2]):
        assert "torchvision" not in f.read_text() and "data.datasets" not in f.read_text()


# ---------------------------------------------------------------- headless app smoke tests
def test_app_runs_without_assets_or_checkpoints(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AAE_RESULTS_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("AAE_ASSETS_DIR", str(tmp_path / "none"))
    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert any("checkpoints" in i.value.lower() for i in at.info)  # graceful message, no crash


def test_app_runs_with_saved_results_and_real_checkpoints_if_present():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=120).run()
    assert not at.exception and len(at.tabs) == 4
    assert any("Single-seed" in i.value for i in at.info)
    if (ROOT / "experiments" / "results" / "latent_analysis_val.json").is_file():
        assert len(at.dataframe) > 0
        json.loads((ROOT / "experiments" / "results" / "latent_analysis_val.json").read_text())


# ---------------------------------------------------------------- privacy
def test_streamlit_project_config_disables_statistics_and_binds_to_localhost():
    config = ROOT / ".streamlit" / "config.toml"
    text = config.read_text()
    assert "gatherUsageStats = false" in text and "[browser]" in text
    assert 'address = "localhost"' in text and "[server]" in text
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    parsed = tomllib.loads(text)
    assert parsed["browser"]["gatherUsageStats"] is False
    assert parsed["server"]["address"] == "localhost"


def test_streamlit_loads_project_config_from_the_repository_root():
    pytest.importorskip("streamlit")
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "streamlit", "config", "show"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    line = next(ln for ln in out.splitlines() if "gatherUsageStats" in ln and "=" in ln)
    assert "false" in line.lower() and not line.lstrip().startswith("#")
    # Streamlit's effective setting (an uncommented `address = "localhost"` line, set by our file)
    assert any(ln.strip() == 'address = "localhost"' for ln in out.splitlines())
