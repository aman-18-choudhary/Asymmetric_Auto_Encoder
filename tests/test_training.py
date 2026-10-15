import json

import pytest
import torch
from torchvision import transforms
from torchvision.datasets import FakeData

from asymmetric_autoencoder.config import ModelConfig, config_from_dict
from asymmetric_autoencoder.data.datasets import make_loader
from asymmetric_autoencoder.models.factory import build_model
from asymmetric_autoencoder.training.checkpoints import load_checkpoint, save_checkpoint
from asymmetric_autoencoder.training.trainer import Trainer, evaluate, run_dir_for
from asymmetric_autoencoder.utils.reproducibility import (
    get_rng_state,
    seed_everything,
    set_rng_state,
)

CPU = torch.device("cpu")
SMALL = {
    "baseline": {"type": "baseline", "latent_dim": 8, "channels": [8, 16, 16]},
    "asymmetric": {
        "type": "asymmetric",
        "latent_dim": 8,
        "enc_channels": [8, 16, 16, 16],
        "dec_channels": [8, 8, 8, 8],
    },
}


def _cfg(model="baseline", **train):
    t = {"epochs": 4, "lr": 0.003, **train}
    return config_from_dict({"seed": 3, "model": SMALL[model], "train": t})


def _loaders(seed=3):
    tf = transforms.ToTensor()
    tr = FakeData(size=48, image_size=(3, 32, 32), transform=tf)
    va = FakeData(size=16, image_size=(3, 32, 32), transform=tf, random_offset=500)
    return make_loader(tr, 16, True, seed), make_loader(va, 16, False, seed)


def _trainer(cfg, run_dir, **kw):
    seed_everything(cfg.seed)
    model = build_model(cfg.model, 3, 32)
    train, val = _loaders(cfg.seed)
    return Trainer(model, train, val, cfg, CPU, run_dir=run_dir, **kw)


# ---- config ----
@pytest.mark.parametrize(
    "train",
    [
        {"optimizer": "sgd"},
        {"scheduler": "step"},
        {"weight_decay": -1},
        {"early_stopping_patience": -1},
    ],
)
def test_train_config_validation(train):
    with pytest.raises(ValueError):
        config_from_dict({"train": train})


def test_run_dir_default_name():
    cfg = config_from_dict({"experiment_name": "exp", "seed": 5})
    assert str(run_dir_for(cfg)).endswith("exp_seed5")


# ---- checkpoints ----
def test_checkpoint_round_trip(tmp_path):
    cfg = _cfg()
    model = build_model(cfg.model, 3, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.rand(2, 3, 32, 32)).mean().backward()
    opt.step()
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, opt, epoch=7, history=[{"a": 1}])
    assert not (tmp_path / "ck.pt.tmp").exists()

    model2 = build_model(cfg.model, 3, 32)
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.5)
    payload = load_checkpoint(path, model2, opt2)
    assert payload["epoch"] == 7 and payload["history"] == [{"a": 1}]
    for a, b in zip(model.state_dict().values(), model2.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert opt2.param_groups[0]["lr"] == 0.01
    assert len(opt2.state) == len(opt.state) > 0


def test_checkpoint_rejects_missing_optimizer_state(tmp_path):
    model = build_model(ModelConfig(latent_dim=8, channels=[8, 16, 16]), 3, 32)
    save_checkpoint(tmp_path / "m.pt", model)
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "m.pt", model, torch.optim.Adam(model.parameters()))


def test_rng_state_round_trip():
    seed_everything(1)
    state = get_rng_state()
    import random

    import numpy as np

    a = (random.random(), float(np.random.rand()), torch.rand(2).tolist())
    set_rng_state(state)
    assert (random.random(), float(np.random.rand()), torch.rand(2).tolist()) == a


# ---- trainer ----
@pytest.mark.parametrize("model", ["baseline", "asymmetric"])
def test_both_models_train_through_same_trainer(tmp_path, model):
    cfg = _cfg(model, epochs=3)
    trainer = _trainer(cfg, tmp_path / "run")
    history = trainer.fit()
    assert [h["epoch"] for h in history] == [1, 2, 3]
    assert all({"train_loss", "val_loss", "val_mse", "lr", "seconds"} <= set(h) for h in history)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    run = tmp_path / "run"
    for name in ("config.yaml", "environment.json", "history.json", "summary.json"):
        assert (run / name).exists()
    assert (run / "checkpoints" / "last.pt").exists()
    assert (run / "checkpoints" / "best.pt").exists()
    assert json.loads((run / "history.json").read_text()) == history
    summary = json.loads((run / "summary.json").read_text())
    assert summary["epochs_completed"] == 3 and not summary["stopped_early"]


def test_best_checkpoint_matches_best_validation_epoch(tmp_path):
    trainer = _trainer(_cfg(epochs=4), tmp_path / "run")
    history = trainer.fit()
    best = min(history, key=lambda h: h["val_mse"])
    ck = torch.load(trainer.best_path, weights_only=True)
    assert ck["epoch"] == best["epoch"] == trainer.best_epoch
    model = build_model(trainer.cfg.model, 3, 32)
    load_checkpoint(trainer.best_path, model)
    assert evaluate(model, trainer.val_loader, CPU)["mse"] == pytest.approx(best["val_mse"])


@pytest.mark.parametrize("model", ["baseline", "asymmetric"])
def test_resume_matches_uninterrupted_run(tmp_path, model):
    cfg_kw = {"epochs": 4, "scheduler": "cosine"}
    full = _trainer(_cfg(model, **cfg_kw), tmp_path / "full")
    full_hist = full.fit()

    # Simulate an interruption after epoch 2: the cosine schedule still spans 4 epochs
    # (T_max=4), but the loop stops early.
    interrupted = _trainer(_cfg(model, **cfg_kw), tmp_path / "part")
    interrupted.cfg.train.epochs = 2
    interrupted.fit()
    assert interrupted.start_epoch == 2

    # A brand-new process-equivalent: fresh model/optimizer/loader, state from last.pt.
    resumed = _trainer(_cfg(model, **cfg_kw), tmp_path / "part")
    resumed_hist = resumed.fit(resume=True)
    assert [h["epoch"] for h in resumed_hist] == [1, 2, 3, 4]
    for a, b in zip(full_hist, resumed_hist, strict=True):
        assert a["train_loss"] == b["train_loss"] and a["val_mse"] == b["val_mse"]
        assert a["lr"] == b["lr"]
    for a, b in zip(
        full.model.state_dict().values(), resumed.model.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)


def test_refuses_to_overwrite_existing_run_and_resume_needs_checkpoint(tmp_path):
    _trainer(_cfg(epochs=1), tmp_path / "run").fit()
    with pytest.raises(FileExistsError):
        _trainer(_cfg(epochs=1), tmp_path / "run").fit()
    with pytest.raises(FileNotFoundError):
        _trainer(_cfg(epochs=1), tmp_path / "other").fit(resume=True)


def test_resume_after_completion_is_noop(tmp_path):
    t1 = _trainer(_cfg(epochs=2), tmp_path / "run")
    h1 = t1.fit()
    t2 = _trainer(_cfg(epochs=2), tmp_path / "run")
    assert t2.fit(resume=True) == h1


def test_early_stopping(tmp_path):
    cfg = _cfg(epochs=10, early_stopping_patience=2, early_stopping_min_delta=10.0)
    trainer = _trainer(cfg, tmp_path / "run")
    history = trainer.fit()
    assert len(history) == 3  # epoch 1 sets best; epochs 2 and 3 fail min_delta
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["stopped_early"] and summary["epochs_completed"] == 3


def test_cosine_scheduler_decays_learning_rate(tmp_path):
    history = _trainer(_cfg(epochs=4, scheduler="cosine"), tmp_path / "run").fit()
    lrs = [h["lr"] for h in history]
    assert lrs[0] == pytest.approx(0.003)
    assert all(a > b for a, b in zip(lrs, lrs[1:], strict=False))


def test_non_finite_loss_raises(tmp_path):
    trainer = _trainer(
        _cfg(epochs=1),
        tmp_path / "run",
        loss_fn=lambda m, x: m(x).mean() * float("nan"),
    )
    with pytest.raises(FloatingPointError):
        trainer.fit()


def test_evaluate_returns_comparable_mse_for_both_models():
    _, val = _loaders()
    for name in SMALL:
        model = build_model(_cfg(name).model, 3, 32)
        out = evaluate(model, val, CPU)
        assert set(out) == {"mse", "loss"} and out["mse"] > 0


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="needs an accelerator",
)
def test_resume_on_accelerator(tmp_path):
    """Regression: resuming used to fail when the checkpoint was mapped onto the device."""
    from asymmetric_autoencoder.utils.reproducibility import resolve_device

    device = resolve_device("auto")
    cfg = _cfg(epochs=2)
    for epochs in (1, 2):
        cfg.train.epochs = epochs
        seed_everything(cfg.seed)
        train, val = _loaders(cfg.seed)
        trainer = Trainer(
            build_model(cfg.model, 3, 32), train, val, cfg, device, run_dir=tmp_path / "run"
        )
        history = trainer.fit(resume=epochs == 2)
    assert [h["epoch"] for h in history] == [1, 2]
