"""Shared training pipeline used by every autoencoder.

One ``Trainer`` handles optimisation, per-epoch validation, history, early stopping,
checkpointing and exact resume. Run directory layout::

    <run_dir>/config.yaml        experiment configuration snapshot
    <run_dir>/environment.json   python / library / device versions
    <run_dir>/history.json       one record per epoch
    <run_dir>/summary.json       final summary (written when fit() finishes)
    <run_dir>/checkpoints/last.pt, best.pt

Validation reports two numbers: ``val_mse`` (final-resolution MSE against the clean input,
comparable across models; used for best-checkpoint selection and early stopping) and
``val_loss`` (the model's own training objective, not comparable across models).

Exact resume: the shuffling generator is re-seeded each epoch from ``seed + epoch`` and RNG
states are stored in the checkpoint, so a resumed run follows the same trajectory as an
uninterrupted one on the same device (verified by tests on CPU).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from asymmetric_autoencoder.config import ExperimentConfig, save_config
from asymmetric_autoencoder.models.losses import model_loss
from asymmetric_autoencoder.models.summary import count_parameters
from asymmetric_autoencoder.training.checkpoints import load_checkpoint, save_checkpoint
from asymmetric_autoencoder.utils.logging import get_logger
from asymmetric_autoencoder.utils.reproducibility import (
    environment_info,
    get_rng_state,
    set_rng_state,
)

log = get_logger(__name__)

LossFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]


def run_dir_for(cfg: ExperimentConfig) -> Path:
    """Default run directory for a configuration."""
    return Path(cfg.train.output_dir) / f"{cfg.experiment_name}_seed{cfg.seed}"


def _images(batch: Any) -> torch.Tensor:
    """Datasets yield (image, label); labels are ignored (self-supervised)."""
    return batch[0] if isinstance(batch, (tuple, list)) else batch


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    loss_fn: LossFn = model_loss,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Return ``{"mse": final-resolution MSE, "loss": model objective}`` (element-weighted)."""
    model.eval()
    mse_sum = loss_sum = 0.0
    n_el = n_samples = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = _images(batch).to(device)
        mse_sum += F.mse_loss(model(x), x, reduction="sum").item()
        loss_sum += loss_fn(model, x).item() * x.shape[0]
        n_el += x.numel()
        n_samples += x.shape[0]
    if n_samples == 0:
        raise ValueError("evaluation loader produced no batches")
    return {"mse": mse_sum / n_el, "loss": loss_sum / n_samples}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable,
        val_loader: Iterable,
        cfg: ExperimentConfig,
        device: torch.device,
        run_dir: str | Path | None = None,
        loss_fn: LossFn = model_loss,
        max_batches: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.run_dir = Path(run_dir) if run_dir is not None else run_dir_for(cfg)
        self.loss_fn = loss_fn
        self.max_batches = max_batches

        t = cfg.train
        params = self.model.parameters()
        if t.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(params, lr=t.lr, weight_decay=t.weight_decay)
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t.epochs)
            if t.scheduler == "cosine"
            else None
        )

        self.history: list[dict[str, Any]] = []
        self.start_epoch = 0
        self.best_val_mse = math.inf
        self.best_epoch = -1
        self.bad_epochs = 0

    # ---- paths ----
    @property
    def last_path(self) -> Path:
        return self.run_dir / "checkpoints" / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "checkpoints" / "best.pt"

    # ---- state ----
    def _restore(self) -> None:
        if not self.last_path.exists():
            raise FileNotFoundError(f"cannot resume: {self.last_path} does not exist")
        # Load on CPU: model/optimizer tensors are copied to the parameters' device by
        # load_state_dict, and RNG states must stay on the host.
        ck = load_checkpoint(self.last_path, self.model, self.optimizer, self.scheduler)
        self.start_epoch = ck["epoch"]
        self.history = ck["history"]
        self.best_val_mse = ck["best_val_mse"]
        self.best_epoch = ck["best_epoch"]
        self.bad_epochs = ck["bad_epochs"]
        set_rng_state(ck["rng"])
        log.info("resumed from %s at epoch %d", self.last_path, self.start_epoch)

    def _save_last(self) -> None:
        save_checkpoint(
            self.last_path,
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=self.start_epoch,
            history=self.history,
            best_val_mse=self.best_val_mse,
            best_epoch=self.best_epoch,
            bad_epochs=self.bad_epochs,
            config=self.cfg.to_dict(),
            rng=get_rng_state(),
        )

    # ---- loop ----
    def _train_epoch(self, epoch: int) -> float:
        generator = getattr(self.train_loader, "generator", None)
        if generator is not None:
            generator.manual_seed(self.cfg.seed + epoch)
        self.model.train()
        total, n = 0.0, 0
        for i, batch in enumerate(self.train_loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            x = _images(batch).to(self.device)
            loss = self.loss_fn(self.model, x)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch + 1}, batch {i}"
                )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * x.shape[0]
            n += x.shape[0]
        if n == 0:
            raise ValueError("training loader produced no batches")
        return total / n

    def fit(self, resume: bool = False) -> list[dict[str, Any]]:
        """Train for ``cfg.train.epochs`` epochs (continuing from ``last.pt`` if ``resume``)."""
        t = self.cfg.train
        if resume:
            self._restore()
        elif self.last_path.exists():
            raise FileExistsError(
                f"{self.last_path} exists; pass resume=True or choose a new run directory"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.cfg, self.run_dir / "config.yaml")
        env = {
            **environment_info(),
            "device": str(self.device),
            "parameters": str(count_parameters(self.model)),
        }
        (self.run_dir / "environment.json").write_text(json.dumps(env, indent=2))

        stopped_early = False
        fit_start = time.perf_counter()
        for epoch in range(self.start_epoch, t.epochs):
            start = time.perf_counter()
            train_loss = self._train_epoch(epoch)
            val = evaluate(self.model, self.val_loader, self.device, self.loss_fn, self.max_batches)
            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            improved = val["mse"] < self.best_val_mse - t.early_stopping_min_delta
            if improved:
                self.best_val_mse, self.best_epoch, self.bad_epochs = val["mse"], epoch + 1, 0
            else:
                self.bad_epochs += 1
            self.start_epoch = epoch + 1
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_mse": val["mse"],
                    "lr": lr,
                    "seconds": time.perf_counter() - start,
                }
            )
            self._save_last()
            if improved:
                save_checkpoint(
                    self.best_path,
                    self.model,
                    epoch=epoch + 1,
                    val_mse=val["mse"],
                    config=self.cfg.to_dict(),
                )
            (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2))
            log.info(
                "epoch %d/%d train_loss=%.5f val_mse=%.5f lr=%.2e%s",
                epoch + 1,
                t.epochs,
                train_loss,
                val["mse"],
                lr,
                " *" if improved else "",
            )
            if t.early_stopping_patience and self.bad_epochs >= t.early_stopping_patience:
                stopped_early = True
                log.info("early stopping after epoch %d", epoch + 1)
                break

        summary = {
            "epochs_completed": self.start_epoch,
            "epochs_configured": t.epochs,
            "stopped_early": stopped_early,
            "best_epoch": self.best_epoch,
            "best_val_mse": self.best_val_mse,
            "parameters": count_parameters(self.model),
            "device": str(self.device),
            "seconds_this_session": time.perf_counter() - fit_start,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return self.history
