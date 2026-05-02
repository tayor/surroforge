"""PyTorch MLP regressor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from surroforge.models.base import ModelNotFittedError


def torch_available() -> bool:
    """Return True when the optional torch dependency is importable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "MLPRegressor requires the optional torch dependency. "
            "Install it with `pip install 'surroforge[torch]'`."
        ) from exc
    return torch, nn


class MLPRegressor:
    """Small fully connected PyTorch regressor for scalar/vector outputs."""

    def __init__(
        self,
        hidden: list[int] | tuple[int, ...] = (64, 64),
        *,
        lr: float = 1e-3,
        epochs: int = 250,
        batch_size: int = 32,
        patience: int = 25,
        validation_fraction: float = 0.2,
        seed: int = 13,
        device: str | None = None,
    ) -> None:
        self.hidden = list(hidden)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.device = device
        self.output_names: list[str] = []
        self.history_: list[dict[str, float]] = []
        self.model_: Any | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: np.ndarray | None = None
        self.y_scale_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        output_names: list[str] | None = None,
    ) -> MLPRegressor:
        """Fit the neural surrogate."""
        torch, nn = _require_torch()
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        if x_array.ndim != 2 or y_array.ndim != 2:
            raise ValueError("x and y must be 2D arrays")
        if len(x_array) != len(y_array):
            raise ValueError("x and y row counts must match")
        if len(x_array) < 2:
            raise ValueError("MLPRegressor needs at least two training samples")

        self.output_names = output_names or [f"y{i}" for i in range(y_array.shape[1])]
        self.x_mean_ = x_array.mean(axis=0)
        self.x_scale_ = _safe_std(x_array)
        self.y_mean_ = y_array.mean(axis=0)
        self.y_scale_ = _safe_std(y_array)
        assert self.x_mean_ is not None
        assert self.x_scale_ is not None
        assert self.y_mean_ is not None
        assert self.y_scale_ is not None
        x_norm = (x_array - self.x_mean_) / self.x_scale_
        y_norm = (y_array - self.y_mean_) / self.y_scale_

        train_index, val_index = _split_indices(len(x_norm), self.validation_fraction, rng=rng)
        x_train = torch.as_tensor(
            x_norm[train_index], dtype=torch.float32, device=self._device(torch)
        )
        y_train = torch.as_tensor(
            y_norm[train_index], dtype=torch.float32, device=self._device(torch)
        )
        x_val = torch.as_tensor(x_norm[val_index], dtype=torch.float32, device=self._device(torch))
        y_val = torch.as_tensor(y_norm[val_index], dtype=torch.float32, device=self._device(torch))

        self.model_ = _build_network(nn, x_array.shape[1], y_array.shape[1], self.hidden).to(
            self._device(torch)
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        self.history_ = []

        for _epoch in range(self.epochs):
            self.model_.train()
            permutation = torch.randperm(len(x_train), device=self._device(torch))
            batch_losses = []
            for start in range(0, len(x_train), self.batch_size):
                batch_index = permutation[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(x_train[batch_index]), y_train[batch_index])
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(x_val), y_val).detach().cpu())
            train_loss = float(np.mean(batch_losses))
            self.history_.append({"train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict output values."""
        torch, _nn = _require_torch()
        self._check_fitted()
        x_array = np.asarray(x, dtype=np.float32)
        if x_array.ndim == 1:
            x_array = x_array[None, :]
        assert self.model_ is not None
        assert self.x_mean_ is not None
        assert self.x_scale_ is not None
        assert self.y_mean_ is not None
        assert self.y_scale_ is not None
        x_norm = (x_array - self.x_mean_) / self.x_scale_
        self.model_.eval()
        with torch.no_grad():
            normalized = self.model_(
                torch.as_tensor(x_norm, dtype=torch.float32, device=self._device(torch))
            )
        return normalized.detach().cpu().numpy() * self.y_scale_ + self.y_mean_

    def save(self, path: str | Path) -> Path:
        """Save a Torch checkpoint."""
        torch, _nn = _require_torch()
        self._check_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        assert self.model_ is not None
        assert self.x_mean_ is not None
        assert self.x_scale_ is not None
        assert self.y_mean_ is not None
        assert self.y_scale_ is not None
        torch.save(
            {
                "class": "MLPRegressor",
                "hidden": self.hidden,
                "input_dim": int(len(self.x_mean_)),
                "output_dim": int(len(self.y_mean_)),
                "output_names": self.output_names,
                "state_dict": self.model_.state_dict(),
                "x_mean": self.x_mean_,
                "x_scale": self.x_scale_,
                "y_mean": self.y_mean_,
                "y_scale": self.y_scale_,
                "history": self.history_,
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> MLPRegressor:
        """Load a checkpoint saved by :meth:`save`."""
        torch, nn = _require_torch()
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        model = cls(hidden=checkpoint["hidden"], device=device)
        model.output_names = list(checkpoint["output_names"])
        model.x_mean_ = np.asarray(checkpoint["x_mean"], dtype=np.float32)
        model.x_scale_ = np.asarray(checkpoint["x_scale"], dtype=np.float32)
        model.y_mean_ = np.asarray(checkpoint["y_mean"], dtype=np.float32)
        model.y_scale_ = np.asarray(checkpoint["y_scale"], dtype=np.float32)
        model.history_ = list(checkpoint.get("history", []))
        model.model_ = _build_network(
            nn, checkpoint["input_dim"], checkpoint["output_dim"], checkpoint["hidden"]
        ).to(model._device(torch))
        model.model_.load_state_dict(checkpoint["state_dict"])
        model.model_.eval()
        return model

    def _device(self, torch: Any) -> Any:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _check_fitted(self) -> None:
        if (
            self.model_ is None
            or self.x_mean_ is None
            or self.x_scale_ is None
            or self.y_mean_ is None
            or self.y_scale_ is None
        ):
            raise ModelNotFittedError("MLPRegressor has not been fitted")


def _safe_std(array: np.ndarray) -> np.ndarray:
    scale = array.std(axis=0)
    return np.where(scale < 1e-8, 1.0, scale).astype(np.float32)


def _split_indices(
    n_samples: int,
    validation_fraction: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    if n_samples < 5:
        return indices, indices
    val_count = max(1, int(round(n_samples * validation_fraction)))
    val_count = min(val_count, n_samples - 1)
    return indices[val_count:], indices[:val_count]


def _build_network(nn: Any, input_dim: int, output_dim: int, hidden: list[int]) -> Any:
    layers = []
    previous = input_dim
    for width in hidden:
        layers.append(nn.Linear(previous, width))
        layers.append(nn.ReLU())
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)
