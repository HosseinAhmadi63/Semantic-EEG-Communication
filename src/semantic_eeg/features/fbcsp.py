"""One-versus-rest FBCSP features followed by training-only PCA compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from mne.decoding import CSP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn

from semantic_eeg.constants import N_CHANNELS, N_CLASSES


FittedCSP = tuple[int, int, CSP]


def _validate_filterbank_trials(
    trials: np.ndarray,
    number_of_bands: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(trials)
    expected_tail = (N_CHANNELS, values.shape[2] if values.ndim == 4 else -1, number_of_bands)
    if values.ndim != 4 or values.shape[1] != N_CHANNELS or values.shape[-1] != number_of_bands:
        raise ValueError(
            f"{name} must have shape (trials, {N_CHANNELS}, time, {number_of_bands}), "
            f"received {values.shape}; expected tail {expected_tail}"
        )
    if len(values) == 0 or values.shape[2] == 0:
        raise ValueError(f"{name} must contain at least one nonempty EEG trial")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def _validate_labels(labels: np.ndarray, number_of_trials: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(values) != number_of_trials:
        raise ValueError(
            f"Expected one label for each of {number_of_trials} trials, received {len(values)}"
        )
    if set(values.tolist()) != set(range(N_CLASSES)):
        received = sorted(set(values.tolist()))
        raise ValueError(f"FBCSP fitting requires all {N_CLASSES} classes, received {received}")
    return values


def _raw_features(
    models: Sequence[FittedCSP],
    trials: np.ndarray,
    number_of_bands: int,
) -> np.ndarray:
    values = _validate_filterbank_trials(trials, number_of_bands, "trials")
    blocks = [
        model.transform(np.asarray(values[:, :, :, band_index], dtype=np.float64))
        for band_index, _, model in models
    ]
    if not blocks:
        raise ValueError("At least one fitted CSP model is required")
    features = np.concatenate(blocks, axis=1)
    if not np.isfinite(features).all():
        raise FloatingPointError("FBCSP produced non-finite features")
    return features


@dataclass
class FBCSPPCA:
    """Fitted FBCSP, standardization, and PCA operations for one data split."""

    models: list[FittedCSP]
    scaler: StandardScaler
    pca: PCA
    frequency_bands: tuple[tuple[float, float], ...]
    components_per_class: int
    feature_manifest: list[dict[str, int | float]]

    @property
    def budget(self) -> int:
        """Return the number of real-valued elements in each transmitted message."""
        return int(self.pca.n_components_)

    @property
    def raw_feature_count(self) -> int:
        """Return the number of FBCSP features before PCA."""
        return len(self.frequency_bands) * N_CLASSES * self.components_per_class

    def raw_features(self, trials: np.ndarray) -> np.ndarray:
        """Transform filter-bank EEG trials into unstandardized log-variance features."""
        return _raw_features(self.models, trials, len(self.frequency_bands))

    def transform(self, trials: np.ndarray) -> np.ndarray:
        """Create the unnormalized, budget-matched message used by the conventional system."""
        raw = self.raw_features(trials)
        message = self.pca.transform(self.scaler.transform(raw))
        if message.shape[1] != self.budget or not np.isfinite(message).all():
            raise FloatingPointError("FBCSP-PCA produced an invalid message array")
        return np.asarray(message, dtype=np.float32)

    def to_torch(self, device: torch.device | str = "cpu") -> "ExactFBCSPTorch":
        """Build the differentiable front end used to initialize semantic refinement."""
        return ExactFBCSPTorch(self).to(torch.device(device)).eval()


def fit_fbcsp_pca(
    train_trials: np.ndarray,
    train_labels: np.ndarray,
    budget: int,
    frequency_bands: Sequence[Sequence[float]],
    components_per_class: int = 4,
) -> FBCSPPCA:
    """Fit the paper's five-band, four-class FBCSP-PCA transmitter.

    All CSP filters, feature standardization statistics, and PCA components are
    estimated from ``train_trials`` only. Each CSP problem distinguishes one
    motor-imagery class from the other three classes.
    """

    bands = tuple((float(band[0]), float(band[1])) for band in frequency_bands)
    if not bands or any(high <= low for low, high in bands):
        raise ValueError(f"Invalid frequency bands: {bands}")
    if components_per_class < 1 or components_per_class > N_CHANNELS:
        raise ValueError(f"components_per_class must be between 1 and {N_CHANNELS}")
    trials = _validate_filterbank_trials(train_trials, len(bands), "train_trials")
    labels = _validate_labels(train_labels, len(trials))

    raw_feature_count = len(bands) * N_CLASSES * components_per_class
    if budget < 1 or budget > min(raw_feature_count, len(trials)):
        raise ValueError(
            f"budget must be between 1 and {min(raw_feature_count, len(trials))}, received {budget}"
        )

    models: list[FittedCSP] = []
    feature_blocks: list[np.ndarray] = []
    manifest: list[dict[str, int | float]] = []
    feature_index = 0
    for band_index, (low_hz, high_hz) in enumerate(bands):
        band_trials = np.asarray(trials[:, :, :, band_index], dtype=np.float64)
        for class_id in range(N_CLASSES):
            csp = CSP(
                n_components=components_per_class,
                reg="ledoit_wolf",
                log=True,
                cov_est="concat",
                norm_trace=False,
                component_order="mutual_info",
            )
            binary_labels = (labels == class_id).astype(np.int64)
            feature_blocks.append(csp.fit_transform(band_trials, binary_labels))
            models.append((band_index, class_id, csp))
            for component in range(components_per_class):
                manifest.append(
                    {
                        "feature_index": feature_index,
                        "band_index": band_index,
                        "low_hz": low_hz,
                        "high_hz": high_hz,
                        "ovr_class_id": class_id,
                        "csp_component": component,
                    }
                )
                feature_index += 1

    raw_train = np.concatenate(feature_blocks, axis=1)
    if raw_train.shape != (len(trials), raw_feature_count):
        raise AssertionError(
            f"Expected an FBCSP matrix with shape {(len(trials), raw_feature_count)}, "
            f"received {raw_train.shape}"
        )
    scaler = StandardScaler().fit(raw_train)
    scaled_train = scaler.transform(raw_train)
    pca = PCA(n_components=budget, svd_solver="full").fit(scaled_train)
    return FBCSPPCA(
        models=models,
        scaler=scaler,
        pca=pca,
        frequency_bands=bands,
        components_per_class=components_per_class,
        feature_manifest=manifest,
    )


class ExactFBCSPTorch(nn.Module):
    """Frozen Torch reproduction of fitted MNE CSP, scaling, and PCA operations."""

    def __init__(self, pipeline: FBCSPPCA) -> None:
        super().__init__()
        spatial_weights: list[np.ndarray] = []
        for band_index in range(len(pipeline.frequency_bands)):
            selected = [
                (class_id, csp)
                for model_band, class_id, csp in pipeline.models
                if model_band == band_index
            ]
            if [class_id for class_id, _ in selected] != list(range(N_CLASSES)):
                raise ValueError(f"Unexpected one-versus-rest CSP ordering in band {band_index}")
            spatial_weights.append(
                np.concatenate(
                    [
                        np.asarray(csp.filters_[: pipeline.components_per_class], dtype=np.float64)
                        for _, csp in selected
                    ],
                    axis=0,
                )
            )
        spatial = np.stack(spatial_weights, axis=0)
        expected = (
            len(pipeline.frequency_bands),
            N_CLASSES * pipeline.components_per_class,
            N_CHANNELS,
        )
        if spatial.shape != expected:
            raise ValueError(
                f"Expected spatial weights with shape {expected}, received {spatial.shape}"
            )

        self.number_of_bands = len(pipeline.frequency_bands)
        self.register_buffer("spatial_weights", torch.from_numpy(spatial))
        self.register_buffer(
            "scaler_mean", torch.from_numpy(np.asarray(pipeline.scaler.mean_, np.float64))
        )
        self.register_buffer(
            "scaler_scale", torch.from_numpy(np.asarray(pipeline.scaler.scale_, np.float64))
        )
        self.register_buffer(
            "pca_mean", torch.from_numpy(np.asarray(pipeline.pca.mean_, np.float64))
        )
        self.register_buffer(
            "pca_components", torch.from_numpy(np.asarray(pipeline.pca.components_, np.float64))
        )

    def raw_features(self, trials: torch.Tensor) -> torch.Tensor:
        """Return log-average-power features in the fitted MNE CSP ordering."""
        if trials.ndim != 4:
            raise ValueError(
                f"Expected four-dimensional filter-bank trials, received {tuple(trials.shape)}"
            )
        if trials.shape[1] != N_CHANNELS or trials.shape[-1] != self.number_of_bands:
            raise ValueError(f"Unexpected filter-bank trial shape: {tuple(trials.shape)}")
        values = trials.to(dtype=torch.float64)
        blocks = []
        for band_index in range(self.number_of_bands):
            projected = torch.einsum(
                "fc,bct->bft",
                self.spatial_weights[band_index],
                values[:, :, :, band_index],
            )
            blocks.append(torch.log(projected.square().mean(dim=-1).clamp_min(1e-30)))
        return torch.cat(blocks, dim=1)

    def forward(self, trials: torch.Tensor) -> torch.Tensor:
        raw = self.raw_features(trials)
        scaled = (raw - self.scaler_mean) / self.scaler_scale
        return (scaled - self.pca_mean) @ self.pca_components.T


@torch.inference_mode()
def transform_exact_fbcsp(
    front_end: ExactFBCSPTorch,
    trials: np.ndarray,
    device: torch.device | str,
    batch_size: int = 256,
    include_raw_features: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate the frozen Torch FBCSP-PCA front end in deterministic batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = torch.device(device)
    values = _validate_filterbank_trials(trials, front_end.number_of_bands, "trials")
    front_end = front_end.to(destination).eval()
    messages: list[np.ndarray] = []
    raw_features: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(np.asarray(values[start : start + batch_size], np.float64)).to(
            destination
        )
        if include_raw_features:
            raw_features.append(front_end.raw_features(batch).cpu().numpy())
        messages.append(front_end(batch).cpu().numpy())
    message = np.concatenate(messages)
    if not np.isfinite(message).all():
        raise FloatingPointError("The frozen FBCSP-PCA front end produced non-finite messages")
    if include_raw_features:
        return message, np.concatenate(raw_features)
    return message
