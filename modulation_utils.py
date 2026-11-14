"""Shared signal-generation and feature-extraction utilities."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import kurtosis, skew

MOD_TYPES = ("QPSK", "8PSK", "16APSK", "32APSK", "64APSK")
TIME_FRAMES = 1_000

def generate_qpsk(n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate unit-energy QPSK symbols."""
    rng = rng or np.random.default_rng()
    constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    return rng.choice(constellation, size=n_samples)

def generate_psk(n_samples: int, order: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate M-PSK symbols."""
    if order < 2:
        raise ValueError("PSK order must be at least 2.")
    rng = rng or np.random.default_rng()
    return rng.choice(np.exp(2j * np.pi * np.arange(order) / order), size=n_samples)

def generate_apsk(n_samples: int, order: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate APSK symbols with ring populations that sum to the requested order."""
    ring_counts = {16: (4, 12), 32: (4, 12, 16), 64: (4, 12, 20, 28)}
    if order not in ring_counts:
        raise ValueError("Supported APSK orders are 16, 32, and 64.")
    rng = rng or np.random.default_rng()
    points = []
    for index, count in enumerate(ring_counts[order]):
        radius = float(index + 1)
        offset = np.pi / count if index else 0.0
        angles = 2 * np.pi * np.arange(count) / count + offset
        points.extend(radius * np.exp(1j * angles))
    return rng.choice(np.asarray(points, dtype=np.complex128), size=n_samples)

def generate_signal(modulation: str, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a signal for one supported modulation label."""
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if modulation == "QPSK":
        return generate_qpsk(n_samples, rng)
    if modulation == "8PSK":
        return generate_psk(n_samples, 8, rng)
    if modulation.endswith("APSK"):
        try:
            order = int(modulation.removesuffix("APSK"))
        except ValueError as exc:
            raise ValueError(f"Unsupported modulation type: {modulation}") from exc
        return generate_apsk(n_samples, order, rng)
    raise ValueError(f"Unsupported modulation type: {modulation}")

def _safe_moments(values: np.ndarray) -> tuple[float, float]:
    """Calculate skewness and kurtosis without warnings for constant inputs."""
    if np.ptp(values) <= np.finfo(np.float64).eps:
        return 0.0, 0.0
    return float(skew(values, bias=False)), float(kurtosis(values, bias=False))

def extract_features(signal: np.ndarray) -> np.ndarray:
    """Return a finite, fixed-length feature vector for a complex baseband signal."""
    data = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if data.size < 2:
        raise ValueError("A signal must contain at least two complex samples.")
    if not np.all(np.isfinite(data.real)) or not np.all(np.isfinite(data.imag)):
        raise ValueError("Signal samples must be finite numbers.")
    real, imag = data.real, data.imag
    magnitude, phase = np.abs(data), np.angle(data)
    rs, rk = _safe_moments(real)
    ims, imk = _safe_moments(imag)
    ms, mk = _safe_moments(magnitude)
    features = np.array([
        np.mean(real), np.std(real), rs, rk,
        np.mean(imag), np.std(imag), ims, imk,
        np.mean(magnitude), np.std(magnitude), ms, mk,
        np.mean(np.cos(phase)), np.mean(np.sin(phase)),
        np.std(np.cos(phase)), np.std(np.sin(phase)),
        np.min(real), np.max(real), np.min(imag), np.max(imag),
        np.min(magnitude), np.max(magnitude),
        np.mean(np.abs(np.diff(data))), np.std(np.abs(np.diff(data))),
    ], dtype=np.float64)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

def load_signal_file(path: str | Path) -> np.ndarray:
    """Load a comma-delimited real/imaginary sample file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Signal file not found: {file_path}")
    values = np.asarray(np.loadtxt(file_path, delimiter=","), dtype=np.float64)
    if values.ndim == 1:
        if values.size != 2:
            raise ValueError("Expected two columns: real,imaginary.")
        values = values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected a CSV file with exactly two columns: real,imaginary.")
    if values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Signal file must contain at least two finite samples.")
    return values[:, 0] + 1j * values[:, 1]
