import warnings

import numpy as np
from scipy.spatial.distance import pdist

from mnara.config import DEFAULTS

try:
    import jax.numpy as jnp
    from jax import jit
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import numba
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ——————————————————————————————
# NumPy implementations (“reference”)
# ——————————————————————————————

def _corr_dist_numpy(data: np.ndarray) -> np.ndarray:
    """1 − Pearson r, via SciPy’s pdist/correlation."""
    # pdist with 'correlation' yields 1 − Pearson r
    return pdist(data, metric="correlation")


def _euclid_dist_numpy(data: np.ndarray) -> np.ndarray:
    """Euclidean distance via pdist."""
    return pdist(data, metric="euclidean")


def _maha_dist_numpy(data: np.ndarray) -> np.ndarray:
    """Mahalanobis distance via pdist."""
    return pdist(data, metric="mahalanobis")


# crossnobis needs custom implementation—fallback to Mahalanobis for now
def _crossnobis_dist_numpy(data: np.ndarray, labels=None, splits=None) -> np.ndarray:
    warnings.warn("NumPy crossnobis stub: using Mahalanobis fallback")
    return _maha_dist_numpy(data)


# ——————————————————————————————
# JAX backend stubs
# ——————————————————————————————

@jit
def _corr_dist_jax(data: jnp.ndarray) -> jnp.ndarray:
    x = data - jnp.mean(data, axis=1, keepdims=True)
    norm = jnp.linalg.norm(x, axis=1, keepdims=True)
    x = x / norm
    corr = x @ x.T
    distmat = 1.0 - corr
    iu = jnp.triu_indices(data.shape[0], k=1)
    return distmat[iu]

@jit
def _euclid_dist_jax(data: jnp.ndarray) -> jnp.ndarray:
    diffs = data[:, None, :] - data[None, :, :]
    distmat = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1))
    iu = jnp.triu_indices(data.shape[0], k=1)
    return distmat[iu]

def _maha_dist_jax(data, *args, **kwargs):
    warnings.warn("JAX Mahalanobis stub falling back to NumPy")
    return _maha_dist_numpy(np.array(data))

def _crossnobis_dist_jax(data, labels=None, splits=None):
    warnings.warn("JAX crossnobis stub falling back to Mahalanobis")
    return _crossnobis_dist_numpy(np.array(data))

# ——————————————————————————————
# PyTorch backend stubs
# ——————————————————————————————
def _get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = _get_torch_device()

def _corr_dist_torch(data):
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed")
    # data: numpy array → torch tensor
    t = torch.from_numpy(data.astype(np.float32)).to(DEVICE)
    # center & normalize
    t = t - t.mean(dim=1, keepdim=True)
    std = t.norm(dim=1, keepdim=True)
    t = t / std
    corr = t @ t.T
    dist = 1.0 - corr
    # upper triangle indices
    iu = torch.triu_indices(dist.shape[0], dist.shape[0], offset=1)
    vals = dist[iu[0], iu[1]]
    return vals.cpu().numpy()

def _euclid_dist_torch(data):
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed")
    t = torch.from_numpy(data.astype(np.float32)).to(DEVICE)
    dist = torch.cdist(t, t, p=2)
    iu = torch.triu_indices(dist.shape[0], dist.shape[0], offset=1)
    vals = dist[iu[0], iu[1]]
    return vals.cpu().numpy()

def _maha_dist_torch(data):
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed")
    warnings.warn("PyTorch Mahalanobis stub falling back to NumPy")
    return _maha_dist_numpy(np.array(data))

def _crossnobis_dist_torch(data, labels=None, splits=None):
    warnings.warn("PyTorch crossnobis stub falling back to Mahalanobis")
    return _crossnobis_dist_numpy(np.array(data))


# ——————————————————————————————
# CuPy backend stubs
# ——————————————————————————————

def _corr_dist_cupy(data):
    if not HAS_CUPY:
        raise ImportError("CuPy not installed")
    a = cp.array(data)
    # compute corrcoef
    corr = cp.corrcoef(a)
    dist = 1.0 - corr
    iu = cp.triu_indices(dist.shape[0], k=1)
    return cp.asnumpy(dist[iu])

def _euclid_dist_cupy(data):
    if not HAS_CUPY:
        raise ImportError("CuPy not installed")
    a = cp.array(data)
    # pairwise squared differences
    diffs = a[:, None, :] - a[None, :, :]
    dist = cp.sqrt(cp.sum(diffs * diffs, axis=-1))
    iu = cp.triu_indices(dist.shape[0], k=1)
    return cp.asnumpy(dist[iu])

def _maha_dist_cupy(data):
    if not HAS_CUPY:
        raise ImportError("CuPy not installed")
    warnings.warn("CuPy Mahalanobis stub falling back to NumPy")
    return _maha_dist_numpy(np.array(data))

def _crossnobis_dist_cupy(data, labels=None, splits=None):
    warnings.warn("CuPy crossnobis stub falling back to Mahalanobis")
    return _crossnobis_dist_numpy(np.array(data))


# ——————————————————————————————
# Numba backend stubs (decorators)
# ——————————————————————————————

if HAS_NUMBA:
    @njit
    def _corr_dist_numba(data):
        return _corr_dist_numpy(data)
    @njit
    def _euclid_dist_numba(data):
        return _euclid_dist_numpy(data)
    @njit
    def _maha_dist_numba(data, *args):
        return _maha_dist_numpy(data)
    @njit
    def _crossnobis_dist_numba(data, *args):
        return _crossnobis_dist_numpy(data)
else:
    def _corr_dist_numba(data):
        raise ImportError("Numba not installed")
    def _euclid_dist_numba(data):
        raise ImportError("Numba not installed")
    def _maha_dist_numba(data, *args):
        raise ImportError("Numba not installed")
    def _crossnobis_dist_numba(data, *args):
        raise ImportError("Numba not installed")


# ——————————————————————————————
# Registry mapping
# ——————————————————————————————

# metric_name → { backend_name: function }
_METRIC_BACKENDS = {
    "correlation": {
        "numpy": _corr_dist_numpy,
        "jax":    _corr_dist_jax,
        "torch":  _corr_dist_torch,
        "cupy":   _corr_dist_cupy,
        "numba":  _corr_dist_numba,
    },
    "euclidean": {
        "numpy": _euclid_dist_numpy,
        "jax":    _euclid_dist_jax,
        "torch":  _euclid_dist_torch,
        "cupy":   _euclid_dist_cupy,
        "numba":  _euclid_dist_numba,
    },
    "mahalanobis": {
        "numpy": _maha_dist_numpy,
        "jax":    _maha_dist_jax,
        "torch":  _maha_dist_torch,
        "cupy":   _maha_dist_cupy,
        "numba":  _maha_dist_numba,
    },
    "crossnobis": {
        "numpy": _crossnobis_dist_numpy,
        "jax":    _crossnobis_dist_jax,
        "torch":  _crossnobis_dist_torch,
        "cupy":   _crossnobis_dist_cupy,
        "numba":  _crossnobis_dist_numba,
    },
}


class DistanceFactory:
    """
    Factory to retrieve a metric function given its name and a backend.
    """
    @staticmethod
    def get(metric: str, backend: str = None):
        b = backend or DEFAULTS["backend"]
        try:
            fn = _METRIC_BACKENDS[metric][b]
        except KeyError:
            raise ValueError(
                f"Metric '{metric}' with backend '{b}' is not available. "
                f"Choose metric from {list(_METRIC_BACKENDS)} and backend from {list(_METRIC_BACKENDS[metric])}."
            )
        return fn
