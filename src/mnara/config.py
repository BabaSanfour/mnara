"""
Global configuration defaults for mnara.
Users can override via kwargs or environment variables.
"""

DEFAULTS = {
    "backend": "numpy",
    "use_gpu": False,
    "n_jobs": -1,
    "random_seed": 42,

    "dtype": "float32",       # default output dtype
    "normalize": False,       # whether to z-score before distances
    "verbose": False,         # show progress bars
}
