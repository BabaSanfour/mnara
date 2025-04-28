"""
mnara: Modeling Neural Activation & Representational Analytics
"""

__version__ = "0.0.1"

# core API
from .rdm import RDM

# pipeline
from .pipeline.rdm import compute_rdms, compute_time_rdms

# expose top-level classes
__all__ = ["RSAModel", "Searchlight", "Pipeline", "__version__"]
