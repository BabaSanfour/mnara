import os
import numpy as np
from typing import Optional
from mnara.rdm import RDM

__all__ = ['compute_rdms', 'compute_time_rdms']


def compute_rdms(
    input_path: str,
    output_path: str,
    metric: str = 'correlation',
    backend: str = 'numpy',
    n_jobs: int = -1,
    parallel: bool = False,
    dtype: str = 'float32',
    normalize: bool = False,
    return_vector: bool = False,
    chunk_size: Optional[int] = None,
    verbose: bool = False,
):
    """
    Orchestrate static/batch RDM computation:
      - Loads NPY/.NPZ input (2D or 3D)
      - Calls RDM.rdms
      - Saves output to NPY

    Args:
        input_path: file or directory of .npy arrays
        output_path: destination .npy
    """
    # load inputs
    if os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if f.endswith('.npy'))
        data_list = [np.load(os.path.join(input_path,f)) for f in files]
    else:
        arr = np.load(input_path)
        if arr.ndim == 3:
            data_list = [arr[i] for i in range(arr.shape[0])]
        elif arr.ndim == 2:
            data_list = [arr]
        else:
            raise ValueError('Input array must be 2D or 3D')
    # compute
    rdm = RDM(metric, backend, n_jobs, parallel,
              dtype=getattr(np, dtype), normalize=normalize, verbose=verbose)
    out = rdm.rdms(data_list, None, None, return_vector, chunk_size)
    np.save(output_path, out)


def compute_time_rdms(
    input_path: str,
    output_path: str,
    metric: str = 'correlation',
    backend: str = 'numpy',
    n_jobs: int = -1,
    parallel: bool = False,
    dtype: str = 'float32',
    normalize: bool = False,
    return_vector: bool = False,
    chunk_size: Optional[int] = None,
    verbose: bool = False,
):
    """
    Orchestrate per-timepoint RDM computation:
      - Loads 3D NPY
      - Calls RDM.time_rdms
      - Saves output

    Args:
        input_path: .npy file (n_elems,n_cond,n_time)
        output_path: destination .npy
    """
    arr = np.load(input_path)
    if arr.ndim != 3:
        raise ValueError('Input array must be 3D')
    rdm = RDM(metric, backend, n_jobs, parallel,
              dtype=getattr(np, dtype), normalize=normalize, verbose=verbose)
    out = rdm.time_rdms(arr, None, None, return_vector, chunk_size)
    np.save(output_path, out)
    
    if verbose:
        print(f"Output saved to {output_path}")
