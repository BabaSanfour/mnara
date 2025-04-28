import numpy as np
from scipy.spatial.distance import squareform
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from typing import Sequence, Optional, Union

from mnara.config import DEFAULTS
from mnara.distances import DistanceFactory


class RDM:
    """
    Core RDM computations (math only):
      - generate: static RDM from 2D [n_conditions, n_features]
      - rdms: batch RDMs from list/array of 2D inputs
      - time_rdms: per-timepoint RDMs from 3D [n_elements, n_conditions, n_time]

    Features:
      - pluggable backends (numpy, numba, jax, torch, cupy)
      - CPU multiprocessing (joblib) and GPU auto-parallelism
      - dtype control, optional z-score normalization
      - progress bars, condensed-vector output, chunked processing
    """

    def __init__(
        self,
        metric: str = "correlation",
        backend: Optional[str] = None,
        n_jobs: Optional[int] = None,
        parallel: bool = False,
        dtype: np.dtype = np.float32,
        normalize: bool = False,
        verbose: bool = False,
    ):
        self.metric = metric
        self.backend = backend or DEFAULTS["backend"]
        self.n_jobs = DEFAULTS["n_jobs"] if n_jobs is None else n_jobs
        self.parallel = parallel
        self.dtype = dtype
        self.normalize = normalize
        self.verbose = verbose
        self._dist_fn = DistanceFactory.get(self.metric, self.backend)

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        if self.normalize:
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            data = (data - mean) / std
        return data.astype(self.dtype, copy=False)

    def generate(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        splits: Optional[np.ndarray] = None,
        return_vector: bool = False,
    ) -> np.ndarray:
        """
        Static RDM from 2D data.
        Args:
            data: (n_conditions, n_features)
            return_vector: if True, return condensed vector
        Returns:
            matrix or vector
        """
        if data.ndim != 2:
            raise ValueError(f"Input must be 2D array, got {data.shape}")
        data = self._preprocess(data)
        vec = self._dist_fn(data, labels, splits) if self.metric=="crossnobis" else self._dist_fn(data)
        vec = vec.astype(self.dtype)
        if return_vector:
            return vec
        return squareform(vec).astype(self.dtype)

    def rdms(
        self,
        data_list: Union[Sequence[np.ndarray], np.ndarray],
        labels: Optional[np.ndarray] = None,
        splits: Optional[np.ndarray] = None,
        return_vector: bool = False,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Batch RDMs over many 2D inputs.
        Args:
            data_list: sequence or 3D array
        """
        # normalize list
        if isinstance(data_list, np.ndarray):
            if data_list.ndim != 3:
                raise ValueError("ndarray must be 3D: (n_items,n_conditions,n_features)")
            arrs = [data_list[i] for i in range(data_list.shape[0])]
        else:
            arrs = list(data_list)
        n_items = len(arrs)
        n_cond = arrs[0].shape[0]
        for arr in arrs:
            if arr.ndim!=2 or arr.shape[0]!=n_cond:
                raise ValueError("All inputs must be 2D with same n_conditions")
        vec_len = n_cond*(n_cond-1)//2
        out = (np.zeros((n_items,vec_len),dtype=self.dtype) if return_vector
               else np.zeros((n_items,n_cond,n_cond),dtype=self.dtype))
        # chunking
        indices = list(range(n_items))
        if chunk_size is None:
            chunks = [indices]
        else:
            if chunk_size<1 or chunk_size>n_items:
                raise ValueError("chunk_size must be 1..n_items")
            chunks = [indices[i:i+chunk_size] for i in range(0,n_items,chunk_size)]
        for chunk in chunks:
            tasks = chunk
            if self.verbose and not self.parallel:
                tasks = tqdm(tasks, desc="Static RDMs chunk")
            def _worker(i): return i, self.generate(arrs[i],labels,splits,return_vector)
            if self.parallel and self.backend in ("numpy","numba"):
                from joblib import Parallel,delayed
                res = Parallel(n_jobs=self.n_jobs)(delayed(_worker)(i) for i in tasks)
                for i,mat in res: out[i]=mat
            else:
                for i in tasks: out[i]=self.generate(arrs[i],labels,splits,return_vector)
        return out

    def time_rdms(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        splits: Optional[np.ndarray] = None,
        return_vector: bool = False,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Per-timepoint RDMs from 3D brain data.
        Args:
            data: (n_elems,n_conditions,n_time)
        Returns:
            array (n_elems,n_time,n_conditions,n_conditions) or vectors
        """
        if not(isinstance(data,np.ndarray) and data.ndim==3):
            raise ValueError("Data must be 3D array (n_elems,n_conditions,n_time)")
        n_elems,n_cond,n_time=data.shape
        vec_len=n_cond*(n_cond-1)//2
        out=(np.zeros((n_elems,n_time,vec_len),dtype=self.dtype)
             if return_vector else np.zeros((n_elems,n_time,n_cond,n_cond),dtype=self.dtype))
        # chunk timepoints
        times=list(range(n_time))
        if chunk_size is None: chunks=[times]
        else:
            if chunk_size<1 or chunk_size>n_time: raise ValueError("chunk_size must be 1..n_time")
            chunks=[times[i:i+chunk_size] for i in range(0,n_time,chunk_size)]
        for chunk in chunks:
            tasks=[(e,t) for e in range(n_elems) for t in chunk]
            if self.verbose and not self.parallel:
                tasks=tqdm(tasks,desc="Time RDMs chunk")
            def _worker(et): e,t=et; feat=data[e,:,t].reshape(n_cond,1);mat=self.generate(feat,labels,splits,return_vector);return e,t,mat
            if self.parallel and self.backend in ("numpy","numba"):
                from joblib import Parallel,delayed
                res=Parallel(n_jobs=self.n_jobs)(delayed(_worker)(et) for et in tasks)
                for e,t,mat in res: out[e,t]=mat
            else:
                for e,t in tasks: _,_,mat=_worker((e,t)); out[e,t]=mat
        return out