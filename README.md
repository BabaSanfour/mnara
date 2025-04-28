# mnara

**Modeling Neural Activation & Representational Analytics**


## Features

- Multiple backends: NumPy, Numba, JAX, PyTorch, CuPy  
- Distance metrics: correlation, euclidean, Mahalanobis, crossnobis  
- High-level API: `RSAModel`, `Searchlight`, `Pipeline`
- End-to-end demos: YAML-driven exploratory pipelines  
- Deep-learning integration: extract CNN activations, neural predictivity  
- Visualization & reporting: RDM heatmaps, MDS plots, sensor topographies  

## Installation

# mnara

**Modeling Neural Activation & Representational Analytics**

mnara is a high-performance Python toolbox for computational brain modeling, RSA/RDM computation, and end-to-end MEG/EEG/fMRI pipelines.

> **mnara** means “lighthouse” in Arabic— which represents the aim of the package: shedding light on brain computations and representations.

---

## 🎯 Project Goal

Provide researchers with a fast, flexible, and user-friendly Python toolkit for computing Representational Dissimilarity Matrices (RDMs) and conducting RSA across modalities (MEG/EEG/fMRI/ANN) with CPU/GPU backends.

---

## 🚀 Installation

Currently, mnara is under active development and not yet published on PyPI. To install the latest version:
```bash
git clone https://github.com/yourusername/mnara.git
cd mnara
pip install -e .
```

**Dependencies include:** NumPy, SciPy, scikit‑learn, MNE, PyYAML, joblib, tqdm, Click.

---

## 🤝 Contributing

We welcome contributions! Before starting your work:

1. Open an issue to discuss your proposed contribution and ensure alignment.  
2. Fork the repository.  
3. Create a feature branch (`git checkout -b feature/my-feature`).  
4. Commit your changes (`git commit -m 'Add my feature'`).  
5. Push to your fork (`git push origin feature/my-feature`).  
6. Open a Pull Request.

Be sure to include tests, update documentation, and follow code style.

---

## 🧩 Implemented Modules

- **mnara.distances**: Pluggable distance metrics with NumPy, Numba, JAX, PyTorch, and CuPy backends.  
- **mnara.rdm**: Core RDM computations (`generate`, `rdms`, `time_rdms`).  
- **mnara.pipeline.rdm**: High-level orchestration for loading input, running RDMs, and saving outputs.  
- **mnara.cli.compute_rdms**: Command-line interface `mnara-compute-rdms` for batch/static (`rdms`) and time-resolved (`time`) RDMs.

---

## 📋 TODO List

- **✅** Finalize core RDM API and CLI wrappers.  
- **⬜** Write comprehensive tests for `mnara.rdm` methods.  
- **⬜** Fix any remaining bugs and edge-case errors.  
- **⬜** Benchmark and optimize RDM computations (CPU vs GPU backends).  
- **⬜** Extend to RSA scoring (`mnara.rsa`), noise ceiling, searchlight analyses.  
- **⬜** Add data I/O adapters for BIDS, fMRI, and model activations.  
- **⬜** Improve documentation, tutorials in `docs/` and `examples/`.
- **⬜** Benchmark with other toolboxes!

---

*Built with ❤️ for computational neuroimaging*