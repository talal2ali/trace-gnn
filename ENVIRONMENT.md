# Environment

The exact environment in which every reported result was produced. Recorded by
`rerun_canonical.py` into each run's `run_manifest.json`, and reproduced here so the versions
travel with the code rather than only with the results.

| | |
|---|---|
| Python | **3.13.9** |
| platform | Linux-6.8.0-136-generic-x86_64-with-glibc2.35 |
| torch | **2.12.1+cu130** |
| CUDA | **13.0** |
| cuDNN | 92000 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| scipy | 1.16.3 |
| xgboost | 3.2.0 |
| matplotlib | 3.10.7 |
| GPUs | 2 × NVIDIA RTX 6000 Ada Generation, 48 GB each |

## Installing

`torch` must be installed from pytorch.org for your CUDA version, not from
`requirements.txt`:

    pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
    pip install -r requirements.txt

## What needs a GPU and what does not

| task | GPU |
|---|---|
| training any arm | **yes**, 48 GB for the three-layer variant |
| `rerun_canonical.py --dry-run` | no, but it loads the corpus and builds the graph |
| the figure scripts | no |
| `tests/`, `protocol_v2/tests/`, `edge_k/tests/` | no |

`optuna` is used only by the per-fold baseline search. `matplotlib` only by the figure scripts.
The model itself needs `torch`, `numpy` and `pandas`, and **no PyTorch Geometric**.

## Reproducibility caveat, stated once

Message passing accumulates neighbour contributions with scatter operations that are not
associative in floating point on GPU, so a fixed seed does not give bit-identical results. Four
repeats of the reported configuration at seeds 42 to 45 give a standard deviation of 0.010 in
mean average precision. Expect a re-run to land within roughly 0.02 of 0.855, not on it.
