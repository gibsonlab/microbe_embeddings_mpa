# Additional required tools:

16S ASV analysis:
- aligned_nearest_neighbor (https://github.com/yk23/aligned_nearest_neighbor) -- tool for quickly computing nearest-neighbor queries from a test taxa set onto a training taxa set. Necessary to implement baseline methods.

# Installation

Install the `gem` package and all dependencies:

```bash
pip install -e ".[notebooks,evo]"
```

- Base install (`-e .`): core gem package, ML/bioinformatics dependencies
- `notebooks`: adds `geopandas` and `pycountry` for visualization notebooks
- `evo`: adds `evo-model` and `flash-attn` (requires a CUDA-enabled build environment)

For Evo2 support (not on PyPI, install from source separately):

```bash
pip install git+https://github.com/ArcInstitute/evo2.git
```