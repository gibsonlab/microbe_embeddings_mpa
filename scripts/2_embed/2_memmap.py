"""
Convert the HDF5-stored tensors in step 1, and convert it into memory-mapped tensordicts.

This step was separated from the previous one, for safety.
The alternative is to bypass the HDF5 storage altogether, opting for only the memmapped tensors.
However, note that these files can accidentally be changed; the extra layer introduces some room for user error.
"""

"""
From https://docs.pytorch.org/tensordict/main/saving.html

1) Saving a memmapped tensordict:
    x = TensorDict()
    x_disk = x.memmap("/path/to/saved/dir", num_threads=30)

2) Loading a memmapped tensordict:
    x = TensorDict.load_memmap("/path/to/saved/dir")
"""
from pathlib import Path
import torch
from tensordict import TensorDict


def populate_embeddings(tdict: TensorDict):
    raise NotImplementedError("todo")


def main(
        memmap_dir: Path,
):
    if memmap_dir.exists():
        print(f"Target memmap dir {memmap_dir} doesn't yet exist. It will be created.")
        memmap_dir.mkdir(parents=True, exist_ok=True)

    x = TensorDict()
    x.memmap(str(memmap_dir))
    populate_embeddings(x)


if __name__ == "__main__":
    main(
        memmap_dir=Path("/path/to/saved/dir"),
    )
