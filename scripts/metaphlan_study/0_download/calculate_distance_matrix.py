from pathlib import Path
import numpy as np

import dendropy
from phylodm import PhyloDM

def main():
    """ Locations """
    data_dir = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/dataset")
    NEWICK_FILE = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/database") / 'mpa_vJan21_CHOCOPhlAnSGB_202103.nwk'
    SGB_SUBSET_ID_PATH = data_dir / "BlancoMiguezA_2023.SGB_subset.txt"
    PRUNED_TREE_PATH = data_dir / "BlancoMiguezA_2023.TREE.nwk"
    DISTMAT_PATH = data_dir / "BlancoMiguezA_2023.DIST_MATRIX.npz"

    """ load the taxa subset. """
    with open(SGB_SUBSET_ID_PATH, "rt") as f:
        target_leaf_names = [l.strip() for l in f]
    target_leaf_names = [x for x in target_leaf_names if len(x) > 0]

    def cleanup_id(x: str) -> str:
        if x.startswith("SGB"):
            x = x[3:]
        if x.endswith("_group"):
            x = x[:-len("_group")]
        return x

    target_leaf_names = [cleanup_id(sgb_id) for sgb_id in target_leaf_names]
    print("Target leaf names:", len(target_leaf_names))

    """ Prune the tree leaf nodes, and save it. """
    tree = dendropy.Tree.get(path=str(NEWICK_FILE), schema='newick')
    taxa_to_retain = [taxon for taxon in tree.taxon_namespace if taxon.label in target_leaf_names]
    print("Pruning tree.")
    tree.retain_taxa(taxa_to_retain)
    tree.write_to_path(PRUNED_TREE_PATH, "newick")

    """ Validate the newly pruned tree. """
    tree2 = dendropy.Tree.get(path=PRUNED_TREE_PATH, schema='newick')
    print("tree2 leaves:", len(tree2.taxon_namespace))
    assert len(tree2.taxon_namespace) == len(taxa_to_retain)

    """ Calculate the distance matrix. """
    pdm = PhyloDM.load_from_newick_path(PRUNED_TREE_PATH)
    dm = pdm.dm(norm=False)
    labels = pdm.taxa()
    np.savez(DISTMAT_PATH, mat=dm, labels=labels)


if __name__ == "__main__":
    main()

