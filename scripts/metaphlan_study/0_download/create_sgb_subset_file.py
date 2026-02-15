from pathlib import Path
import pandas as pd

from gem.datasets import MetaphlanProfileParser


def main(
        profile_tsv: Path,
        out_path: Path,
):
    profiles = pd.read_csv(profile_tsv, sep="\t")
    profiles_indexed = profiles.set_index("clade_name").transpose()
    profiles_indexed.index.name = "SampleID"
    extractor = MetaphlanProfileParser(profiles_indexed)

    sgb_ids_subset = set()
    n_samples = 0
    for sample in extractor.samples():
        sgb_ids_subset.update(sample.taxa_ids_raw)
        n_samples += 1

    print("# of samples in study: {}".format(
        n_samples
    ))
    print("# of SGBs found in study: {}".format(
        len(sgb_ids_subset)
    ))

    with open(out_path, "w") as f:
        for sgb_id in sorted(sgb_ids_subset):
            print(sgb_id, file=f)
    print("Wrote SGB IDs to {}".format(out_path))


if __name__ == '__main__':
    data_dir = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/dataset")
    main(
        profile_tsv=data_dir / "BlancoMiguezA_2023_profiles.tsv",
        out_path=data_dir / "BlancoMiguezA_2023.SGB_subset.txt",
    )
