from pathlib import Path
import pandas as pd

from gem.datasets import MetaphlanProfileParser


def select_profiles_in_metadata(metadata_df, profiles_df) -> pd.DataFrame:
    return profiles_df.loc[profiles_df.index.isin(metadata_df['Sample ID'])]


def main(
        profile_tsv: Path,
        metadata_tsv: Path,
):
    profiles = pd.read_csv(profile_tsv, sep="\t")
    profiles_indexed = profiles.set_index("clade_name").transpose()
    profiles_indexed.index.name = "SampleID"

    metadata = pd.read_csv(metadata_tsv, sep="\t")
    print("Number of samples (All): {}".format(metadata.shape[0]))
    metadata_subset = metadata.loc[
        (metadata['age_category'] == 'adult')
        & (metadata['disease'] == 'healthy')
    ]

    profile_df = select_profiles_in_metadata(metadata_subset, profiles_indexed)
    extractor = MetaphlanProfileParser(profile_df)

    sgb_ids_subset = set()
    for sample in extractor.samples():
        sgb_ids_subset.update(sample.taxa_ids)

    print("Found: {} SGBs in healthy-adult sample subset.")


if __name__ == '__main__':
    data_dir = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/dataset")
    main(
        profile_tsv=data_dir / "BlancoMiguezA_2023_profiles.tsv",
        metadata_tsv=data_dir / "BlancoMiguezA_2023_metadata.tsv",
    )
