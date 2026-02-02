import pandas as pd


def print_project_info(proj_id: str, sample_metadata: pd.DataFrame):
    sample_section = sample_metadata.loc[sample_metadata['project'] == proj_id]
    regions = pd.unique(sample_section['region'])
    isos = pd.unique(sample_section['iso'])
    n_samples = sample_section.shape[0]
    print(f"Project: {proj_id}")
    print(f"\tregions: {regions}")
    print(f"\tisos: {isos}")
    print(f"\tnum samples: {n_samples}")
    