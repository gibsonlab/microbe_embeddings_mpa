from typing import *
import numpy as np
import pandas as pd

from .profile import AbundanceProfile, AbundanceProfileParser


class MetaphlanProfile(AbundanceProfile):
    def __init__(self, sample_id: str, taxa_ids: List[str], abundances: np.ndarray):
        super().__init__(sample_id, taxa_ids, abundances)

        # Quirk: sometimes there are SGB groups instead of SGBs, e.g. "SGB1498_group".
        # Here, let's handle these simply as SGB1498 by removing the suffix.
        suffix = "_group"
        suffix_len = len(suffix)
        self.taxa_ids_raw = taxa_ids
        self.taxa_ids = [sgb[:-suffix_len] if sgb.endswith(suffix) else sgb for sgb in self.taxa_ids_raw]


class MetaphlanProfileParser(AbundanceProfileParser):
    """
    Implementation of AbundanceProfileExtractor for Metaphlan tables.
    """
    def __init__(self, profile_df: pd.DataFrame):
        sgb_cols = [col for col in profile_df.columns if col.split("|")[-1].startswith("t__SGB")]
        self.sgb_profile_df = profile_df[sgb_cols].rename(
            columns=lambda _col: _col.split("|")[-1][len("t__"):]
        )

    def __len__(self):
        return self.sgb_profile_df.shape[0]

    def samples(self) -> Iterator[MetaphlanProfile]:
        for sample_id, row in self.sgb_profile_df.iterrows():
            sgb_names = []
            sgb_abunds = []

            for sgb_id, sgb_abund in row.items():
                if sgb_abund > 0:
                    sgb_names.append(sgb_id)
                    sgb_abunds.append(sgb_abund)

            sgb_abunds = np.array(sgb_abunds, dtype=float)
            sgb_abunds = sgb_abunds / np.sum(sgb_abunds)
            yield MetaphlanProfile(sample_id, sgb_names, sgb_abunds)
