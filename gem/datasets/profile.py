from abc import abstractmethod, ABC
from typing import *
import numpy as np


class AbundanceProfile:
    def __init__(self, sample_id: str, taxa_ids: List[str], abundances: np.ndarray):
        self.sample_id = sample_id
        self.taxa_ids = taxa_ids
        self.abundances = abundances
        assert len(taxa_ids) == len(abundances), "Taxa ids must match abundances shape."

        abund_sum = float(np.sum(abundances))
        assert np.isclose(abund_sum, 1.0), f"Abundances don't sum to 1.0; got {abund_sum}"

    @property
    def abundances_ensure_normalized(self) -> np.ndarray:
        return self.abundances / np.sum(self.abundances)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"[{self.sample_id} --> {len(self.taxa_ids)} Taxa]"


class AbundanceProfileCollection(ABC):
    """
    Generic class which extracts profiles one by one.
    """
    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def samples(self) -> Iterator[AbundanceProfile]:
        raise NotImplementedError()