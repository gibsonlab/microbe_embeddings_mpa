import torch
import torch.nn as nn
from typing import Optional


class LinearInitializedModule(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self, init_rng: Optional[torch.Generator] = None, weight_decay_compatible: bool = True):
        gain = 0.5 if weight_decay_compatible else 1.0

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain, generator=init_rng)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
