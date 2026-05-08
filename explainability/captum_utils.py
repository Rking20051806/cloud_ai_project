from __future__ import annotations

import torch


def normalize_attribution(attribution: torch.Tensor) -> torch.Tensor:
    attribution = attribution.detach()
    attribution = attribution - attribution.min()
    max_value = attribution.max().clamp(min=1e-7)
    return attribution / max_value
