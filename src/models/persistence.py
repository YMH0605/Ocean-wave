"""Persistence baseline: the forecast is whatever the field was at time t.

The source paper reports MAE 0.14 m for 1-hour SWH forecasts without comparing
against persistence anywhere. Significant wave height is strongly
autocorrelated at one hour, so a substantial part of that score is available
for free, and no learned model should be believed until it clears this bar at
the same lead time.

Note the paper's own setup makes this comparison less lopsided than it first
appears: in `paper_swh` channel mode the network never sees SWH as an input, so
it cannot simply copy the field forward. Persistence can. Reporting both is the
only way to tell what the model actually contributes.

Not a nn.Module: there is nothing to train.
"""

from __future__ import annotations

import torch


class Persistence:
    """Zero-parameter reference model."""

    name = "persistence"

    def __call__(self, batch) -> torch.Tensor:
        return batch["persist"]

    @staticmethod
    def predict(batch) -> torch.Tensor:
        return batch["persist"]


class Climatology:
    """Second reference: predict the training-set mean field.

    Weaker than persistence at short leads but it does not degrade with lead
    time, so it is the curve any useful model must stay under out to 48 hours.
    """

    name = "climatology"

    def __init__(self, mean_field: torch.Tensor):
        self.mean_field = mean_field  # (1, H, W) in physical units

    def __call__(self, batch) -> torch.Tensor:
        b = batch["persist"].shape[0]
        return self.mean_field.to(batch["persist"].device).expand(b, -1, -1, -1)
