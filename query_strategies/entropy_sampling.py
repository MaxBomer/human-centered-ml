from __future__ import annotations

import numpy as np
import torch
from .strategy import Strategy
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from data import Data
    from parameters import TaskConfig

class EntropySampling(Strategy):
    def __init__(
        self,
        dataset: Data,
        net: Any,
        args_input: Any,
        args_task: TaskConfig
    ) -> None:
        super(EntropySampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
        uncertainties = (probs * log_probs).sum(1)
        _, sorted_indices = uncertainties.sort(descending=True)
        selected_indices: np.ndarray = sorted_indices[:n].cpu().numpy()
        result = cast(np.ndarray, unlabeled_idxs[selected_indices])
        return result
