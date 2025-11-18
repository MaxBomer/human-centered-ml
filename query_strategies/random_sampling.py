from __future__ import annotations

import numpy as np
from .strategy import Strategy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data import Data
    from parameters import TaskConfig

class RandomSampling(Strategy):
    def __init__(
        self,
        dataset: Data,
        net: Any,
        args_input: Any,
        args_task: TaskConfig
    ) -> None:
        super(RandomSampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n: int) -> np.ndarray:
        unlabeled_mask = ~self.dataset.labeled_idxs
        unlabeled_indices = np.where(unlabeled_mask)[0]
        return np.random.choice(unlabeled_indices, n, replace=False)
