from __future__ import annotations

import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data import Data
    from parameters import TaskConfig

class KCenterGreedy(Strategy):
    def __init__(
        self,
        dataset: Data,
        net: Any,
        args_input: Any,
        args_task: TaskConfig
    ) -> None:
        super(KCenterGreedy, self).__init__(dataset, net, args_input, args_task)

    def query(self, n: int) -> np.ndarray:
        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data)
        embeddings_np = embeddings.numpy()

        # Compute pairwise distance matrix using squared Euclidean distance
        dist_mat = np.matmul(embeddings_np, embeddings_np.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(np.maximum(dist_mat, 0))  # Ensure non-negative for sqrt

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            
        selected_indices: np.ndarray = np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]
        return selected_indices
