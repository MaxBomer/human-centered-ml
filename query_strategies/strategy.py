from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from data import Data
    from parameters import TaskConfig
    
    TaskArgs = TaskConfig

class Strategy:
    def __init__(
        self,
        dataset: Data,
        net: Any,
        args_input: Any,
        args_task: TaskArgs
    ) -> None:
        self.dataset = dataset
        self.net = net
        self.args_input = args_input
        self.args_task = args_task

    def query(self, n: int) -> np.ndarray:
        """Query n samples from the unlabeled pool."""
        raise NotImplementedError("Subclasses must implement query method")
    
    def get_labeled_count(self) -> int:
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        return len(labeled_idxs)
    
    def get_model(self) -> torch.nn.Module:
        return self.net.get_model()

    def update(self, pos_idxs: np.ndarray, neg_idxs: np.ndarray | None = None) -> None:
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs is not None:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(
        self,
        data: DataLoader | None = None,
        model_name: str | None = None,
        wandb_log_callback: Any | None = None
    ) -> None:
        if model_name is None:
            if data is None:
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                self.net.train(labeled_data, wandb_log_callback=wandb_log_callback)
            else:
                self.net.train(data, wandb_log_callback=wandb_log_callback)
        else:
            if model_name == 'WAAL':
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                self.net.train(labeled_data, X_labeled, Y_labeled, X_unlabeled, Y_unlabeled)
            else:
                raise NotImplementedError(f"Model {model_name} not implemented")

    def predict(self, data: Union[DataLoader, Dataset]) -> torch.Tensor:
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data: Union[DataLoader, Dataset]) -> torch.Tensor:
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data: Union[DataLoader, Dataset], n_drop: int = 10) -> torch.Tensor:
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data: Union[DataLoader, Dataset], n_drop: int = 10) -> torch.Tensor:
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data: Union[DataLoader, Dataset]) -> torch.Tensor:
        embeddings = self.net.get_embeddings(data)
        return embeddings
    
    def get_grad_embeddings(self, data: Union[DataLoader, Dataset]) -> np.ndarray:
        embeddings = self.net.get_grad_embeddings(data)
        return embeddings

