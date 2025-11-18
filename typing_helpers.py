"""Type definitions and helpers for type safety."""
from __future__ import annotations

from typing import Protocol, TypeVar, Callable, Any, TYPE_CHECKING
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np

if TYPE_CHECKING:
    from parameters import TaskConfig

# Type variables
T = TypeVar('T')
HandlerType = TypeVar('HandlerType', bound=Dataset)

# Re-export TaskConfig for convenience
if TYPE_CHECKING:
    TaskArgs = TaskConfig
else:
    from typing import TypeAlias
    TaskArgs: TypeAlias = "TaskConfig"  # Alias for backward compatibility

# Protocol for dataset handlers
class DatasetHandler(Protocol):
    """Protocol for dataset handlers."""
    def __init__(self, X: torch.Tensor | np.ndarray, Y: torch.Tensor | np.ndarray, transform: transforms.Compose | None) -> None: ...
    def __getitem__(self, index: int) -> tuple[Any, Any, int]: ...
    def __len__(self) -> int: ...

# Protocol for network classes
class NetworkProtocol(Protocol):
    """Protocol for network classes."""
    def __call__(self, dim: tuple[int, ...], pretrained: bool, num_classes: int) -> torch.nn.Module: ...
    def get_embedding_dim(self) -> int: ...

# Type aliases
HandlerCallable = Callable[[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray, transforms.Compose | None], Dataset]
DatasetCallable = Callable[[HandlerCallable, TaskArgs], Any]

