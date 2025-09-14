from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

from verl.protocol import DataProto

RawRewardFn = Callable[..., Any]


class AbstractRewardManager(ABC):
    @abstractmethod
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ):
        pass

    @abstractmethod
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        pass
