from abc import ABC, abstractmethod
from typing import Generic
from src.dataset.base.models import T, S, P


class BaseSpliter(ABC, Generic[T, S, P]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data_to_be_splitted: S) -> P:
        splitted_data = self.split(data_to_be_splitted)
        return self.post_process(splitted_data)

    @abstractmethod
    def split(self, timeline_data: S) -> T:
        pass

    @abstractmethod
    def post_process(self, splitted_data: T) -> P:
        pass
