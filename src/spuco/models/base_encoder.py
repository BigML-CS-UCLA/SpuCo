from abc import ABC, abstractmethod
from torch import nn 

class BaseEncoder(ABC, nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        pass 

    @abstractmethod
    def encode(self, x):
        pass