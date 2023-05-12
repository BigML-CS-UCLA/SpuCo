from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class BaseGroupInference(ABC):
    def __init__():
        pass 

    @abstractmethod
    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        pass 