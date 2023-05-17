from abc import ABC, abstractmethod

class BaseInvariantTrain(ABC):
    """
    Abstract base class for InvariantTrain methods
    """
    def __init__(self):
        pass 
    
    @abstractmethod
    def train():
        """
        Must be implemented by all InvariantTrain methods
        """
        pass 
    