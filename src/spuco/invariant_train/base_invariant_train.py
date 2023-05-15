from abc import ABC, abstractmethod

class BaseInvariantTrain(ABC):
    def __init__(self):
        pass 
    
    @abstractmethod
    def train():
        pass 
    