from abc import ABC, abstractmethod
from copy import deepcopy

from spuco.evaluate import Evaluator


class BaseInvariantTrain(ABC):
    """
    Abstract base class for InvariantTrain methods
    Provides support for worst group accuracy early stopping
    """
    def __init__(
        self, 
        valid_evaluator: Evaluator = None, 
        verbose: bool = False
    ):
        self.valid_evaluator = valid_evaluator
        self.best_model = None 
        self.best_wg_acc = -1
        self.verbose = verbose
        self.trainer = None
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            if self.valid_evaluator is not None:
                self.valid_evaluator.evaluate()
                if self.valid_evaluator.worst_group_accuracy[1] > self.best_wg_acc:
                    self.best_wg_acc = self.valid_evaluator.worst_group_accuracy[1]
                    self.best_model = deepcopy(self.trainer.model)
                if self.verbose:
                    print('Epoch {}: Val Worst-Group Accuracy: {}'.format(epoch, self.valid_evaluator.worst_group_accuracy[1]))
                    print('Best Val Worst-Group Accuracy: {}'.format(self.best_wg_acc))
                
    @abstractmethod
    def train_epoch(self, epoch: int):
        self.trainer.train_epoch(epoch)