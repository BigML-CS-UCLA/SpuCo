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
        val_evaluator: Evaluator = None, 
        verbose: bool = False
    ):
        self.val_evaluator = val_evaluator
        self._best_model = None 
        self._best_wg_acc = -1
        self.verbose = verbose
        self.trainer = None
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            if self.val_evaluator is not None:
                self.val_evaluator.evaluate()
                if self.val_evaluator.worst_group_accuracy[1] > self._best_wg_acc:
                    self._best_wg_acc = self.val_evaluator.worst_group_accuracy[1]
                    self._best_model = deepcopy(self.trainer.model)
                if self.verbose:
                    print('Epoch {}: Val Worst-Group Accuracy: {}'.format(epoch, self.val_evaluator.worst_group_accuracy[1]))
                    print('Best Val Worst-Group Accuracy: {}'.format(self._best_wg_acc))
                
    def train_epoch(self, epoch: int):
        self.trainer.train_epoch(epoch)
        
    @property
    def best_model(self):
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get best model if no val_evaluator set to \
                get worst group validation accuracy.")
        else:
            return self._best_model

    @property
    def best_wg_acc(self):
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get worst group validation accuracy \
                no val_evaluator passed.")
        else:
            return self._best_wg_acc