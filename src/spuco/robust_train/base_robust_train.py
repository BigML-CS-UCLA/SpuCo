from abc import ABC, abstractmethod
from copy import deepcopy

from spuco.evaluate import Evaluator

try:
    import wandb
except ImportError:
    pass

class BaseRobustTrain(ABC):
    """
    Abstract base class for InvariantTrain methods
    Provides support for worst group accuracy early stopping
    """
    def __init__(
        self, 
        val_evaluator: Evaluator = None, 
        verbose: bool = False,
        use_wandb: bool = False
    ):
        """
        Initializes the model trainer.

        :param val_evaluator: Evaluator object for validation evaluation. Default is None.
        :type val_evaluator: Evaluator, optional
        :param verbose: Whether to print training progress. Default is False.
        :type verbose: bool
        """
        self.val_evaluator = val_evaluator
        self._best_model = None 
        self._best_wg_acc = -1
        self._avg_acc_at_best_wg_acc = -1
        self.verbose = verbose
        self.trainer = None
        self.use_wandb = use_wandb
        
    def train(self):
        """
        Train for specified number of epochs (and do early stopping if val_evaluator given)
        """
        for epoch in range(self.num_epochs):
            train_avg_acc, train_avg_loss = self.train_epoch(epoch)
            if self.val_evaluator is not None:
                self.val_evaluator.evaluate()
                if self.val_evaluator.worst_group_accuracy[1] > self._best_wg_acc:
                    self._best_wg_acc = self.val_evaluator.worst_group_accuracy[1]
                    self._avg_acc_at_best_wg_acc = self.val_evaluator.average_accuracy
                    self._best_model = deepcopy(self.trainer.model)
                    self._best_epoch = epoch
                if self.verbose:
                    print('Epoch {}: {} Val Worst-Group Accuracy: {}'.format(epoch, self.trainer.name, self.val_evaluator.worst_group_accuracy[1]))
                    print('Best Val Worst-Group Accuracy: {}'.format(self._best_wg_acc))

                if self.use_wandb:
                    results = {
                            f'{self.trainer.name}_train_avg_acc': train_avg_acc, f'{self.trainer.name}_train_avg_loss': train_avg_loss,
                            f'{self.trainer.name}_val_wg_acc': self.val_evaluator.worst_group_accuracy[1], f'{self.trainer.name}_best_val_wg_acc': self._best_wg_acc, 
                            f'{self.trainer.name}_val_avg_acc': self.val_evaluator.average_accuracy, f'{self.trainer.name}_best_val_avg_acc': self._avg_acc_at_best_wg_acc,
                            f'{self.trainer.name}_epoch': epoch
                        }
                    
                    for group, acc in self.val_evaluator.accuracies.items():
                        results[f'{self.trainer.name}_val_acc_{group}'] = acc

                    if hasattr(self.val_evaluator, 'spurious_attribute_prediction'):
                        results[f'{self.trainer.name}_spurious_attribute_prediction'] = self.val_evaluator.evaluate_spurious_attribute_prediction()
                
                    wandb.log(results)
                
    def train_epoch(self, epoch: int):
        """
        Trains the model for a single epoch.

        :param epoch: The current epoch number.
        :type epoch: int
        """
        return self.trainer.train_epoch(epoch)
        
    @property
    def best_model(self):
        """
        Property for accessing the best model.

        :return: The best model.
        :rtype: Any
        :raises NotImplementedError: If no val_evaluator is set to get worst group validation accuracy.
        """
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get best model if no val_evaluator set to \
                get worst group validation accuracy.")
        else:
            return self._best_model

    @property
    def best_wg_acc(self):
        """
        Property for accessing the best worst group validation accuracy.

        :return: The best worst group validation accuracy.
        :rtype: Any
        :raises NotImplementedError: If no val_evaluator is passed.
        """
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get worst group validation accuracy \
                no val_evaluator passed.")
        else:
            return self._best_wg_acc
        
    
    @property
    def avg_acc_at_best_wg_acc(self):
        """
        Property for accessing the average accuracy at best worst group validation accuracy.

        :return: The average accuracy at best worst group validation accuracy.
        :rtype: Any
        :raises NotImplementedError: If no val_evaluator is passed.
        """
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get average accuracy at best worst group validation accuracy \
                no val_evaluator passed.")
        else:
            return self._avg_acc_at_best_wg_acc
        
    @property
    def best_epoch(self):
        """
        Property for accessing the best epoch number.

        :return: The best epoch number.
        :rtype: Any
        :raises NotImplementedError: If no val_evaluator is passed.
        """
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get early stopping epoch \
                no val_evaluator passed.")
        else:
            return self._best_epoch