from spuco.invariant_train import BaseInvariantTrain
import copy

class BaseEarlyStopper(BaseInvariantTrain):
    """
    BaseEarlyStopper class for training a model with early stopping based on 
    worst group accuracy on validation data.
    """

    def __init__(self):
        pass

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            self.evaluator.evaluate()
            if self.evaluator.worst_group_accuracy[1] > self.best_wg_acc:
                self.best_wg_acc = self.evaluator.worst_group_accuracy[1]
                self.best_model = copy.deepcopy(self.trainer.model)
            if self.verbose_val:
                print('Epoch {}: Val worst-group accuracy: {}'.format(epoch, self.evaluator.worst_group_accuracy[1]))
                print('          Best val worst-group accuracy: {}'.format(self.best_wg_acc))
