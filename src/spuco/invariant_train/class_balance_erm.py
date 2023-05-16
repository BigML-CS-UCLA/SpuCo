import torch 
from torch import nn, optim
from spuco.utils import Trainer, CustomIndicesSampler, convert_labels_to_partition, get_class_labels
import random  # TODO: Do we want to control the randomness here?
from torch.utils.data import Dataset

class ClassBalanceERM():
    """
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: Dataset,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        device: torch.device = torch.device("cpu"),
        verbose=False
    ):
        """
        Initializes ClassBalanceERM
        """

        self.class_partition = convert_labels_to_partition(get_class_labels(trainset))
        assert batch_size >= len(self.class_partition), "batch_size must be >= number of groups (Group DRO requires at least 1 example from each group)"
        
        self.num_epochs = num_epochs

        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            sampler=CustomIndicesSampler(indices=[]),
            verbose=verbose,
            device=device
        )

        max_class_len = max([len(self.class_partition[key]) for key in self.class_partition.keys()])
        self.base_indices = []
        self.sampling_weights = []
        for key in self.class_partition.keys():
            self.base_indices.extend(self.class_partition[key])
            self.sampling_weights.extend([max_class_len / len(self.class_partition[key])] * len(self.class_partition[key]))
        
    def train(self):
        """
        Trains the model using the given hyperparameters.
        """
        for epoch in range(self.num_epochs):
            self.trainer.sampler.indices = random.choices(
                population=self.base_indices,
                weights=self.sampling_weights, 
                k=len(self.trainer.trainset)
            )
            self.trainer.train_epoch(epoch)
