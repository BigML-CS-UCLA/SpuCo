from torch import nn, optim 
from models import BaseEncoder
from spuco.utils import Trainer 
from torch.utils.data import Dataset

class DFR():
    def __init__(
        self,
        group_balanced_dataset: Dataset,
        model: BaseEncoder, 
        optimizer: optim.Optimizer,
        num_epochs: int,
        batch_size: int = 64,
    ):
        
        # Literally just train last layer on dataset provided and you're done 
        self.trainer = Trainer(
            
        )
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            self.trainer.train(epoch)