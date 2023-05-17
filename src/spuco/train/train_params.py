from torch import nn, optim 

class TrainParams():
    def __init__(
        self, 
        model: nn.Module,
        optimizer: optim.Optimizer, 
        lr_scheduler: LR_S,
        batch_size: int,
        num_epochs; int,
    ):
        self.model = model 
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.num_epochs = num_epochs
