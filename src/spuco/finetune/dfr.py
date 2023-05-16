from torch import nn, optim 
from spuco.models import SpuCoModel
from spuco.utils import Trainer 
from torch.utils.data import Dataset
import torch 

class DFR():
    def __init__(
        self,
        group_balanced_dataset: Dataset,
        model: SpuCoModel,
        num_epochs: int,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False
    ):
        in_features = model.classifier.in_features
        out_features = model.classifier.out_features
        model.classifier = nn.Linear(in_features, out_features).to(device)
        optimizer = optim.SGD(
            model.classifier.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True
        )

        def forward_pass(self, batch):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                self.model.backbone.eval()
                outputs = self.model.backbone(inputs)
            outputs = self.model.classifier(outputs)
            loss = self.criterion(outputs, labels)
            return loss, outputs, labels 
        
        self.trainer = Trainer(
            trainset=group_balanced_dataset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            forward_pass=forward_pass,
            device=device,
            verbose=verbose
        )

        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            self.trainer.train_epoch(epoch)