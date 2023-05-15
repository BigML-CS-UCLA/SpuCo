import torch
from torch import optim

from spuco.invariant_train import BaseInvariantTrain
from spuco.factory.models import BaseEncoder
from spuco.utils import GroupLabeledDataset
from spuco.models import Trainer


class CorrectNContrastTrain(BaseInvariantTrain):
    def __init__(
        self,
        trainset: GroupLabeledDataset,
        model: BaseEncoder,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_pos: int, 
        num_neg: int,
        num_epochs: int,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False  
    ):
        
        super().__init__()
        self.num_epochs = num_epochs 

        def forward_pass(self, batch):
            inputs, labels, groups = batch 
            inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)
            cross_entropy_loss = self.criterion(self.model(inputs), labels)


            anchor_idx = torch.randint(0, len(inputs))
            pos_idx = []
            neg_idx = []
            
            i = 0
            for i in range(len(inputs)):
                if len(pos_idx) < num_pos:
                    if labels[i] == labels[anchor_idx] and groups[i] != groups[anchor_idx]:
                        pos_idx.append(i)

                if len(neg_idx) < num_neg:
                    if labels[i] != labels[anchor_idx] and groups[i] == groups[anchor_idx]:
                        neg_idx.append(i)
            anchor = self.model.encode(torch.unsqueeze(inputs[anchor_idx], dim=0))
            pos = self.model.encode(inputs[pos_idx])
            neg = self.model.encode(inputs[neg_idx])
            pos_sim = torch.exp(torch.cosine_similarity(anchor[:, :, None], pos[None, :, :]))[0]
            pos_sum_sim = torch.sum(pos_sim) 
            neg_sum_sim = torch.sum(torch.exp(torch.cosine_similarity(anchor[:, :, None], neg[None, :, :])))

            sup_cl_loss = torch.sum(torch.log(pos_sim / ((pos_sum_sim + neg_sum_sim) / (num_pos + num_neg))))

            return cross_entropy_loss + sup_cl_loss

        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            forward_pass=forward_pass,
            verbose=verbose,
            device=device
        )

    def train(self):
        for epoch in range(self.num_epochs):
            self.trainer.train(epoch)