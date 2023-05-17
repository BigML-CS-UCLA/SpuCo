import torch
from torch import optim

from spuco.invariant_train import BaseInvariantTrain
from spuco.utils import GroupLabeledDataset, Trainer
from spuco.models import SpuCoModel


class CorrectNContrastTrain(BaseInvariantTrain):
    def __init__(
        self,
        trainset: GroupLabeledDataset,
        model: SpuCoModel,
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
            # Unpack inputs and move to correct device
            inputs, labels, groups = batch 
            inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)

            # Regular Cross Entropy Loss
            outputs = self.model(inputs)
            cross_entropy_loss = self.criterion(outputs, labels)

            # Randomly choose anchor
            anchor_idx = torch.randint(0, len(inputs), (1,))[0].item()

            pos_idx = []
            neg_idx = []
            for i in range(len(inputs)):
                if len(pos_idx) < num_pos: # Positives = same label, but different group
                    if labels[i] == labels[anchor_idx] and groups[i] != groups[anchor_idx]:
                        pos_idx.append(i)

                if len(neg_idx) < num_neg: # Negatives = different label, but same spurious attribute
                    if labels[i] != labels[anchor_idx] and groups[i] == groups[anchor_idx]:
                        neg_idx.append(i)
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                anchor = self.model.backbone(torch.unsqueeze(inputs[anchor_idx], dim=0))
                pos = self.model.backbone(inputs[pos_idx])
                neg = self.model.backbone(inputs[neg_idx])
                pos_sim = torch.exp(torch.cosine_similarity(anchor, pos))
                print("pos sim shape", pos_sim.shape, len(pos_idx))
                pos_sum_sim = torch.sum(pos_sim) 
                neg_sum_sim = torch.sum(torch.exp(torch.cosine_similarity(anchor, neg)))
                sup_cl_loss = torch.sum(torch.log(pos_sim / ((pos_sum_sim + neg_sum_sim) / (num_pos + num_neg))))
              
                return cross_entropy_loss + sup_cl_loss, outputs, labels
            else:
                return cross_entropy_loss, outputs, labels

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
        self.trainer.train(self.num_epochs)