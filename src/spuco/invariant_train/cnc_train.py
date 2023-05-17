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
        lambda_ce: float,
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
            loss = lambda_ce * self.criterion(outputs, labels)

            # Contrastive Loss
            for anchor_idx in range(len(batch)):
                # Only computed with majority examples as anchor
                if labels[anchor_idx] != groups[anchor_idx]:
                    continue 
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
                    pos_sum_sim = torch.sum(pos_sim) 
                    neg_sum_sim = torch.sum(torch.exp(torch.cosine_similarity(anchor, neg)))
                    sup_cl_loss = torch.sum(torch.log(pos_sim / (pos_sum_sim + neg_sum_sim)))
                    loss += sup_cl_loss / len(pos_idx)

            return loss, outputs, labels

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