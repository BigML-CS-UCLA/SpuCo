from spuco.utils import Trainer
from torch import nn, optim
from torch.utils.data import Dataset
import torch 
from tqdm import tqdm

class LFF():
    def __init__(
            self,
            trainset: Dataset,
            model: nn.Module,
            batch_size: int,
            optimizer: optim.Optimizer,
            q: float,
            device: torch.device = torch.device("cpu"),
            verbose: bool = False
    ):
        self.trainset_labels = [y for X, y in trainset]
        self.cross_entropy_no_reduction = nn.CrossEntropyLoss(reduction="none")

        def biased_loss(outputs, labels):
            ce_loss_vector = self.cross_entropy_no_reduction(outputs, labels)
            weights = [torch.pow(outputs[i][label], q) for i, label in enumerate(labels)]
            return torch.mean(ce_loss_vector * weights)
        self.biased_trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=biased_loss,
            verbose=verbose,
            device=device
        )

        # Initialize with inital loss of model (w/o training)
        outputs = self.biased_trainer.get_trainset_outputs()
        self.bias_loss_vector = self.cross_entropy_no_reduction(outputs, self.trainset_labels.to(self.trainer.device))
        self.debias_loss_vector = None

    def train(self):
        for epoch in range(self.num_epochs):
            self.biased_trainer.train_epoch(epoch)
            outputs = self.biased_trainer.get_trainset_outputs()
            self.bias_loss_vector = self.cross_entropy_no_reduction(outputs, self.trainset_labels.to(self.trainer.device))

            with tqdm(self.trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
                pbar.set_description(f"Debiased Model: Epoch {epoch}")
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)

                    accuracy = Trainer.compute_accuracy(outputs, labels)
                    
                    # Compute Loss 
                    loss = self.debias_loss(outputs, labels)

                    # Backward Pass and Optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy}%")

    def debias_loss(self, outputs, labels):
        ce_loss_vector = self.cross_entropy_no_reduction(outputs, labels)
        weights = self.bias_loss_vector / (self.bias_loss_vector + self.debias_loss_vector)
        return torch.mean(ce_loss_vector * weights)