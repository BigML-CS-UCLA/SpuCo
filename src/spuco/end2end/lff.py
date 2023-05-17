from spuco.utils import Trainer
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch 
from tqdm import tqdm


class LFF():
    def __init__(
            self,
            trainset: Dataset,
            bias_model: nn.Module,
            debias_model: nn.Module,
            batch_size: int,
            bias_optimizer: optim.Optimizer,
            debias_optimizer: optim.Optimizer,
            q: float,
            num_epochs: int,
            device: torch.device = torch.device("cpu"),
            verbose: bool = False
    ):
        assert q >= 0. and q <= 1., "q must be in [0,1]"

        self.bias_model = bias_model
        self.debias_model = debias_model
        self.cross_entropy_no_reduction = nn.CrossEntropyLoss(reduction="none")
        self.bias_optimizer = bias_optimizer
        self.debias_optimizer = debias_optimizer
        self.num_epochs = num_epochs
        self.q = q
        self.device = device
        self.verbose = verbose 
        self.softmax = nn.Softmax()
        self.trainloader = DataLoader(
            trainset, 
            batch_size=batch_size, 
            shuffle=True
        )

    def train(self):
        for epoch in range(self.num_epochs):
            with tqdm(self.trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
                pbar.set_description(f"Epoch {epoch}")
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                   
                    # Bias Model Training
                    bias_outputs = self.bias_model(inputs)
                    bias_loss = self.bias_loss(bias_outputs, labels)
                    self.bias_optimizer.zero_grad()
                    bias_loss.backward()
                    self.bias_optimizer.step()
                    bias_accuracy = Trainer.compute_accuracy(bias_outputs, labels)

                    # Compute W(x)
                    with torch.no_grad():
                        # Compute Bias and Debias Loss Vector  
                        bias_loss_vector = self.cross_entropy_no_reduction(self.bias_model(inputs), labels)
                        debias_loss_vector = self.cross_entropy_no_reduction(self.debias_model(inputs), labels)

                    # Debias Model Training
                    debias_outputs = self.debias_model(inputs)
                    debias_loss = self.debias_loss(debias_outputs, labels, bias_loss_vector, debias_loss_vector)
                    self.debias_optimizer.zero_grad()
                    debias_loss.backward()
                    debias_accuracy = Trainer.compute_accuracy(debias_outputs, labels)
                    pbar.set_postfix(bias_loss=bias_loss.item(), bias_accuracy=f"{bias_accuracy}%", 
                                    debias_loss=debias_loss.item(), debias_accuracy=f"{debias_accuracy}%")

    def bias_loss(self, outputs, labels):
        ce_loss_vector = self.cross_entropy_no_reduction(outputs, labels)
        outputs = self.softmax(outputs)
        weights = torch.tensor([torch.pow(outputs[i][label].item(), self.q) for i, label in enumerate(labels)]).to(self.device)
        return torch.mean(ce_loss_vector * weights)
    
    def debias_loss(self, outputs, labels, bias_loss_vector, debias_loss_vector):
        ce_loss_vector = self.cross_entropy_no_reduction(outputs, labels)
        weights = bias_loss_vector / (bias_loss_vector + debias_loss_vector)
        return torch.mean(ce_loss_vector * weights)