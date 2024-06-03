import numpy as np
import random
import torch
import wandb

from copy import deepcopy
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from spuco.datasets import IndexDatasetWrapper
from spuco.evaluate import Evaluator 
from spuco.utils import Trainer
from spuco.utils.random_seed import seed_randomness

class EMA:
    def __init__(self, label, alpha=0.7):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        
    def update(self, data, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data
        self.updated[index] = 1
        
    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()

class LFF():
    def __init__(
            self,
            trainset: Dataset,
            bias_model: nn.Module,
            debias_model: nn.Module,
            batch_size: int,
            bias_optimizer: optim.Optimizer,
            debias_optimizer: optim.Optimizer,
            num_epochs: int,
            q: float = 0.7,
            alpha: float = 0.7,
            val_evaluator: Evaluator = None,
            device: torch.device = torch.device("cpu"),
            verbose: bool = False,
            use_wandb: bool = False,
    ):
        """
        Initializes an instance of the LFF class.

        :param trainset: IndexDatasetWrapper object containing the training set.
        :type trainset: IndexDatasetWrapper
        :param bias_model: PyTorch model for the biased phase.
        :type bias_model: nn.Module
        :param debias_model: PyTorch model for the debiased phase.
        :type debias_model: nn.Module
        :param batch_size: Batch size for DataLoader.
        :type batch_size: int
        :param bias_optimizer: Optimizer for the biased phase.
        :type bias_optimizer: optim.Optimizer
        :param debias_optimizer: Optimizer for the debiased phase.
        :type debias_optimizer: optim.Optimizer
        :param q: Parameter controlling the trade-off between fairness and accuracy.
        :type q: float
        :param alpha: Exponential moving average parameter.
        :type alpha: float
        :param num_epochs: Number of training epochs.
        :type num_epochs: int
        :param device: Device to use for computations. Default is torch.device("cpu").
        :type device: torch.device, optional
        :param verbose: Whether to print training progress. Default is False.
        :type verbose: bool, optional
        """
         
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        assert q >= 0. and q <= 1., "q must be in [0,1]"

        self.bias_model = bias_model
        self.debias_model = debias_model
        self.cross_entropy_no_reduction = nn.CrossEntropyLoss(reduction="none")
        self.bias_optimizer = bias_optimizer
        self.debias_optimizer = debias_optimizer
        self.sample_loss_ema_b = EMA(torch.LongTensor(trainset.labels), alpha=alpha)
        self.sample_loss_ema_d = EMA(torch.LongTensor(trainset.labels), alpha=alpha)
        self.num_epochs = num_epochs
        self.q = q
        self.device = device
        self.verbose = verbose 
        self.use_wandb = use_wandb
        self.num_classes = trainset.num_classes
        self.trainloader = DataLoader(
            IndexDatasetWrapper(trainset), 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4, 
            pin_memory=True
        )
        self.val_evaluator = val_evaluator
        self._best_model = None 
        self._best_wg_acc = -1
        self._avg_acc_at_best_wg_acc = -1

    def train(self):
        """
        Train the bias model and debias model for the specified number of epochs.

        :return: None
        """
        for epoch in range(self.num_epochs):
            self.bias_model.train()
            self.debias_model.train()

            bias_avg_loss = 0.
            bias_avg_acc = 0.
            debias_avg_loss = 0.
            debias_avg_acc = 0.
            with tqdm(self.trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
                pbar.set_description(f"Epoch {epoch}")
                for inputs, labels, index in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    bias_outputs = self.bias_model(inputs)
                    debias_outputs = self.debias_model(inputs)

                    loss_b = self.cross_entropy_no_reduction(bias_outputs, labels).cpu().detach()
                    loss_d = self.cross_entropy_no_reduction(debias_outputs, labels).cpu().detach()

                    # EMA sample loss
                    self.sample_loss_ema_b.update(loss_b, index)
                    self.sample_loss_ema_d.update(loss_d, index)

                    # class-wise normalize
                    loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
                    loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

                    label_cpu = labels.cpu().detach().numpy()

                    for c in range(self.num_classes):
                        class_index = np.where(label_cpu == c)[0]
                        max_loss_b = self.sample_loss_ema_b.max_loss(c)
                        max_loss_d = self.sample_loss_ema_d.max_loss(c)
                        loss_b[class_index] /= max_loss_b
                        loss_d[class_index] /= max_loss_d

                    # Bias Model Training
                    bias_loss = self.bias_loss(bias_outputs, labels)
                    self.bias_optimizer.zero_grad()
                    bias_loss.backward()
                    self.bias_optimizer.step()
                    bias_accuracy = Trainer.compute_accuracy(bias_outputs, labels)

                    # Debias Model Training                    
                    debias_loss = self.debias_loss(debias_outputs, labels, loss_b, loss_d)
                    self.debias_optimizer.zero_grad()
                    debias_loss.backward()
                    self.debias_optimizer.step()
                    debias_accuracy = Trainer.compute_accuracy(debias_outputs, labels)

                    pbar.set_postfix(bias_loss=bias_loss.item(), bias_accuracy=f"{bias_accuracy}%", 
                                    debias_loss=debias_loss.item(), debias_accuracy=f"{debias_accuracy}%")
                    
                    if self.use_wandb:
                        wandb.log({"bias_loss": bias_loss.item(), "bias_accuracy": bias_accuracy, 
                                    "debias_loss": debias_loss.item(), "debias_accuracy": debias_accuracy})
                        
                    bias_avg_loss += bias_loss.item() * inputs.size(0)
                    bias_avg_acc += bias_accuracy * inputs.size(0)
                    debias_avg_loss += debias_loss.item() * inputs.size(0)
                    debias_avg_acc += debias_accuracy * inputs.size(0)

            bias_avg_loss /= len(self.trainloader.dataset)
            bias_avg_acc /= len(self.trainloader.dataset)
            debias_avg_loss /= len(self.trainloader.dataset)
            debias_avg_acc /= len(self.trainloader.dataset)
                        
            if self.val_evaluator is not None:
                self.val_evaluator.evaluate()
                if self.val_evaluator.worst_group_accuracy[1] > self._best_wg_acc:
                    self._best_wg_acc = self.val_evaluator.worst_group_accuracy[1]
                    self._avg_acc_at_best_wg_acc = self.val_evaluator.average_accuracy
                    self._best_model = deepcopy(self.trainer.model)
                    self._best_epoch = epoch
                if self.verbose:
                    print('Epoch {}: {} Val Worst-Group Accuracy: {}'.format(epoch, self.trainer.name, self.val_evaluator.worst_group_accuracy[1]))
                    print('Best Val Worst-Group Accuracy: {}'.format(self._best_wg_acc))

                if self.use_wandb:
                    results = {
                            'bias_avg_loss': bias_avg_loss, 'bias_avg_acc': bias_avg_acc,
                            'debias_avg_loss': debias_avg_loss, 'debias_avg_acc': debias_avg_acc,
                            'val_wg_acc': self.val_evaluator.worst_group_accuracy[1], 'best_val_wg_acc': self._best_wg_acc, 
                            'val_avg_acc': self.val_evaluator.average_accuracy, 'best_val_avg_acc': self._avg_acc_at_best_wg_acc,
                            'epoch': epoch
                        }
                    
                    for group, acc in self.val_evaluator.accuracies.items():
                        results[f'{self.trainer.name}_val_acc_{group}'] = acc

                    results[f'{self.trainer.name}_spurious_attribute_prediction'] = self.val_evaluator.evaluate_spurious_attribute_prediction()
                
                    wandb.log(results)
                        


    def bias_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Compute the bias loss for a batch of outputs and labels.

        :param outputs: Output predictions from the model.
        :type outputs: torch.Tensor
        :param labels: True labels for the inputs.
        :type labels: torch.Tensor
        :return: The bias loss.
        :rtype: torch.Tensor
        """
        ce_loss_vector = self.cross_entropy_no_reduction(outputs, labels)
        outputs = torch.softmax(outputs, dim=-1)
        weights = torch.tensor([np.float_power(outputs[i][label].item(), self.q) for i, label in enumerate(labels)]).to(self.device)
        return torch.mean(ce_loss_vector * weights)
    
    def debias_loss(
            self, 
            outputs: torch.Tensor, 
            labels: torch.Tensor, 
            bias_loss_vector: torch.Tensor, 
            debias_loss_vector: torch.Tensor
        ):
        """
        Compute the debias loss for a batch of outputs, labels, bias loss vector, and debias loss vector.

        :param outputs: Output predictions from the model.
        :type outputs: torch.Tensor
        :param labels: True labels for the inputs.
        :type labels: torch.Tensor
        :param bias_loss_vector: Bias loss vector.
        :type bias_loss_vector: torch.Tensor
        :param debias_loss_vector: Debias loss vector.
        :type debias_loss_vector: torch.Tensor
        :return: The debias loss.
        :rtype: torch.Tensor
        """
        ce_loss_vector = self.cross_entropy_no_reduction(outputs, labels)
        weights = bias_loss_vector / (bias_loss_vector + debias_loss_vector + 1e-8)
        return torch.mean(ce_loss_vector * weights.to(self.device))