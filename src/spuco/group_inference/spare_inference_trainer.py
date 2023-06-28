from copy import deepcopy
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from spuco.utils.trainer import Trainer


class SpareInferenceTrainer(Trainer):
    """
    Sparse Inference Trainer
    """

    def __init__(
            self,
            trainset: Dataset,
            model: nn.Module,
            batch_size: int,
            optimizer: optim.Optimizer,
            random_seed: int = 0,
            valset: Optional[Dataset] = None,
            val_freq: int = 1,
            lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
            max_grad_norm: Optional[float] = None,
            criterion: nn.Module = nn.CrossEntropyLoss(),
            forward_pass: Optional[Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
            sampler: Sampler = None,
            device: torch.device = torch.device("cpu"),
            verbose: bool = False
    ) -> None:
        """
        Initializes the Sparse Inference Trainer
        
        :param trainset: The training dataset
        :type trainset: Dataset
        :param model: The model to train
        :type model: nn.Module
        :param batch_size: The batch size
        :type batch_size: int
        :param optimizer: The optimizer to use
        :type optimizer: optim.Optimizer
        :param random_seed: The random seed for reproducibility. Defaults to 0.
        :type random_seed: int, optional
        :param valset: The validation dataset for early stopping. Defaults to None.
        :type valset: Optional[Dataset], optional
        :param val_freq: The validation frequency (in iterations). Defaults to 1.
        :type val_freq: int, optional
        :param lr_scheduler: The learning rate scheduler. Defaults to None.
        :type lr_scheduler: Optional[optim.lr_scheduler._LRScheduler], optional
        :param max_grad_norm: The maximum gradient norm. Defaults to None.
        :type max_grad_norm: Optional[float], optional
        :param criterion: The loss function. Defaults to nn.CrossEntropyLoss().
        :type criterion: nn.Module, optional
        :param forward_pass: The forward pass function. Defaults to None.
        :type forward_pass: Optional[Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], optional
        :param sampler: The sampler. Defaults to None.
        :type sampler: Sampler, optional
        :param device: The device to use. Defaults to torch.device("cpu").
        :type device: torch.device, optional
        :param verbose: Whether to print progress. Defaults to False.
        :type verbose: bool, optional
        """
        
        super().__init__(trainset, model, batch_size, optimizer, lr_scheduler, max_grad_norm, criterion, forward_pass, sampler, device, verbose)
        
        self.random_seed = random_seed
        if valset is not None:
            self.valset = valset
            self.val_freq = val_freq
            self.num_spurious = len(np.unique(valset.spurious))
            self.max_mean_f1 = np.zeros(self.valset.num_classes)
            self.inference_models = {}

            if self.verbose:
                print(f"Number of spurious classes: {self.num_spurious}")
                print(f"Number of classes: {self.valset.num_classes}")
                print(f"Number of validation samples: {len(self.valset)}")
                print("Sparse Inference Trainer initialized")

    def train_epoch(self, epoch: int) -> None:
        """
        Trains the PyTorch model for 1 epoch

        :param epoch: epoch number that is being trained (only used by logging)
        :type epoch: int
        """

        average_accuracy = 0.
        for iteration, batch in enumerate(self.trainloader):
            self.model.train()
            loss, outputs, labels = self.forward_pass(self, batch)
            accuracy = Trainer.compute_accuracy(outputs, labels)

            # backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.lr_scheduler is not None and isinstance(self.optimizer, optim.AdamW):
                self.lr_scheduler.step()
            self.optimizer.step()
            average_accuracy += accuracy

            if self.valset is not None and (iteration+1) % self.val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.get_val_outputs()
                    val_preds = nn.functional.softmax(val_logits, dim=1).detach().cpu().numpy()

                    # cluster val preds with kmeans for each class
                    for class_label in range(self.valset.num_classes):
                        f1s = []
                        class_val_preds = val_preds[np.array(self.valset.labels) == class_label]
                        if class_val_preds.shape[0] > 0:
                            kmeans = KMeans(n_clusters=self.num_spurious, random_state=self.random_seed, n_init=10)
                            cluster_labels = kmeans.fit_predict(class_val_preds)

                            # compare cluster labels to spurious labels
                            class_spurious = np.array(self.valset.spurious)[np.array(self.valset.labels) == class_label]

                            # rank cluster labels by size
                            cluster_label_counts = np.zeros(self.num_spurious)
                            for cluster_label in range(self.num_spurious):
                                cluster_label_counts[cluster_label] = (cluster_labels == cluster_label).sum().item()
                            cluster_label_ranks = np.argsort(cluster_label_counts)

                            # rank spurious labels by size
                            spurious_label_counts = np.zeros(self.num_spurious)
                            for spurious_label in range(self.num_spurious):
                                spurious_label_counts[spurious_label] = (class_spurious == spurious_label).sum().item()
                            spurious_label_ranks = np.argsort(spurious_label_counts)

                            # compute f1 for each spurious label
                            for spurious_label_rank, spurious_label in enumerate(spurious_label_ranks):
                                spurious_all = (class_spurious == spurious_label)
                                if spurious_all.sum().item() == 0:
                                    continue
                                spurious_cluster = (cluster_labels == cluster_label_ranks[spurious_label_rank])
                                spurious_f1 = f1_score(spurious_all, spurious_cluster)
                                f1s.append(spurious_f1)

                            mean_f1 = np.mean(f1s)
                            if mean_f1 > self.max_mean_f1[class_label]:
                                if self.verbose:
                                    print(f"Class {class_label} has new max mean f1 of {mean_f1} at epoch {epoch} iteration {iteration}")
                                self.max_mean_f1[class_label] = mean_f1
                                self.inference_models[class_label] = deepcopy(self.model)

        return average_accuracy / len(self.trainloader)
        
    def get_trainset_outputs(self):
        """
        Gets output of model on trainset
        """
        outputs = torch.zeros((len(self.trainset), self.trainset.num_classes))

        if self.valset is None:
            self.inference_models = {0: self.model}

        for class_label in self.inference_models:
            self.model = self.inference_models[class_label]
            class_indices = np.where(np.array(self.trainset.labels) == class_label)[0]
            
            model_outputs = []
            with torch.no_grad():
                self.model.eval()
                eval_trainloader = DataLoader(
                    dataset=self.trainset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=4, 
                    pin_memory=True
                )
                with tqdm(eval_trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
                    
                    pbar.set_description("Getting Trainset Outputs")
                    for input, _ in pbar:
                        model_outputs.append(self.model(input.to(self.device)))
                    model_outputs = torch.cat(model_outputs, dim=0)
                    outputs[class_indices] = model_outputs.detach().cpu()[class_indices]
        
        return outputs
        
    def get_val_outputs(self):
        """
        Gets output of model on valset
        """
        with torch.no_grad():
            self.model.eval()
            eval_loader = DataLoader(
                dataset=self.valset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4, 
                pin_memory=True
            )
            
            outputs = []
            for input, _ in eval_loader:
                outputs.append(self.model(input.to(self.device)))
            return torch.cat(outputs, dim=0)