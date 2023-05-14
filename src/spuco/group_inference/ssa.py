from copy import deepcopy
from typing import Dict, List, Tuple
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
from spuco.group_inference import BaseGroupInference
from spuco.utils import GroupLabeledDataset, CustomIndicesSampler, Trainer
import torch
import numpy as np 

class SSA(BaseGroupInference):
    def __init__(
        self, 
        group_unlabeled_dataset: Dataset, 
        group_labeled_dataset: GroupLabeledDataset,
        model: nn.Module, 
        optimizer: optim.Optimizer,
        labeled_valset_size: float,
        num_iters: int,
        tau_g_min: float,
        batch_size: int = 64,
        num_splits: int = 3, 
    ):
        self.group_unlabeled_dataset = group_unlabeled_dataset
        self.group_labeled_dataset = group_labeled_dataset
        self.model = model 
        self.optimizer = optimizer 
        self.batch_size = batch_size 
        self.num_iters = num_iters
        self.num_splits = num_splits 
        self.tau_g_min = tau_g_min
        
        # Create Splits 
        self.unlabeled_data_splits = []
        indices = np.array(range(len(self.group_unlabeled_dataset)))
        indices = indices[torch.randperm(len(indices)).numpy()]
        self.unlabeled_data_splits = np.array.split(indices, self.num_splits)

        indices = np.array(range(len(self.group_labeled_dataset)))
        indices = indices[torch.randperm(len(indices)).numpy()]
        self.labeled_valid_indices = indices[:int(len(indices) * labeled_valset_size)]
        self.labeled_train_indices = indices[int(len(indices) * labeled_valset_size):]

    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        spurious_labels = torch.zeros(len(self.group_unlabeled_dataset))
        for k in 
        return 

    def ssa_kth_fold(self, k: int):
        trainer = SSATrainer(self, k)


class SSATrainer:
    def __init__(
            self,
            ssa: SSA,
            split_num: int,
    ):
        """
        Constructor for the SSATrainer class.
        """
        self.ssa = ssa
        self.tau_g_min = self.ssa.tau_g_min
        self.best_model = None
        self.model = deepcopy(self.ssa.model).to(self.device)
        self.optim = self.ssa.optimizer # FIXME: model zoo refactor
        self.split_num = split_num
        self.num_iters = self.ssa.num_iters
        self.verbose = ssa.verbose
        
        # Create member variable for best model, accuracy
        self.best_model = None
        self.best_acc = -1.

        self.unlabeled_trainloader = DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            shuffle=(self.sampler is None), 
            sampler=CustomIndicesSampler(
                indices=[self.ssa.unlabeled_data_splits[i] for i in range(self.num_splits) if i != self.split_num].flatten(),
                shuffle=True
            )
        )
        self.labeled_trainloader = DataLoader(
            self.trainset, 
            batch_size=self.ssa.batch_size, 
            shuffle=(self.sampler is None), 
            sampler=CustomIndicesSampler(
                indices=self.ssa.labeled_train_indices,
                shuffle=True
            )
        )
        self.valloader = DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            shuffle=(self.sampler is None), 
            sampler=CustomIndicesSampler(
                indices=self.ssa.labeled_valid_indices,
                shuffle=True
            )
        )

        self.cross_entropy = nn.CrossEntropyLoss()

    def train(self) -> None:
        """
        Trains model targetting spurious attribute for given split
        """
        self.model.train()
        with tqdm(range(self.num_iters), total=self.num_iters, disable=not self.verbose) as pbar:
            pbar.set_description(f"Split ")
            unlabeled_train_iter = iter(self.unlabeled_trainloader)
            labeled_train_iter = iter(self.labeled_trainloader)
            for _ in pbar:
                self.model.train()
                try:            
                    unlabeled_train_batch = next(unlabeled_train_iter)
                except StopIteration:
                    unlabeled_train_iter = iter(self.unlabeled_trainloader)
                    unlabeled_train_batch = next(labeled_train_iter)

                try:
                    labeled_train_batch = next(labeled_train_iter)
                except StopIteration:
                    labeled_train_iter = iter(self.labeled_trainloader)
                    labeled_train_batch = next(labeled_train_iter)

                # Compute loss
                loss = self.train_step(
                    unlabeled_train_batch=unlabeled_train_batch,
                    labeled_train_batch=labeled_train_batch
                )   
                loss.backward()
                self.optimizer.step()

                # Validation
                self.validate()

                pbar.set_postfix(loss=loss.item())
    
    def train_step(self, unlabeled_train_batch, labeled_train_batch):
        """
        Trains a single step of SSA for given batch of unlabeled and labeled data
        """
        ###########################
        # Compute supervised loss #
        ###########################
        X, _, label = labeled_train_batch
        group_counter = Counter(label.long().tolist())
        g_min_label, g_min_count = group_counter.most_common()[-1]
        X, label = X.to(self.device), label.to(self.device)
        supervised_loss = self.cross_entropy(self.model(X), label)

        #############################
        # Compute unsupervised loss #
        #############################
        
        # Load data
        X, _ = unlabeled_train_batch
        X = X.to(self.device)

        # Get pseudo-labels
        outputs = self.model(X)
        pseudo_labels = torch.argmax(outputs, dim=-1)

        # Get indices of g_min outputs that are >-= tau_g_min (threshold: hyperparameter)
        g_min_indices = torch.nonzero(pseudo_labels == g_min_label)
        g_min_indices = g_min_indices[torch.nonzero(outputs[g_min_indices] >= self.tau_g_min)]
        
        # Get all unsupervised indices for loss 
        unsup_indices = g_min_indices.cpu().tolist()
        for label, count in group_counter.most_common()[:-1]:
            k = g_min_count + len(g_min_indices) - count
            group_indices = torch.nonzero(pseudo_labels == label)
            group_outputs = outputs[group_indices]
            group_indices = group_indices[torch.topk(group_outputs, k=k)[1]].cpu().to.list()
            unsup_indices.extend(group_indices)

        # Compute cross entropy with respect to pseudo-labels 
        unsupervised_loss = self.cross_entropy(outputs[unsup_indices], pseudo_labels[unsup_indices])
         
        return supervised_loss + unsupervised_loss 
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            # Compute accuracy on validation
            preds = []
            labels = []
            for X, _, label in self.valloader:
                preds.append(self.model(X.to(self.device)))
                labels.append(label)
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            acc = Trainer.compute_accuracy(preds, labels)

            # Update best model based on validation 
            if acc >= self.best_acc:
                self.best_acc = acc 
                self.best_model = deepcopy(self.model)
            return acc