from collections import Counter
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from spuco.group_inference import BaseGroupInference
from spuco.utils import GroupLabeledDataset, Trainer, get_class_labels


class SSA(BaseGroupInference):
    def __init__(
        self, 
        group_unlabeled_dataset: Dataset, 
        group_labeled_dataset: GroupLabeledDataset,
        model: nn.Module, 
        labeled_valset_size: float,
        num_iters: int,
        tau_g_min: float,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 64,
        num_splits: int = 3, 
    ):
        super().__init__()
        self.group_unlabeled_dataset = group_unlabeled_dataset
        self.group_labeled_dataset = group_labeled_dataset
        self.model = model 
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size 
        self.num_iters = num_iters
        self.num_splits = num_splits 
        self.tau_g_min = tau_g_min
        
        # Create unlabeled data splits
        self.unlabeled_data_splits = []
        indices = np.array(range(len(self.group_unlabeled_dataset)))
        indices = indices[torch.randperm(len(indices)).numpy()]
        self.unlabeled_data_splits = np.array.split(indices, self.num_splits)

        # Create labeled data splits
        indices = np.array(range(len(self.group_labeled_dataset)))
        indices = indices[torch.randperm(len(indices)).numpy()]
        self.labeled_val_indices = indices[:int(len(indices) * labeled_valset_size)]
        self.labeled_train_indices = indices[int(len(indices) * labeled_valset_size):]

        # Create class_partition
        self.class_labels = get_class_labels(group_unlabeled_dataset)

    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        # Train SSA models to get spurious target predictions
        spurious_labels = np.zeros(len(self.group_unlabeled_dataset))
        for split_num in range(self.num_splits):
            best_ssa_model = self.train_ssa(split_num)
            curr_split_labels = self.label_split(split_num, best_ssa_model)
            spurious_labels[self.unlabeled_data_splits[split_num]] = curr_split_labels
        spurious_labels = spurious_labels.cpu().tolist()

        # Convert class and spurious labels into group partition
        group_partition = {}
        for i in range(len(self.group_unlabeled_dataset)):
            group_label = (self.class_labels[i], spurious_labels[i])
            if group_label not in group_partition:
                group_partition[group_label] = []
            group_partition[group_label].append(i)
        return group_partition
    
    def train_ssa(self, split_num: int) -> nn.Module:
        """
        Learn model to predict spurious attribute for splt_num
        """
        trainer = SSATrainer(self, split_num)
        trainer.train()

        return trainer.best_model
    
    def label_split(self, split_num: int) -> np.array:
        """
        Label [split_num] split
        """
        split_dataloader = DataLoader(
            dataset=Subset(
                dataset=self.group_unlabeled_dataset,
                indices=self.ssa.unlabeled_data_splits[split_num]
            ),
            batch_size=self.batch_size, 
            shuffle=False
        )

        with torch.no_grad():
            preds = []
            for X, _ in split_dataloader:
                X = X.to(self.device)
                preds.append(torch.argmax(self.model(X), dim=-1))
            preds = torch.cat(preds, dim=0).cpu().numpy()
            return preds

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
        self.optim = optim.SGD(self.model.parameters(), lr=self.ssa.lr, weight_decay=self.ssa.weight_decay)
        self.split_num = split_num
        self.num_iters = self.ssa.num_iters
        self.verbose = ssa.verbose
        
        # Create member variable for best model, accuracy
        self.best_model = None
        self.best_acc = -1.

        self.unlabeled_trainloader = DataLoader(
            dataset=Subset(
                dataset=self.ssa.group_unlabeled_dataset,
                indices=[i for split_num in range(self.num_splits) if split_num != self.split_num for i in self.ssa.unlabeled_data_splits[split_num]]
            ),
            batch_size=self.ssa.batch_size, 
            shuffle=True
        )

        self.labeled_trainloader = DataLoader(
            dataset=Subset(
                dataset=self.ssa.group_labeled_dataset,
                indices=self.ssa.labeled_train_indices
            ),
            batch_size=self.ssa.batch_size, 
            shuffle=True
        )
        self.valloader = DataLoader(
            dataset=Subset(
                dataset=self.ssa.group_labeled_dataset,
                indices=self.ssa.labeled_val_indices
            ),
            batch_size=self.ssa.batch_size, 
            shuffle=False
        )

        self.cross_entropy = nn.CrossEntropyLoss()

    def train(self) -> None:
        """
        Trains model targetting spurious attribute for given split
        """
        self.model.train()
        with tqdm(range(self.num_iters), total=self.num_iters, disable=not self.verbose) as pbar:
            pbar.set_description(f"Labelling Split {self.split_num}")
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
                val_acc = self.validate()

                pbar.set_postfix(loss=loss.item(), val_acc=val_acc)
    
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

        # Get indices of g_min outputs that are >= tau_g_min (threshold: hyperparameter)
        g_min_indices = torch.nonzero(pseudo_labels == g_min_label)
        g_min_indices = g_min_indices[torch.nonzero(outputs[g_min_indices] >= self.tau_g_min)]
        
        # Get all unsupervised indices for loss 
        unsup_indices = g_min_indices.cpu().tolist()
        for label, count in group_counter.most_common()[:-1]:
            k = g_min_count + len(g_min_indices) - count
            group_indices = torch.nonzero(pseudo_labels == label)
            group_outputs = outputs[group_indices]
            group_indices = group_indices[torch.topk(group_outputs, k=k)[1]].cpu().tolist()
            unsup_indices.extend(group_indices)

        # Compute cross entropy with respect to pseudo-labels 
        unsupervised_loss = self.cross_entropy(outputs[unsup_indices], pseudo_labels[unsup_indices])
         
        return supervised_loss + unsupervised_loss 
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            # Compute accuracy on validation
            outputs = []
            labels = []
            for X, _, label in self.valloader:
                X, label = X.to(self.device), label.to(self.device)
                outputs.append(self.model(X))
                labels.append(label)
            outputs = torch.cat(outputs, dim=0)
            labels = torch.cat(labels, dim=0)

            acc = Trainer.compute_accuracy(outputs, labels)

            # Update best model based on validation 
            if acc >= self.best_acc:
                self.best_acc = acc 
                self.best_model = deepcopy(self.model)
            return acc