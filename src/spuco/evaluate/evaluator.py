import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from spuco.datasets import SpuriousTargetDatasetWrapper
from spuco.utils.random_seed import seed_randomness

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class Evaluator:
    def __init__(
        self,
        testset: Dataset, 
        group_partition: Dict[Tuple[int, int], List[int]],
        group_weights: Dict[Tuple[int, int], float],
        batch_size: int,
        model: nn.Module,
        sklearn_linear_model: Optional[Tuple[float, float, float, Optional[StandardScaler]]] = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False
    ):
        """
        Initializes an instance of the Evaluator class.

        :param testset: Dataset object containing the test set.
        :type testset: Dataset

        :param group_partition: Dictionary object mapping group keys to a list of indices corresponding to the test samples in that group.
        :type group_partition: Dict[Tuple[int, int], List[int]]

        :param group_weights: Dictionary object mapping group keys to their respective weights.
        :type group_weights: Dict[Tuple[int, int], float]

        :param batch_size: Batch size for DataLoader.
        :type batch_size: int

        :param model: PyTorch model to evaluate.
        :type model: nn.Module

        :param sklearn_linear_model: Tuple representing the coefficients and intercept of the linear model from sklearn. Default is None.
        :type sklearn_linear_model: Optional[Tuple[float, float, float, Optional[StandardScaler]]], optional

        :param device: Device to use for computations. Default is torch.device("cpu").
        :type device: torch.device, optional

        :param verbose: Whether to print evaluation results. Default is False.
        :type verbose: bool, optional
        """
          
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        self.testloaders = {}
        self.group_partition = group_partition
        self.group_weights = group_weights
        self.model = model
        self.device = device
        self.verbose = verbose
        self.accuracies = None
        self.sklearn_linear_model = sklearn_linear_model
        self.n_classes = np.max(testset.labels) + 1

        # Create DataLoaders 

        # Group-Wise DataLoader
        for key in group_partition.keys():
            self.testloaders[key] = DataLoader(Subset(testset, group_partition[key]), batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)
        
        # SpuriousTarget Dataloader
        core_labels = []
        spurious = []
        for key in self.group_partition.keys():
            for _ in self.group_partition[key]:
                core_labels.append(key[0])
                spurious.append(key[1])
        try:
            spurious_dataset = SpuriousTargetDatasetWrapper(dataset=testset, spurious_labels=spurious, num_classes=np.max(core_labels) + 1)
            self.spurious_dataloader = DataLoader(spurious_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        except:
            print("WARNING: spurious dataloader not correctly intiialized, evaluating spurious attribute prediction may fail.")

    def evaluate(self):
        """
        Evaluates the PyTorch model on the test dataset and computes the accuracy for each group.
        """
        self.model.eval()
        self.accuracies = {}
        for key in tqdm(sorted(self.group_partition.keys()), "Evaluating group-wise accuracy", ):
            if self.sklearn_linear_model:
                self.accuracies[key] = self._evaluate_accuracy_sklearn_logreg(self.testloaders[key])
            else:
                self.accuracies[key] = self._evaluate_accuracy(self.testloaders[key])
            if self.verbose:
                print(f"Group {key} Accuracy: {self.accuracies[key]}")
        return self.accuracies
    
    def _evaluate_accuracy(self, testloader: DataLoader):
        with torch.no_grad():
            correct = 0
            total = 0    
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            return 100 * correct / total
    
    def _evaluate_accuracy_sklearn_logreg(self, testloader: DataLoader):
        C, coef, intercept, scaler = self.sklearn_linear_model

        X_test, y_test = self._encode_testset(testloader)
        X_test = X_test.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        if scaler:
            X_test = scaler.transform(X_test)
        logreg = LogisticRegression(penalty='l1', C=C, solver="liblinear")
        # the fit is only needed to set up logreg
        X_dummy = np.random.rand(self.n_classes, X_test.shape[1])
        logreg.fit(X_dummy, np.arange(self.n_classes))
        logreg.coef_ = coef
        logreg.intercept_ = intercept
        preds_test = logreg.predict(X_test)
        return (preds_test == y_test).mean() * 100
    
    def _encode_testset(self, testloader):
        X_test = []
        y_test = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                X_test.append(self.model.backbone(inputs))
                y_test.append(labels)
            return torch.cat(X_test), torch.cat(y_test)
        
    def evaluate_spurious_attribute_prediction(self):
        """
        Evaluates accuracy if the task was predicting the spurious attribute.
        """
        return self._evaluate_accuracy(self.spurious_dataloader)

    @property
    def worst_group_accuracy(self):
        """
        Returns the group with the lowest accuracy and its corresponding accuracy.

        :returns: A tuple containing the key of the worst-performing group and its corresponding accuracy.
        :rtype: tuple
        """
        if self.accuracies is None:
            print("Run evaluate() first")
            return None
        else:
            min_key = min(self.accuracies, key=self.accuracies.get)
            min_value = min(self.accuracies.values())
            return (min_key, min_value)
    
    @property
    def average_accuracy(self):
        """
        Returns the weighted average accuracy across all groups.

        :returns: The weighted average accuracy across all groups.
        :rtype: float
        """
        if self.accuracies is None:
            print("Run evaluate() first")
            return None
        else:
            accuracy = 0
            for key in self.group_partition.keys():
                accuracy += self.group_weights[key] * self.accuracies[key]
            return accuracy