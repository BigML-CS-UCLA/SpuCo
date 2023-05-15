from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from spuco.utils import SpuriousTargetDataset

class Evaluator:
    def __init__(
        self,
        testset: Dataset, 
        group_partition: Dict[Tuple[int, int], List[int]],
        group_weights: Dict[Tuple[int, int], float],
        batch_size: int,
        model: nn.Module,
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
        :param device: Device to use for computations. Default is torch.device("cpu").
        :type device: torch.device, optional
        :param verbose: Whether to print evaluation results. Default is False.
        :type verbose: bool, optional
        """
        self.testloaders = {}
        self.group_partition = group_partition
        self.group_weights = group_weights
        self.model = model
        self.device = device
        self.verbose = verbose
        self.accuracies = None

        # Create DataLoaders 

        # Group-Wise DataLoader
        for key in group_partition.keys():
            sampler = SubsetRandomSampler(group_partition[key])
            self.testloaders[key] = DataLoader(testset, batch_size=batch_size, sampler=sampler)
        
        # SpuriousTarget Dataloader
        spurious = torch.zeros(len(testset))
        for key in self.group_partition.keys():
            for i in self.group_partition[key]:
                spurious[i] = key[1]
        spurious_dataset = SpuriousTargetDataset(dataset=testset, spurious_labels=spurious)
        self.spurious_dataloader = DataLoader(spurious_dataset, batch_size=batch_size)

    def evaluate(self):
        """
        Evaluates the PyTorch model on the test dataset and computes the accuracy for each group.
        """
        self.model.eval()
        self.accuracies = {}
        for key in sorted(self.group_partition.keys()):
            self.accuracies[key] = self._evaluate_accuracy(self.testloaders[key])
            if self.verbose:
                print(f"Group {key} Test Accuracy: {self.accuracies[key]}")
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
        
    def evaluate_spurious_task(self):
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