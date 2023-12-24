from copy import deepcopy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np 

class GroupEvaluator:
    def __init__(
        self,
        inferred_group_partition: Dict[Tuple[int, int], List[int]],
        true_group_partition: Dict[Tuple[int, int], List[int]],
        num_classes: int,
        verbose: bool = False
    ):      
    
        self.num_classes = num_classes
        self.verbose = verbose
        
        if self.verbose:
            print("Merging true group partition into majority / minority groups only")
        self.true_group_partition = {}
        for key in true_group_partition.keys():
            if key[0] == key[1]:
                self.true_group_partition[(key[0], "maj")] = deepcopy(true_group_partition[key])
            else:
                min_group = (key[0], "min")
                if min_group not in self.true_group_partition:
                    self.true_group_partition[min_group] = []
                self.true_group_partition[min_group].extend(deepcopy(true_group_partition[key]))
                        
        if self.verbose:
            print("Merging inferred group partition into majority / minority groups only")
        self.inferred_group_partition = {}
        for key in inferred_group_partition.keys():
            if key[0] == key[1]:
                self.inferred_group_partition[(key[0], "maj")] = deepcopy(inferred_group_partition[key])
            else:
                min_group = (key[0], "min")
                if min_group not in self.inferred_group_partition:
                    self.inferred_group_partition[min_group] = []
                self.inferred_group_partition[min_group].extend(deepcopy(inferred_group_partition[key]))

        if self.verbose:
            print("Inverting partitions")
        self.inferred_group_labels = GroupEvaluator.invert_group_partition(self.inferred_group_partition)
        self.true_group_labels = GroupEvaluator.invert_group_partition(self.true_group_partition)
    
    @staticmethod
    def invert_group_partition(group_partition: Dict):
        group_labels_dict = {}
        for key in group_partition.keys():
            for i in group_partition[key]:
                group_labels_dict[i] = key
                
        group_labels = []
        for i in range(len(group_labels_dict)):
            group_labels.append(group_labels_dict[i])
        
        return group_labels
    
    def evaluate_accuracy(self):
        correct = 0
        total = 0
        
        for inferred, true in zip(self.inferred_group_labels, self.true_group_labels):
            if inferred == true:
                correct += 1
            total += 1
        
        return correct / total
    
    def evaluate_precision(self):
        precisions = []
        
        for class_num in range(self.num_classes):
            true_pos = 0
            min_group = (class_num, "min")
            for i in self.inferred_group_partition[min_group]:
                if self.true_group_labels[i] == min_group:
                    true_pos += 1
            precisions.append(true_pos / len(self.inferred_group_partition[min_group]))
        
        return np.mean(precisions), np.min(precisions)
    
    def evaluate_recall(self):
        recall = []
        
        for class_num in range(self.num_classes):
            true_pos = 0
            min_group = (class_num, "min")
            for i in self.inferred_group_partition[min_group]:
                if self.true_group_labels[i] == min_group:
                    true_pos += 1
            recall.append(true_pos / len(self.true_group_partition[min_group]))
        
        return np.mean(recall), np.min(recall)    

   
                
        
        
        