import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .dfr import DFR
from spuco.datasets import BaseSpuCoCompatibleDataset, GroupLabeledDatasetWrapper

from spuco.models import SpuCoModel
from spuco.utils.random_seed import seed_randomness

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Optional
import random

class DISPEL(DFR):
    def __init__(
        self,
        group_labeled_set:  GroupLabeledDatasetWrapper,
        model: SpuCoModel,
        labeled_valset_size: float = 0.5,
        C_range: List[float] = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01],
        s_range: List[float] = [1.0, 0.9, 0.8, 0.7],
        alpha_range : List[float] = [1.0],
        size_of_mixed: Optional[int] = None,
        n_lin_models: int = 20, 
        data_for_scaler: Optional[Dataset] = None,
        group_unlabeled_set: Optional[Dataset] = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ):
        """
        Initializes the DISPEL object.

        :param group_labeled_dataset: The group-labeled dataset.
        :type group_labeled_dataset: Dataset
        TODO
        """

        super().__init__(
        group_labeled_set=group_labeled_set,
        model=model,
        labeled_valset_size=labeled_valset_size,
        C_range=C_range,
        n_lin_models=n_lin_models, 
        device=device,
        verbose=verbose,
        data_for_scaler=data_for_scaler)

        self.alpha_range = alpha_range
        self.s_range = s_range
        self.group_unlabeled_set = group_unlabeled_set
        self.size_of_mixed = size_of_mixed

    def train_single_model(self, alpha, s, C, X_labeled, y_labeled, g_labeled):
        
        # sample the maximal group balanced set from the group labeled data
        group_partition = []
        for g in range(np.max(g_labeled)+1):
            group_partition.append(np.where(g_labeled==g)[0])
        min_size = np.min([len(g) for g in group_partition])
        X_balanced = []
        y_balanced = []
        for g in group_partition:
            indices = np.random.choice(g, size=min_size, replace=False)
            X_balanced.append(X_labeled[indices])
            y_balanced.append(y_labeled[indices])
        X_balanced = np.concatenate(X_balanced)
        y_balanced = np.concatenate(y_balanced)

        # construct the group unbalanced set, which is the union of all group labeled and unlabeled data
        if self.X_unlabeled is not None:
            X_unbalanced = np.concatenate((X_labeled, self.X_unlabeled))
            y_unbalanced = np.concatenate((y_labeled, self.y_unlabeled))
        else:
            X_unbalanced = X_labeled
            y_unbalanced = y_labeled
        
        X_class_balanced = []
        y_class_balanced = []
        n_classes = np.max(y_unbalanced)+1
        if self.size_of_mixed is None:
            c_min = np.min([len(np.where(y_unbalanced == c)[0]) for c in range(n_classes) ])
            for c in range(n_classes):
                ids = np.where(y_unbalanced == c)[0]
                ids = np.random.choice(ids, size=c_min, replace=False)
                X_class_balanced.append(X_unbalanced[ids])  
                y_class_balanced.append(y_unbalanced[ids])  
        else:  
            for c in range(n_classes):
                ids = np.where(y_unbalanced == c)[0]
                ids = np.random.choice(ids, size=self.size_of_mixed//n_classes, replace=True)
                X_class_balanced.append(X_unbalanced[ids])
                y_class_balanced.append(y_unbalanced[ids])  
        X_unbalanced = np.vstack(X_class_balanced)
        y_unbalanced = np.hstack(y_class_balanced)

        X_mixed = []
        y_mixed = []
        for i in range(X_unbalanced.shape[0]):
            if np.random.rand() > alpha:
                X_mixed.append(X_unbalanced[i])
                y_mixed.append(y_unbalanced[i])
            else:
                # sample another example in the same class from the balanced dataset
                y = y_unbalanced[i]
                idx = np.where(y_balanced == y)[0]
                j = np.random.choice(idx)
                X_mixed.append(X_balanced[j] * s + X_unbalanced[i] * (1 - s))
                y_mixed.append(y)
        X_mixed = np.array(X_mixed)
        y_mixed = np.array(y_mixed)

        logreg = LogisticRegression(penalty='l1', C=C, solver="liblinear")
        logreg.fit(X_mixed, y_mixed)
        return logreg.coef_, logreg.intercept_
    

    def train_multiple_model(self, alpha, s, C, X_labeled_train, y_labeled_train, g_labeled_train):
    
        coefs, intercepts = [], []
        for i in range(self.n_lin_models):
            coef_, intercept_ = self.train_single_model(alpha, s, C, X_labeled_train, y_labeled_train, g_labeled_train)
            coefs.append(coef_)
            intercepts.append(intercept_)
        return np.mean(coefs, axis=0), np.mean(intercepts, axis=0)
    

    def hyperparam_selection(self, X_labeled_train, y_labeled_train, g_labeled_train, X_labeled_val, y_labeled_val, g_labeled_val):

        best_wg_acc = -1
        if self.verbose:
            print('Searching for best hyperparameters ...')
        for alpha in self.alpha_range:
            for s in self.s_range:
                for C in self.C_range:
                    coef, intercept = self.train_single_model(alpha, s, C, X_labeled_train, y_labeled_train, g_labeled_train)
                    # coef, intercept = self.train_multiple_model(C, X_labeled_train, y_labeled_train, g_labeled_train)
                    wg_acc = self.evaluate_worstgroup_acc(C, coef, intercept, X_labeled_val, y_labeled_val, g_labeled_val)
                    if wg_acc > best_wg_acc:
                        best_wg_acc = wg_acc
                        self.best_alpha = alpha
                        self.best_s = s
                        self.best_C = C
                    if self.verbose:
                        print('alpha {} | s {} |C {}: Val Worst-Group Accuracy: {}'.format(alpha, s, C, wg_acc))
                        print('Best alpha {} | s {} |C {}: Best Val Worst-Group Accuracy: {}'.format(self.best_alpha, self.best_s, self.best_C, best_wg_acc))
        
    
    def train(self):
        X_labeled, y_labeled, g_labeled = self.encode_trainset(self.group_labeled_set)
        X_labeled = X_labeled.detach().cpu().numpy()
        y_labeled = y_labeled.detach().cpu().numpy()
        g_labeled = g_labeled.detach().cpu().numpy()

        # Load the unlabeled dataset if it is given
        if self.group_unlabeled_set:
            self.X_unlabeled, self.y_unlabeled = self.encode_trainset(self.group_unlabeled_set)
            self.X_unlabeled = self.X_unlabeled.detach().cpu().numpy()
            self.y_unlabeled = self.y_unlabeled.detach().cpu().numpy()
        else:
            self.X_unlabeled, self.y_unlabeled = None, None

        # Standardize features
        if self.preprocess:
            self.scaler = StandardScaler()
            if self.data_for_scaler:
                X_scaler, _ = self.encode_trainset(self.data_for_scaler)
                X_scaler = X_scaler.detach().cpu().numpy()
                self.scaler.fit(X_scaler)
            else:
                self.scaler = StandardScaler()
                self.scaler.fit(X_labeled)
            X_labeled = self.scaler.transform(X_labeled)
            if self.group_unlabeled_set:
                self.X_unlabeled = self.scaler.transform(self.X_unlabeled)
        
        # Splite labeled data into training and validation data
        n_labeled = X_labeled.shape[0]
        ids = {i for i in range(n_labeled)}
        ids_val = set(random.sample(ids, int(self.labeled_valset_size*n_labeled)))
        ids_train = ids - ids_val
        ids_val = np.array(list(ids_val))
        ids_train = np.array(list(ids_train))

        X_labeled_train, y_labeled_train, g_labeled_train = X_labeled[ids_train], y_labeled[ids_train], g_labeled[ids_train]
        X_labeled_val, y_labeled_val, g_labeled_val = X_labeled[ids_val], y_labeled[ids_val], g_labeled[ids_val]
        
        self.hyperparam_selection(X_labeled_train, y_labeled_train, g_labeled_train, X_labeled_val, y_labeled_val, g_labeled_val)
        coef, intercept = self.train_multiple_model(self.best_alpha, self.best_s, self.best_C, X_labeled_train, y_labeled_train, g_labeled_train)
        self.linear_model = (self.best_C, coef, intercept, self.scaler)