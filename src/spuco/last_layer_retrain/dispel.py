import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .dfr import DFR
from spuco.datasets import BaseSpuCoCompatibleDataset, GroupLabeledDatasetWrapper
from spuco.models import SpuCoModel
from spuco.utils.random_seed import seed_randomness
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Dict
import random

class DISPEL(DFR):
    def __init__(
        self,
        group_labeled_set:  GroupLabeledDatasetWrapper,
        model: SpuCoModel,
        labeled_valset_size: float = 0.5,
        C_range: List[float] = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01],
        s_range: List[float] = [1.0, 0.9, 0.8, 0.7],
        groups_with_spu: Optional[List[int]] = None,
        class_weight_options: Optional[List[Dict]] = None,
        validation_set: Optional[GroupLabeledDatasetWrapper] = None,
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

        :param group_labeled_set: The group-labeled dataset.
        :type group_labeled_set: Dataset
        :param model: The base model.
        :type model: SpucoModel
        :param n_lin_models: Number of linear models to average.
        :type trainset: int
        :param labeled_valset_size: The ratio of the labeled data to be used for validation if no validation set is given. 
        :type labeled_valset_size: float
        :param C_range: Options of C, which is the inverse of l1 regularization strength as in sklearn.
        :type C_range: list
        :param s_range: Options of s, which is the weights assigned to the group-labeled data when generating mixed data.
        :type s_range: list
        :param alpha_range: Options of alpha, which is the probabilty of mixing data 
        :type alpha_range: list
        :param size_of_mixed: size of the mixed dataset. 
        :type size of_mixed: int
        :param group_unlabeled_set: group unlabeled dataset. If provided, it will be included in the group unbalanced dataset.
        :type group_unlabeled_set: Dataset
        :param class_weight_options: options for class weight.
        :type class_weight_options: list
        :param validation_set: data used for hyperparameter selection. If not provided, half of the group labeled data will be used.
        :type validation_set: GroupLabeledDatasetWrapper
        :param data_for_scaler: Data used for fitting the sklearn scaler. If not provided, group labeled data will be used.
        :type data_for_scaler: Dataset
        :param groups_with_spu: In case of missing groups, pass indices of existing groups that share the spurious feature with the missing groups here
        :type groups_with_spu: list
        """

        super().__init__(
        group_labeled_set=group_labeled_set,
        model=model,
        labeled_valset_size=labeled_valset_size,
        C_range=C_range,
        class_weight_options=class_weight_options,
        validation_set=validation_set,
        n_lin_models=n_lin_models, 
        device=device,
        verbose=verbose,
        data_for_scaler=data_for_scaler)

        self.alpha_range = alpha_range
        self.s_range = s_range
        self.group_unlabeled_set = group_unlabeled_set
        self.size_of_mixed = size_of_mixed
        self.groups_with_spu = groups_with_spu

    def train_single_model(self, alpha, s, C, X_labeled, y_labeled, g_labeled, class_weight):
        """
        Trains a single model.

        :param alpha: Trade-off parameter between accuracy and fairness.
        :type alpha: float

        :param s: Sensitivity parameter for fairness regularization.
        :type s: float

        :param C: Regularization parameter C for the SVM model.
        :type C: float

        :param X_labeled: Labeled training features.
        :type X_labeled: numpy.ndarray

        :param y_labeled: Labeled training labels.
        :type y_labeled: numpy.ndarray

        :param g_labeled: Labeled training group labels.
        :type g_labeled: numpy.ndarray

        :param class_weight: Weight associated with each class.
        :type class_weight: dict or 'balanced', optional
        """
        # sample the maximal group balanced set from the group labeled data

        if self.groups_with_spu:
            group_names = {g for g in g_labeled if g in self.groups_with_spu}
        else:
            group_names = {g for g in g_labeled}
        group_partition = []
        for g in group_names:
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
                y = y_unbalanced[i]
                idx = np.where(y_balanced == y)[0]
                if idx.shape[0] == 0:
                    # randomly sample an example from X_balanced
                    j = np.random.choice(np.arange(X_balanced.shape[0]))
                else:
                    # sample another example in the same class from the balanced dataset
                    j = np.random.choice(idx)
                X_mixed.append(X_balanced[j] * s + X_unbalanced[i] * (1 - s))
                y_mixed.append(y)
        X_mixed = np.array(X_mixed)
        y_mixed = np.array(y_mixed)

        logreg = LogisticRegression(penalty='l1', C=C, solver="liblinear", class_weight=class_weight)
        logreg.fit(X_mixed, y_mixed)
        
        return logreg.coef_, logreg.intercept_
    

    def train_multiple_model(self, alpha, s, C, X_labeled_train, y_labeled_train, g_labeled_train, class_weight):
        """
        Trains the DFR model.

        :param alpha: Trade-off parameter between accuracy and fairness.
        :type alpha: float

        :param s: Sensitivity parameter for fairness regularization.
        :type s: float

        :param C: Regularization parameter C for the SVM model.
        :type C: float

        :param X_labeled_train: Labeled training features.
        :type X_labeled_train: numpy.ndarray

        :param y_labeled_train: Labeled training labels.
        :type y_labeled_train: numpy.ndarray

        :param g_labeled_train: Labeled training group labels.
        :type g_labeled_train: numpy.ndarray

        :param class_weight: Weight associated with each class.
        :type class_weight: dict or 'balanced', optional
        """
        coefs, intercepts = [], []
        for i in range(self.n_lin_models):
            coef_, intercept_ = self.train_single_model(alpha, s, C, X_labeled_train, y_labeled_train, g_labeled_train, class_weight)
            coefs.append(coef_)
            intercepts.append(intercept_)
        return np.mean(coefs, axis=0), np.mean(intercepts, axis=0)
    

    def hyperparam_selection(self, X_labeled_train, y_labeled_train, g_labeled_train, X_labeled_val, y_labeled_val, g_labeled_val):
        """
        Performs hyperparameter selection for the DFR model.

        :param X_labeled_train: Labeled training features.
        :type X_labeled_train: numpy.ndarray

        :param y_labeled_train: Labeled training labels.
        :type y_labeled_train: numpy.ndarray

        :param g_labeled_train: Labeled training group labels.
        :type g_labeled_train: numpy.ndarray

        :param X_labeled_val: Labeled validation features.
        :type X_labeled_val: numpy.ndarray

        :param y_labeled_val: Labeled validation labels.
        :type y_labeled_val: numpy.ndarray

        :param g_labeled_val: Labeled validation group labels.
        :type g_labeled_val: numpy.ndarray
        """
        best_wg_acc = -1
        if self.verbose:
            print('Searching for best hyperparameters ...')
        for alpha in self.alpha_range:
            for s in self.s_range:
                for C in self.C_range:
                    for class_weight in self.class_weight_options:
                        coef, intercept = self.train_single_model(alpha, s, C, X_labeled_train, y_labeled_train, g_labeled_train, class_weight)
                        # coef, intercept = self.train_multiple_model(C, X_labeled_train, y_labeled_train, g_labeled_train)
                        wg_acc = self.evaluate_worstgroup_acc(C, coef, intercept, X_labeled_val, y_labeled_val, g_labeled_val)
                        if wg_acc > best_wg_acc:
                            best_wg_acc = wg_acc
                            self.best_alpha = alpha
                            self.best_s = s
                            self.best_C = C
                            self.best_class_weight = class_weight
                        if self.verbose:
                            print('alpha {} | s {} |C {} | class weight {} : Val Worst-Group Accuracy: {}'.format(alpha, s, C, class_weight, wg_acc))
                            print('Best alpha {} | s {} |C {} | class weight {}: Best Val Worst-Group Accuracy: {}'.format(self.best_alpha, self.best_s, self.best_C, self.best_class_weight, best_wg_acc))
            
    
    def train(self):
        """
        Last Layer Retraining.
        """
        
        if self.verbose:
            print('Encoding data ...')

        X_labeled, y_labeled, g_labeled = self.encode_dataset(self.group_labeled_set)
        X_labeled = X_labeled.detach().cpu().numpy()
        y_labeled = y_labeled.detach().cpu().numpy()
        g_labeled = g_labeled.detach().cpu().numpy()

        # Load the unlabeled dataset if it is given
        if self.group_unlabeled_set:
            self.X_unlabeled, self.y_unlabeled = self.encode_dataset(self.group_unlabeled_set)
            self.X_unlabeled = self.X_unlabeled.detach().cpu().numpy()
            self.y_unlabeled = self.y_unlabeled.detach().cpu().numpy()
        else:
            self.X_unlabeled, self.y_unlabeled = None, None

        # Standardize features
        if self.preprocess:
            self.scaler = StandardScaler()
            if self.data_for_scaler:
                X_scaler, _ = self.encode_dataset(self.data_for_scaler)
                X_scaler = X_scaler.detach().cpu().numpy()
                self.scaler.fit(X_scaler)
            else:
                self.scaler = StandardScaler()
                self.scaler.fit(X_labeled)
            X_labeled = self.scaler.transform(X_labeled)
            if self.group_unlabeled_set:
                self.X_unlabeled = self.scaler.transform(self.X_unlabeled)

        # If validation set is not provided, split labeled data into training and validation data
        # Otherwise, use the given validation set 
        if self.validation_set is None:
            n_labeled = X_labeled.shape[0]
            ids = {i for i in range(n_labeled)}
            ids_val = set(random.sample(ids, int(self.labeled_valset_size*n_labeled)))
            ids_train = ids - ids_val
            ids_val = np.array(list(ids_val))
            ids_train = np.array(list(ids_train))

            X_labeled_train, y_labeled_train, g_labeled_train = X_labeled[ids_train], y_labeled[ids_train], g_labeled[ids_train]
            X_labeled_val, y_labeled_val, g_labeled_val = X_labeled[ids_val], y_labeled[ids_val], g_labeled[ids_val]
        else:
            X_labeled_train, y_labeled_train, g_labeled_train = X_labeled, y_labeled, g_labeled
            X_labeled_val, y_labeled_val, g_labeled_val = self.encode_dataset(self.validation_set)
            X_labeled_val = X_labeled_val.detach().cpu().numpy()
            y_labeled_val = y_labeled_val.detach().cpu().numpy()
            g_labeled_val = g_labeled_val.detach().cpu().numpy()
            if self.preprocess:
                X_labeled_val = self.scaler.transform(X_labeled_val)    

        if self.class_weight_options is None:
            n_class = np.max(y_labeled_val) + 1
            self.class_weight_options = [{c: 1 for c in range(n_class)}]

        self.hyperparam_selection(X_labeled_train, y_labeled_train, g_labeled_train, X_labeled_val, y_labeled_val, g_labeled_val)
        if self.validation_set is not None:
            X_labeled = np.concatenate((X_labeled, X_labeled_val))
            y_labeled = np.concatenate((y_labeled, y_labeled_val))
            g_labeled = np.concatenate((g_labeled, g_labeled_val))

        coef, intercept = self.train_multiple_model(self.best_alpha, self.best_s, self.best_C, X_labeled, y_labeled, g_labeled, self.best_class_weight)
        self.linear_model = (self.best_C, coef, intercept, self.scaler)