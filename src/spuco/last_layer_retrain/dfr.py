import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from spuco.datasets import BaseSpuCoCompatibleDataset, GroupLabeledDatasetWrapper
from spuco.models import SpuCoModel
from spuco.utils.random_seed import seed_randomness
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Dict
import random


class DFR():
    def __init__(
        self,
        group_labeled_set: GroupLabeledDatasetWrapper,
        model: SpuCoModel,
        n_lin_models: int = 20, 
        labeled_valset_size: float = 0.5,
        C_range: List[float] = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01],
        class_weight_options: Optional[List[Dict]] = None,
        validation_set: Optional[GroupLabeledDatasetWrapper] = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
        data_for_scaler: Optional[Dataset] = None
    ):
        """
        Initializes the DFR object.

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
        :param class_weight_options: options for class weight.
        :type class_weight_options: list
        :param validation_set: data used for hyperparameter selection. If not provided, half of the group labeled data will be used.
        :type validation_set: GroupLabeledDatasetWrapper
        :param data_for_scaler: Data used for fitting the sklearn scaler. If not provided, group labeled data will be used.
        :type data_for_scaler: Dataset
        """
        
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        self.group_labeled_set = group_labeled_set
        self.model = model 
        self.C_range = C_range
        self.n_lin_models = n_lin_models
        self.labeled_valset_size = labeled_valset_size
        self.device = device 
        self.verbose = verbose
        self.data_for_scaler = data_for_scaler
        self.scaler = None
        self.preprocess = self.data_for_scaler is not None
        self.class_weight_options = class_weight_options
        self.validation_set = validation_set
    

    def train_single_model(self, C, X_train, y_train, g_train, class_weight):
        """
        Trains a single model.

        :param C: Regularization parameter C for the SVM model.
        :type C: float

        :param X_train: Training features.
        :type X_train: numpy.ndarray

        :param y_train: Training labels.
        :type y_train: numpy.ndarray

        :param g_train: Training group labels.
        :type g_train: numpy.ndarray

        :param class_weight: Weight associated with each class.
        :type class_weight: dict or 'balanced', optional
        """
        group_names = {g for g in g_train}
        group_partition = []
        for g in group_names:
            group_partition.append(np.where(g_train==g)[0])
        min_size = np.min([len(g) for g in group_partition])
        X_train_balanced = []
        y_train_balanced = []
        for g in group_partition:
            indices = np.random.choice(g, size=min_size, replace=False)
            X_train_balanced.append(X_train[indices])
            y_train_balanced.append(y_train[indices])
        X_train_balanced = np.concatenate(X_train_balanced)
        y_train_balanced = np.concatenate(y_train_balanced)

        logreg = LogisticRegression(penalty='l1', C=C, solver="liblinear", class_weight=class_weight)
        logreg.fit(X_train_balanced, y_train_balanced)
        
        return logreg.coef_, logreg.intercept_

    def train_multiple_model(self, C, X_labeled_train, y_labeled_train, g_labeled_train, class_weight):
        """
        Trains the DFR model.

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
            coef_, intercept_ = self.train_single_model(C, X_labeled_train, y_labeled_train, g_labeled_train, class_weight)
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

        for C in self.C_range:
            for class_weight in self.class_weight_options:
                coef, intercept = self.train_single_model(C, X_labeled_train, y_labeled_train, g_labeled_train, class_weight)
                # coef, intercept = self.train_multiple_model(C, X_labeled_train, y_labeled_train, g_labeled_train)
                wg_acc = self.evaluate_worstgroup_acc(C, coef, intercept, X_labeled_val, y_labeled_val, g_labeled_val)
                if wg_acc > best_wg_acc:
                    best_wg_acc = wg_acc
                    self.best_C = C
                    self.best_class_weight = class_weight
                if self.verbose:
                    print('C {} | class weight {}: Val Worst-Group Accuracy: {}'.format(C, class_weight, wg_acc))
                    print('Best C {} | class weight {}: Best Val Worst-Group Accuracy: {}'.format(self.best_C, self.best_class_weight, best_wg_acc))
    
    def train(self):
        """
        Retrain last layer
        """
        if self.verbose:
            print('Encoding data ...')
        X_labeled, y_labeled, g_labeled = self.encode_dataset(self.group_labeled_set)
        X_labeled = X_labeled.detach().cpu().numpy()
        y_labeled = y_labeled.detach().cpu().numpy()
        g_labeled = g_labeled.detach().cpu().numpy()

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
        coef, intercept = self.train_multiple_model(self.best_C, X_labeled, y_labeled, g_labeled, self.best_class_weight)
        self.linear_model = (self.best_C, coef, intercept, self.scaler)


    def evaluate_worstgroup_acc(self, C, coef, intercept, X_val, y_val, g_val):
        """
        Evaluates the worst-group accuracy for the DFR model.

        :param C: Regularization parameter C for the SVM model.
        :type C: float

        :param coef: Coefficients of the linear SVM model.
        :type coef: numpy.ndarray

        :param intercept: Intercept of the linear SVM model.
        :type intercept: numpy.ndarray

        :param X_val: Validation features.
        :type X_val: numpy.ndarray

        :param y_val: Validation labels.
        :type y_val: numpy.ndarray

        :param g_val: Validation group labels.
        :type g_val: numpy.ndarray
        """
        logreg = LogisticRegression(penalty='l1', C=C, solver="liblinear")
        n_classes = np.max(y_val) + 1
        # the fit is only needed to set up logreg
        logreg.fit(X_val[:n_classes], np.arange(n_classes))
        
        logreg.coef_ = coef
        logreg.intercept_ = intercept
        
        preds = logreg.predict(X_val)
        group_names = {g for g in g_val}
        accs = [(preds == y_val)[g_val == g].mean() for g in group_names]
        return np.min(accs)

    def encode_dataset(self, dataset):
        """
        Encodes the training set using the DFR model.

        :param dataset: The training dataset.
        :type dataset: torch.utils.data.Dataset

        :return: The encoded features and labels of the training set.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        labeled = type(dataset) == GroupLabeledDatasetWrapper

        X_train = []
        y_train = []
        if labeled:
            g_train = []

        trainloader = DataLoader(
            dataset=dataset, 
            batch_size=100,
            shuffle=False,
            num_workers=4, 
            pin_memory=True
        )

        self.model.eval()
        with torch.no_grad():
            for batch in trainloader:
                if labeled:
                    inputs, labels, groups = batch
                    inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)
                    g_train.append(groups)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                X_train.append(self.model.backbone(inputs))
                y_train.append(labels)
                    
            if labeled:
                return torch.cat(X_train), torch.cat(y_train), torch.cat(g_train)
            else:
                return torch.cat(X_train), torch.cat(y_train)