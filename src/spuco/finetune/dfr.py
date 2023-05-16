from torch import nn, optim 
from spuco.models import SpuCoModel
from spuco.utils import Trainer 
from torch.utils.data import Dataset, DataLoader
import torch 
from tqdm import tqdm 

class DFR():
    def __init__(
        self,
        group_balanced_dataset: Dataset,
        model: SpuCoModel,
        num_epochs: int,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False
    ):
        self.trainset = group_balanced_dataset
        self.model = model 
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device 
        self.verbose = verbose

    def train(self):
        X_train, y_train = self.encode_trainset()
        in_features = X_train.shape[1]
        classifer = torch.nn.Linear(in_features, self.model.classifier.out_features).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            classifer.parameters(), 
            weight_decay=self.weight_decay, 
            lr=self.lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            eta_min=1e-2,
            T_max=self.num_epochs
        )
        
        pbar = tqdm(range(self.num_epochs), desc="DFR: ", disable=not self.verbose)
        for epoch in pbar:
            optimizer.zero_grad()
            pred = classifer(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            schedule.step()
            if epoch % (self.num_epochs // 10) == 0:
                acc = (torch.argmax(pred, dim=-1) == y_train).float().mean().item() * 100
                pbar.set_postfix(epoch=epoch, acc=acc)

        self.model.classifier = classifer

    
    def encode_trainset(self):
        X_train = []
        y_train = []

        trainloader = DataLoader(
            dataset=self.trainset, 
            batch_size=self.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                X_train.append(self.model.backbone(inputs))
                y_train.append(labels)
            
            return torch.cat(X_train), torch.cat(y_train)

