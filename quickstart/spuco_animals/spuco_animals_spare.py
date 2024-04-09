import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch.optim import SGD

from spuco.datasets import SpuCoAnimals
from spuco.evaluate import Evaluator, GroupEvaluator
from spuco.group_inference import SpareInference
from spuco.robust_train import SpareTrain
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="results/spucoanimals_spare.csv")

parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50", "cliprn50"])
parser.add_argument("--only_train_projection", action="store_true", help="only train projection, applicable only for cliprn50")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_lr", type=float, default=1e-3)
parser.add_argument("--infer_weight_decay", type=float, default=1e-4)
parser.add_argument("--infer_momentum", type=float, default=0.9)
parser.add_argument("--infer_num_epochs", type=int, default=1)
parser.add_argument("--num_clusters", type=int, default=4)
parser.add_argument("--high_sampling_power", type=int, default=2)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

trainset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    transform=transform,
)
trainset.initialize()

valset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="val",
    transform=transform,
)
valset.initialize()

testset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="test",
    transform=transform,
)
testset.initialize()

print(trainset.group_partition.keys())

model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
if args.arch == "cliprn50" and args.only_train_projection:
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone._modules['attnpool'].parameters():
        param.requires_grad = True

trainer = Trainer(
    trainset=trainset,
    model=model,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.infer_lr, weight_decay=args.infer_weight_decay, momentum=args.infer_momentum),
    device=device,
    verbose=True
)

trainer.train(num_epochs=args.infer_num_epochs)

logits = trainer.get_trainset_outputs()
predictions = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
spare_infer = SpareInference(
    logits=predictions,
    class_labels=trainset.labels,
    num_clusters=args.num_clusters,
    device=device,
    high_sampling_power=args.high_sampling_power,
    verbose=True,
    num_clusters=2
)

group_partition = spare_infer.infer_groups()
sampling_powers = spare_infer.sampling_powers

evaluator = Evaluator(
    testset=trainset,
    group_partition=group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()

# Log evaluation of groups
group_eval = GroupEvaluator(group_partition, trainset.group_partition, 4)
print("group_infer_acc:", group_eval.evaluate_accuracy())
print("group_infer_precision:", group_eval.evaluate_precision())
print("group_infer_recall:", group_eval.evaluate_recall())

# reinstantiate the model
model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
if args.arch == "cliprn50" and args.only_train_projection:
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone._modules['attnpool'].parameters():
        param.requires_grad = True
        
valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    transform=train_transform,
)
trainset.initialize()

spare_train = SpareTrain(
    model=model,
    num_epochs=args.num_epochs,
    trainset=trainset,
    group_partition=group_partition,
    sampling_powers=sampling_powers,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    val_evaluator=valid_evaluator,
    verbose=True
)
spare_train.train()

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()
results = pd.DataFrame(index=[0])
results["timestamp"] = pd.Timestamp.now()
results["seed"] = args.seed
results["high_sampling_power"] = args.high_sampling_power
results["lr"] = args.lr
results["arch"] = args.arch
results["weight_decay"] = args.weight_decay
results["momentum"] = args.momentum
results["num_epochs"] = args.num_epochs
results["batch_size"] = args.batch_size
results["infer_num_epochs"] = args.infer_num_epochs
results["num_clusters"] = args.num_clusters
results["infer_lr"] = args.infer_lr
results["infer_weight_decay"] = args.infer_weight_decay

results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=spare_train.best_model,
    device=device,
    verbose=True
)
evaluator.evaluate()

results["early_stopping_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["early_stopping_average_accuracy"] = evaluator.average_accuracy

if os.path.exists(args.results_csv):
    results_df = pd.read_csv(args.results_csv)
else:
    results_df = pd.DataFrame()

results_df = pd.concat([results_df, results], ignore_index=True)
results_df.to_csv(args.results_csv, index=False)

print('Done!')
print('Results saved to', args.results_csv)