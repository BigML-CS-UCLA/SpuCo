import argparse
import os

import pandas as pd
import torch
from torch.optim import SGD

from spuco.datasets import SpuCoMNIST, SpuriousFeatureDifficulty
from spuco.datasets.group_labeled_dataset_wrapper import GroupLabeledDatasetWrapper
from spuco.evaluate import Evaluator
from spuco.group_inference import GeorgeInference
from spuco.models import model_factory
from spuco.robust_train.group_dro import GroupDRO
from spuco.utils import Trainer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--results_csv", type=str, default="spuco_mnist_spare.csv")

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_lr", type=float, default=1e-3)
parser.add_argument("--infer_weight_decay", type=float, default=1e-2)
parser.add_argument("--infer_momentum", type=float, default=0.9)
parser.add_argument("--infer_num_epochs", type=int, default=1)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
difficulty = SpuriousFeatureDifficulty.MAGNITUDE_LARGE

trainset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    spurious_correlation_strength=0.995,
    classes=classes,
    split="train"
)
trainset.initialize()
valset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="val",
    #spurious_correlation_strength=0.995
)
valset.initialize()

testset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="test"
)
testset.initialize()

model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

trainer = Trainer(
    trainset=trainset,
    model=model,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.infer_lr, weight_decay=args.infer_weight_decay, momentum=args.infer_momentum),
    device=device,
    verbose=True
)

trainer.train(num_epochs=args.infer_num_epochs)
george_infer = GeorgeInference(
    Z=trainer.get_trainset_outputs(features=True),
    class_labels=trainset.labels,
    device=device,
    max_clusters=5,
    verbose=True
)
group_partition = george_infer.infer_groups()

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

model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
robust_trainset = GroupLabeledDatasetWrapper(trainset, group_partition)
group_dro = GroupDRO(
    model=model,
    num_epochs=5,
    trainset=robust_trainset,
    batch_size=64,
    optimizer=SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True),
    device=device,
    verbose=True
)
valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
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
results["lr"] = args.lr
results["weight_decay"] = args.weight_decay
results["momentum"] = args.momentum
results["num_epochs"] = args.num_epochs
results["batch_size"] = args.batch_size

results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=group_dro.best_model,
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