import argparse
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD
from wilds import get_dataset

from spuco.datasets import WILDSDatasetWrapper
from spuco.evaluate import Evaluator
from spuco.group_inference import SpareInference, SpareInferenceTrainer
from spuco.invariant_train import SpareTrain
from spuco.models import model_factory
from spuco.utils import set_seed, Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--results_csv", type=str, default="results/waterbirds_spare.csv")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_lr", type=float, default=1e-3)
parser.add_argument("--infer_weight_decay", type=float, default=1e-4)
parser.add_argument("--infer_momentum", type=float, default=0.9)
parser.add_argument("--infer_num_epochs", type=int, default=1)

parser.add_argument("--high_sampling_power", type=int, default=2)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True, root_dir=args.root_dir)

train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=transform
)

val_data = dataset.get_subset(
    "val",
    transform=transform
)

# Get the training set
test_data = dataset.get_subset(
    "test",
    transform=transform
)

trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True)
testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)

model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

# trainer = SpareInferenceTrainer(
#     trainset=trainset,
#     model=model,
#     batch_size=args.batch_size,
#     valset=valset,
#     random_seed=args.seed,
#     optimizer=SGD(model.parameters(), lr=args.infer_lr, weight_decay=args.infer_weight_decay, momentum=args.infer_momentum),
#     device=device,
#     verbose=True
# )

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
    Z=predictions,
    class_labels=trainset.labels,
    device=device,
    high_sampling_power=args.high_sampling_power,
    verbose=True
)

group_partition, sampling_powers = spare_infer.infer_groups()
print("Sampling powers: {}".format(sampling_powers))
for key in sorted(group_partition.keys()):
    for true_key in sorted(trainset.group_partition.keys()):
        print("Inferred group: {}, true group: {}, size: {}".format(key, true_key, len([x for x in trainset.group_partition[true_key] if x in group_partition[key]])))

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

valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=train_transform
)
trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)

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