import argparse
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD
from wilds import get_dataset

from spuco.datasets import GroupLabeledDatasetWrapper, WILDSDatasetWrapper
from spuco.evaluate import Evaluator
from spuco.group_inference import JTTInference
from spuco.invariant_train import CustomSampleERM
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--results_csv", type=str, default="results/waterbirds_jtt.csv")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1.0)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_num_epochs", type=int, default=60)

parser.add_argument("--upsample_factor", type=int, default=100)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True, root_dir=args.root_dir)

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

trainer = Trainer(
    trainset=trainset,
    model=model,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
trainer.train(num_epochs=args.infer_num_epochs)

predictions = torch.argmax(trainer.get_trainset_outputs(), dim=-1).detach().cpu().tolist()
jtt = JTTInference(
    predictions=predictions,
    class_labels=trainset.labels
)

group_partition = jtt.infer_groups()
for key in sorted(group_partition.keys()):
    print(key, len(group_partition[key]))
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

invariant_trainset = GroupLabeledDatasetWrapper(trainset, group_partition)

val_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)

indices = []
indices.extend(group_partition[(0,0)])
indices.extend(group_partition[(0,1)] * args.upsample_factor)

print("Training on", len(indices), "samples")

jtt_train = CustomSampleERM(
    model=model,
    num_epochs=args.num_epochs,
    trainset=trainset,
    batch_size=args.batch_size,
    indices=indices,
    val_evaluator=val_evaluator,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
jtt_train.train()

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
results["pretrained"] = args.pretrained
results["lr"] = args.lr
results["weight_decay"] = args.weight_decay
results["momentum"] = args.momentum
results["num_epochs"] = args.num_epochs
results["batch_size"] = args.batch_size
results["upsample_factor"] = args.upsample_factor

results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=jtt.best_model,
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


