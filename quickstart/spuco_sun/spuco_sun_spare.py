import argparse
import os
import sys
import wandb

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from spuco.datasets import GroupLabeledDatasetWrapper, SpuCoSun
from spuco.evaluate import Evaluator
from spuco.group_inference import SpareInference
from spuco.robust_train import SpareTrain
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed


# parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data/spucosun/4.0/")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="/data/spucosun/results/spare.csv")
parser.add_argument("--stdout_file", type=str, default="spuco_sun_spare.out")
parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50", "cliprn50"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=40)
parser.add_argument("--erm_lr", type=float, default=1e-3)
parser.add_argument("--erm_weight_decay", type=float, default=1e-4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="spuco")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default="spuco_sun_spare")

parser.add_argument("--infer_num_epochs", type=int, default=1)
parser.add_argument("--num_clusters", type=int, default=5, choices=[2, 5])
parser.add_argument("--high_sampling_power", type=int, default=2)

args = parser.parse_args()

if args.wandb:
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=args)
    # remove the stdout_file argument
    del args.stdout_file
    del args.results_csv
else:
    # check if the stdout file already exists, and if want to overwrite it
    if os.path.exists(args.stdout_file):
        print(f"stdout file {args.stdout_file} already exists, overwrite? (y/n)")
        response = input()
        if response != "y":
            sys.exit()
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    # redirect stdout to a file
    sys.stdout = open(args.stdout_file, "w")

print(args)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

trainset = SpuCoSun(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    transform=transform,
)
trainset.initialize()

valset = SpuCoSun(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="val",
    transform=transform,
)
valset.initialize()

testset = SpuCoSun(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="test",
    transform=transform,
)
testset.initialize()

model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

trainer = Trainer(
    trainset=trainset,
    model=model,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.erm_lr, weight_decay=args.erm_weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
trainer.train(num_epochs=args.infer_num_epochs)

logits = trainer.get_trainset_outputs()
predictions = torch.nn.functional.softmax(logits, dim=1)
spare_infer = SpareInference(
    Z=predictions,
    class_labels=trainset.labels,
    num_clusters=args.num_clusters,
    device=device,
    high_sampling_power=args.high_sampling_power,
    verbose=True
)

group_partition = spare_infer.infer_groups()
sampling_powers = spare_infer.sampling_powers

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

robust_trainset = GroupLabeledDatasetWrapper(trainset, group_partition)

# initialize the model and the trainer
model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=trainset.group_weights,
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

trainset = SpuCoSun(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    transform=transform,
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

results = pd.DataFrame(index=[0])

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

results[f"wg_acc"] = evaluator.worst_group_accuracy[1]
results[f"avg_acc"] = evaluator.average_accuracy

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
results["spurious_attribute_prediction"] = evaluator.evaluate_spurious_attribute_prediction()

results[f"early_stopping_wg_acc"] = evaluator.worst_group_accuracy[1]
results[f"early_stopping_avg_acc"] = evaluator.average_accuracy

print(results)

if args.wandb:
    # convert the results to a dictionary
    results = results.to_dict(orient="records")[0]
    wandb.log(results)
else:
    results["alg"] = "spare"
    results["timestamp"] = pd.Timestamp.now()
    results["seed"] = args.seed
    results["pretrained"] = args.pretrained
    results["lr"] = args.lr
    results["weight_decay"] = args.weight_decay
    results["momentum"] = args.momentum
    results["num_epochs"] = args.num_epochs
    results["batch_size"] = args.batch_size

    results["infer_lr"] = args.erm_lr
    results["infer_weight_decay"] = args.erm_weight_decay
    results["infer_num_epochs"] = args.infer_num_epochs

    results["high_sampling_power"] = args.high_sampling_power
    results["num_clusters"] = args.num_clusters

    results

    if os.path.exists(args.results_csv):
        results_df = pd.read_csv(args.results_csv)
    else:
        results_df = pd.DataFrame()

    results_df = pd.concat([results_df, results], ignore_index=True)
    results_df.to_csv(args.results_csv, index=False)

    print('Results saved to', args.results_csv)

    # close the stdout file
    sys.stdout.close()

    # restore stdout
    sys.stdout = sys.__stdout__

print('Done!')