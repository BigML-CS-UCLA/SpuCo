from datetime import datetime
import argparse
import os
import sys
import wandb

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import Adam, SGD

from spuco.evaluate import Evaluator
from spuco.end2end import LFF
from spuco.models import model_factory
from spuco.utils import set_seed
from spuco.datasets import SpuCoSun, IndexDatasetWrapper


# parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data/spucosun/4.0/")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="/data/spucosun/results/lff.csv")
parser.add_argument("--stdout_file", type=str, default="spuco_sun_lff.out")
parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50", "cliprn50"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=40)
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument("--lr", type=float, default=1e-3, choices=[1e-3, 1e-4, 1e-5])
parser.add_argument("--weight_decay", type=float, default=1e-4, choices=[1e-4, 5e-4, 1e-2, 1e-1, 1.0])
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="spuco")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default="spuco_sun_lff")

args = parser.parse_args()

if args.wandb:
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=args)
    # remove the stdout_file argument
    del args.stdout_file
    del args.results_csv
else:
    # check if the stdout file already exists, and if want to overwrite it
    DT_STRING = "".join(str(datetime.now()).split())
    args.stdout_file = f"{DT_STRING}-{args.stdout_file}"
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

def get_model_and_optimizer(args, trainset, valset, device):
    # initialize the model and the trainer
    model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
    valid_evaluator = Evaluator(
        testset=valset,
        group_partition=valset.group_partition,
        group_weights=valset.group_weights,
        batch_size=args.batch_size,
        model=model,
        device=device,
        verbose=True
    )

    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    return model, valid_evaluator, optimizer

bias_model, bias_valid_evaluator, bias_optimizer = get_model_and_optimizer(args, trainset, valset, device)
debias_model, debias_valid_evaluator, debias_optimizer = get_model_and_optimizer(args, trainset, valset, device)

lff = LFF(
    bias_model=bias_model,
    debias_model=debias_model,
    bias_optimizer=bias_optimizer,
    debias_optimizer=debias_optimizer,
    num_epochs=args.num_epochs,
    trainset=trainset,
    batch_size=args.batch_size,
    device=device,
    verbose=True,
    use_wandb=args.wandb
)

lff.train()

results = pd.DataFrame(index=[0])

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=debias_model,
    device=device,
    verbose=True
)
evaluator.evaluate()

results[f"wg_acc"] = evaluator.worst_group_accuracy[1]
results[f"avg_acc"] = evaluator.average_accuracy
results["spurious_attribute_prediction"] = evaluator.evaluate_spurious_attribute_prediction()

print(results)

if args.wandb:
    # convert the results to a dictionary
    results = results.to_dict(orient="records")[0]
    wandb.log(results)
else:
    results["alg"] = "lff"
    results["timestamp"] = pd.Timestamp.now()
    results["seed"] = args.seed
    results["pretrained"] = args.pretrained
    results["lr"] = args.lr
    results["weight_decay"] = args.weight_decay
    results["momentum"] = args.momentum
    results["num_epochs"] = args.num_epochs
    results["batch_size"] = args.batch_size

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