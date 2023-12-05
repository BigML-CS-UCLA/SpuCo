import argparse
import os
import sys
import threading
import wandb

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD
from spuco.datasets.base_spuco_dataset import MASK_SPURIOUS

from spuco.evaluate import Evaluator
from spuco.robust_train import ERM, GroupBalanceBatchERM
from spuco.models import model_factory
from spuco.utils import set_seed
from spuco.datasets import SpuCoSun


# parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data/spuco_image_folder_demo/")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="results/SpuCoSun.csv")
parser.add_argument("--stdout_file", type=str, default="spuco_sun_erm_gb.out")
parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="spuco")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default="spuco_sun_erm_gb")
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

######
# ERM #
######
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

# initialize the model and the trainer
erm_model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
erm_valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=erm_model,
    device=device,
    verbose=True
)
erm = ERM(
    model=erm_model,
    val_evaluator=erm_valid_evaluator,
    num_epochs=args.num_epochs,
    trainset=trainset,
    batch_size=args.batch_size,
    optimizer=SGD(erm_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True,
    use_wandb=args.wandb
)


#################
# CORE ONLY ERM #
#################

core_only_trainset = SpuCoSun(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    mask_type=MASK_SPURIOUS,
    transform=transform,
)
core_only_trainset.initialize()

core_only_valset = SpuCoSun(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="val",
    mask_type=MASK_SPURIOUS,
    transform=transform,
)
core_only_valset.initialize()

core_only_testset = SpuCoSun(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="test",
    mask_type=MASK_SPURIOUS,
    transform=transform,
)
core_only_testset.initialize()

# initialize the model and the trainer
core_only_erm_model = model_factory(args.arch, core_only_trainset[0][0].shape, core_only_trainset.num_classes, pretrained=args.pretrained).to(device)
core_only_erm_valid_evaluator = Evaluator(
    testset=core_only_valset,
    group_partition=core_only_valset.group_partition,
    group_weights=core_only_valset.group_weights,
    batch_size=args.batch_size,
    model=core_only_erm_model,
    device=device,
    verbose=True
)
core_only_erm = ERM(
    model=core_only_erm_model,
    val_evaluator=core_only_erm_valid_evaluator,
    num_epochs=args.num_epochs,
    trainset=core_only_trainset,
    batch_size=args.batch_size,
    optimizer=SGD(core_only_erm_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True,
    use_wandb=args.wandb
)

###################
#  Group Balance  #
###################

gb_model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
gb_valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=gb_model,
    device=device,
    verbose=True
)

group_balance = GroupBalanceBatchERM(
    model=gb_model,
    group_partition=trainset.group_partition,
    val_evaluator=gb_valid_evaluator,
    num_epochs=args.num_epochs,
    trainset=trainset,
    batch_size=args.batch_size,
    optimizer=SGD(gb_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True,
    use_wandb=args.wandb
)

erm.train()

core_only_erm.train()

group_balance.train()

results = pd.DataFrame(index=[0])
for alg, model, model_name in zip([erm, core_only_erm, group_balance], [erm_model, core_only_erm_model, gb_model], ["erm", "core_only_erm", "gb"]):
    evaluator = Evaluator(
        testset=testset,
        group_partition=testset.group_partition,
        group_weights=trainset.group_weights,
        batch_size=64,
        model=model,
        device=device,
        verbose=True
    )
    evaluator.evaluate()

    results[f"wg_acc_{model_name}"] = evaluator.worst_group_accuracy[1]
    results[f"avg_acc_{model_name}"] = evaluator.average_accuracy

    evaluator = Evaluator(
        testset=testset,
        group_partition=testset.group_partition,
        group_weights=trainset.group_weights,
        batch_size=args.batch_size,
        model=alg.best_model,
        device=device,
        verbose=True
    )
    evaluator.evaluate()
    results["spurious_attribute_prediction"] = evaluator.evaluate_spurious_attribute_prediction()

    results[f"early_stopping_wg_acc_{model_name}"] = evaluator.worst_group_accuracy[1]
    results[f"early_stopping_avg_acc_{model_name}"] = evaluator.average_accuracy
    
    evaluator = Evaluator(
        testset=core_only_testset,
        group_partition=testset.group_partition,
        group_weights=trainset.group_weights,
        batch_size=args.batch_size,
        model=model,
        device=device,
        verbose=True
    )
    evaluator.evaluate()
    results[f"core_only_test_wg_acc_{model_name}"] = evaluator.worst_group_accuracy[1]
    results[f"core_only_test_avg_acc_{model_name}"] = evaluator.average_accuracy

print(results)

if args.wandb:
    # convert the results to a dictionary
    results = results.to_dict(orient="records")[0]
    wandb.log(results)
else:
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