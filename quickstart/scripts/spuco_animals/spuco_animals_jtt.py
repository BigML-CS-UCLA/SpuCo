import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, precision_score
from torch.optim import SGD
from wilds import get_dataset

from spuco.datasets import GroupLabeledDatasetWrapper, SpuCoAnimals
from spuco.evaluate import Evaluator
from spuco.group_inference import JTTInference
from spuco.invariant_train import CustomSampleERM
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="results/spucoanimals_jtt.csv")

parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_num_epochs", type=int, default=7)

parser.add_argument("--upsample_factor", type=int, default=100)

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


max_precision = 0
logits_files = glob(f"logits/spucoanimals/lr=0.001_wd=0.0001_seed={args.seed}/valset*.pt")
for logits_file in logits_files:
    epoch = int(logits_file.split("/")[-1].split(".")[0].split("_")[-1])
    if epoch >= args.num_epochs:
        continue
    logits = torch.load(logits_file)
    predictions = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    jtt = JTTInference(
        predictions=predictions,
        class_labels=valset.labels
    )
    epoch_group_partition = jtt.infer_groups()

    upsampled_indices = epoch_group_partition[(0,1)]
    minority_indices = valset.group_partition[(0,1)]
    minority_indices.extend(valset.group_partition[(1,0)])
    # compute precision score on the validation set
    upsampled = np.zeros(len(predictions))
    upsampled[np.array(upsampled_indices)] = 1
    minority = np.zeros(len(predictions))
    minority[np.array(minority_indices)] = 1
    precision = precision_score(minority, upsampled)
    if precision > max_precision:
        max_precision = precision
        args.infer_num_epochs = epoch
        group_partition = epoch_group_partition
        print("New best precision score:", precision, "at epoch", epoch)