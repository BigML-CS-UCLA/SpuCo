import argparse
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD
from wilds import get_dataset
from torch.utils.data import Dataset 
from PIL import Image
import numpy as np 

from spuco.datasets import WILDSDatasetWrapper
from spuco.datasets.base_spuco_dataset import MASK_CORE, MASK_SPURIOUS
from spuco.evaluate import Evaluator
from spuco.robust_train import ERM
from spuco.models import model_factory
from spuco.utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--results_csv", type=str, default="results/waterbirds_erm.csv")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--mask-type", type=str, default="", choices=[MASK_CORE, MASK_SPURIOUS, ""])

args = parser.parse_args()
if args.mask_type == "":
    args.mask_type = None

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

train_data = None
val_data = None 
test_data = None 

if args.mask_type is None:
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
else:
    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=None
    )

    val_data = dataset.get_subset(
        "val",
        transform=None
    )

    # Get the training set
    test_data = dataset.get_subset(
        "test",
        transform=None 
    )

    class MaskedDataWrapper(Dataset):
        def __init__(self, data, transforms, filenames):
            self.data = data 
            self.transforms = transforms
            self.filenames = filenames
            self.mask_type = args.mask_type
            self.n_classes = data.n_classes
            self.metadata_fields = data.metadata_fields
            self.y_array = data.y_array
            self.metadata_array = data.metadata_array
        
        def __getitem__(self, i):
            image, label, metadata = self.data[i]
            mask = np.array(Image.open(f"/data/waterbirds_v1.0/segmentations/{self.filenames[i]}").convert('L'))
            if self.mask_type == MASK_CORE:
                mask = np.invert(mask)
            image = np.array(image)
            try:
                for i in range(3):
                    image[:, :, i] *= mask
            except:
                print("Error, not masking this image")
            return self.transforms(Image.fromarray(image)), label, metadata
        
        def __len__(self):
            return len(self.data)
    import numpy as np 

    split_mask = dataset.split_array == dataset.split_dict["train"]
    train_split = np.where(split_mask)[0]

    split_mask = dataset.split_array == dataset.split_dict["val"]
    val_split = np.where(split_mask)[0]
    
    split_mask = dataset.split_array == dataset.split_dict["test"]
    test_split = np.where(split_mask)[0]

    train_filenames = [dataset._input_array[i].replace(".jpg", ".png") for i in train_split]
    val_filenames = [dataset._input_array[i].replace(".jpg", ".png") for i in val_split]
    test_filenames = [dataset._input_array[i].replace(".jpg", ".png") for i in test_split]


    train_data = MaskedDataWrapper(train_data, transform, train_filenames)
    val_data = MaskedDataWrapper(val_data, transform, val_filenames)
    test_data = MaskedDataWrapper(test_data, transform, test_filenames)

trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True)
testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)

model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)

erm = ERM(
    model=model,
    val_evaluator=valid_evaluator,
    num_epochs=args.num_epochs,
    trainset=trainset,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
erm.train()

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

results = pd.DataFrame(index=[0])
results["timestamp"] = pd.Timestamp.now()
results["seed"] = args.seed
results["pretrained"] = args.pretrained
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
    model=erm.best_model,
    device=device,
    verbose=True
)
evaluator.evaluate()

results["spurious_attribute_prediction"] = evaluator.evaluate_spurious_attribute_prediction()
results["early_stopping_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["early_stopping_average_accuracy"] = evaluator.average_accuracy

print(results)

print(results["spurious_attribute_prediction"])

if os.path.exists(args.results_csv):
    results_df = pd.read_csv(args.results_csv)
else:
    results_df = pd.DataFrame()

results_df = pd.concat([results_df, results], ignore_index=True)
results_df.to_csv(args.results_csv, index=False)

print('Done!')
print('Results saved to', args.results_csv)


