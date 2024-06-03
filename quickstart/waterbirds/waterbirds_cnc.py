import torch 
import argparse
from spuco.evaluate import Evaluator

parser = argparse.ArgumentParser(description="Tune and evaluate SSA.")
parser.add_argument(
    "--inf_lr", default=0.001, type=float)
parser.add_argument(
    "--inf_wd", default=1.0, type=float)
parser.add_argument(
    "--train_lr", default=0.0001, type=float)
parser.add_argument(
    "--train_wd", default=0.001, type=float)
parser.add_argument(
    "--accum", default=32, type=int)
parser.add_argument(
    "--lambda_ce", default=0.75, type=float)
parser.add_argument(
    "--gpu", type=int)
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu))

from spuco.utils import set_seed

set_seed(0)

from wilds import get_dataset
import torchvision.transforms as transforms

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True,  root_dir='/data')

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=transform
)

# Get the test set
test_data = dataset.get_subset(
    "test",
    transform=transform
)

# Get the val set
val_data = dataset.get_subset(
    "val",
    transform=transform
)

from spuco.datasets import WILDSDatasetWrapper

trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)
valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True)

from spuco.models import model_factory
from torch.optim import SGD
from spuco.group_inference import CorrectNContrastInference

# initialize model
model = model_factory("resnet50", trainset[0][0].shape, 2, pretrained=True).to(device)

# state_dict_load = torch.load('./pretrained_model/initial_erm/waterbirds_erm_regularized.pt')['state_dict']
# param_name_classifier = ['module.fc.weight', 'module.fc.bias']
# prefix = 'module.'
# state_dict = model.backbone.state_dict()
# for key in state_dict_load.keys():
#     if key not in param_name_classifier:
#         key_shorten = key[len(prefix):]
#         if key_shorten not in state_dict:
#             print(key_shorten)
#             raise NotImplementedError
#         else:
#             state_dict[key_shorten] = state_dict_load[key]
# model.backbone.load_state_dict(state_dict)

# initialize CNC
print('initializing')
cnc_inf = CorrectNContrastInference(
        trainset=trainset,
        model=model,
        batch_size=128,
        optimizer=SGD(model.parameters(), lr=args.inf_lr, weight_decay=args.inf_wd, momentum=0.9),
        num_epochs=2,
        device=device,
        verbose=True
    )

print('inferring groups ...')
group_partition = trainset.group_partition
# cnc_inf.infer_groups(train=False)
# torch.save(cnc_inf.trainer.model.state_dict(), 'log_cnc/pretrained_model_LR[{}]_WD[{}]_debug.pth'.format(args.inf_lr, args.inf_wd))

for key in sorted(group_partition.keys()):
    print(key, len(group_partition[key]))

from torch.optim import SGD

model = model_factory("resnet50", trainset[0][0].shape, 2).to(device)

from spuco.datasets import GroupLabeledDatasetWrapper
from spuco.robust_train import CorrectNContrastTrain

invariant_trainset = GroupLabeledDatasetWrapper(trainset, group_partition)

val_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)

cnc_train = CorrectNContrastTrain(
    trainset=invariant_trainset,
    val_evaluator=val_evaluator,
    model=model,
    batch_size=68,
    optimizer_encoder=SGD(model.backbone.parameters(), lr=args.train_lr, weight_decay=args.train_wd, momentum=0.9),
    optimizer_classifier=SGD(model.classifier.parameters(), lr=args.train_lr, weight_decay=args.train_wd, momentum=0.9),
    accum=args.accum,
    num_pos=17, 
    num_neg=17,
    num_epochs=50,
    lambda_ce=args.lambda_ce,
    temp=0.1,
    device=device,
    verbose=True
)

print("training ...")

cnc_train.train()
evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)
print("evaluating on test")
evaluator.evaluate()

with open('./log_cnc/downloaded_LR[{}]_WD[{}]_AC[{}]_LA[{}].txt'.format(args.train_lr, args.train_wd, args.accum, args.lambda_ce), 'w') as f:
    print(evaluator.worst_group_accuracy, file = f)
    print(evaluator.average_accuracy, file = f)
evaluator.evaluate_spurious_task()
