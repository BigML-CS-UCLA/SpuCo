from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List
from torch import nn
from torch.utils.data import Dataset 
from tqdm import tqdm 
import numpy as np 
import torch

from spuco.utils.misc import get_model_outputs

class GradCamEvaluator:
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        masks: List[np.array],
        device: torch.device = torch.device("cpu"),
        verbose: bool = False
    ):
        self.device = device
        self.model = model.to(self.device)
        self.dataset = dataset
        self.masks = masks
        self.verbose = verbose 
        if self.verbose:
            print("Computing predicted labels")
        self.pred_labels = torch.argmax(get_model_outputs(
            self.model, 
            self.dataset,
            self.device,
            verbose=self.verbose
        ), dim=-1).detach().cpu().tolist()
        
    def evaluate(self):
        target_layers = [self.model.backbone.layer4[-1]]
        cam = GradCAM(model=self.model.backbone, target_layers=target_layers)
        scores = []
        for i in tqdm(range(len(self.dataset)), desc="Computing IoU between gradcam and mask", disable=not self.verbose):
            targets = [ClassifierOutputTarget(self.pred_labels[i])]
            input_tensor = self.dataset[i][0].unsqueeze(dim=0).to(self.device)
            predicted_mask = cam(input_tensor=input_tensor, targets=targets)
            scores.append(GradCamEvaluator.compute_iou(predicted_mask, self.masks[i]))
        return np.mean(scores)

    @staticmethod
    def compute_iou(predicted_mask, ground_truth_mask, threshold=0.5):
        # Thresholding
        predicted_binary = (predicted_mask > threshold).astype(int)
        ground_truth_binary = (ground_truth_mask > 0.5).astype(int)

        # Intersection
        intersection = np.logical_and(predicted_binary, ground_truth_binary).sum()

        # Union
        union = np.logical_or(predicted_binary, ground_truth_binary).sum()

        # Calculate IoU
        iou = intersection / union if union != 0 else 0.0

        return iou