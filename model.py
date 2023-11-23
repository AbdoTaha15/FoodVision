import torch
from torch import nn
import torchvision
import os
from timeit import default_timer as timer
from typing import Tuple, Dict


def create_effnetb2_model(num_classes: int = 3, seed: int = 42):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )

    return model, transforms


def pred(img, effnetb2, effnetb2_transforms, class_names) -> Tuple[Dict, float]:
    start = timer()

    img = effnetb2_transforms(img).unsqueeze(0)

    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    end = timer()
    pred_time = round(end - start, 4)

    return pred_labels_and_probs, pred_time
