# model.py
import torch
import torchvision.models as models
import torch.nn as nn


# Example: Using ResNet18 for simplicity
def load_model(model_path="model.pth"):
    model = models.resnet18(pretrained=False)
    num_classes = 10  # Change according to your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model
