import torch
from torchvision import models
import torch.nn as nn


def load_model(model_path=None, NUM_CLASSES=None):
    """
    Load EfficientNet B0 with custom classifier for NUM_CLASSES
    """
    model = models.efficientnet_b0(weights=None)

    # Replace classifier for your dataset
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model
