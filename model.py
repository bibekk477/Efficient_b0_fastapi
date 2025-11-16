import torch
import torch.nn as nn
from torchvision import models


def load_model(model_path, num_classes, device):
    """
    Load EfficientNet B0 with a custom classifier.
    """
    # Create model with no pretrained weights
    model = models.efficientnet_b0(weights=None)

    # Replace classifier layer
    model.classifier[1] = nn.Linear(
        in_features=model.classifier[1].in_features, out_features=num_classes
    )

    # Load trained weights
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    return model.to(device)
