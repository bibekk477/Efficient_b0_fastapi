from PIL import Image
from torchvision import transforms
import torchvision


def preprocess_image(image: Image.Image):
    """
    Preprocesses an image for EfficientNet B0.
    Returns a tensor with shape (1, C, H, W).
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)  # add batch dimension
