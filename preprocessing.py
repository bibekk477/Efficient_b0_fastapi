from PIL import Image
from torchvision import transforms


def preprocess_image(image: Image.Image):
    """
    Preprocesses an image for EfficientNet B0.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)  # add batch dimension
