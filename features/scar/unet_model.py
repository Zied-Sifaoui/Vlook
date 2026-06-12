import torch
import segmentation_models_pytorch as smp

def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model.eval()
    return model