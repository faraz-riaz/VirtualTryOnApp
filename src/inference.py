# src/inference.py
import torch
from PIL import Image
from torchvision import transforms
import subprocess

def load_model():
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True)
    model.eval()
    return model

def infer(model, input_image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)
    output = model(input_tensor)['out'].argmax(1)
    return output
