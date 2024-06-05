import requests
from PIL import Image
import torch
from torchvision import models, transforms
import warnings

warnings.filterwarnings("ignore")

model = models.resnet50(pretrained=True)
model.eval()


def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

def predict(image_path):
    input_batch = preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_label = labels[torch.argmax(probabilities).item()]

    return predicted_label,probabilities

image_path = "apple/Image_7.jpg"
predicted_label, probabilities = predict(image_path)
print(f"Predicted label: {predicted_label}")
print(f"Probabilities: {probabilities}")
