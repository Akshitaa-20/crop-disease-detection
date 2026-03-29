import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import get_model
from data_loader import class_names

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model(len(class_names))
model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌱 Crop Disease Detection",
    description="Upload a leaf image to detect disease"
)

interface.launch()