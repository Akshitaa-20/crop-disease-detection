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

# Transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

# Test
if __name__ == "__main__":
    path = "test.jpg"  # put your image name here
    result = predict_image(path)
    print("🌱 Prediction:", result)