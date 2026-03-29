import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import train_loader, val_loader, class_names
from model import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Number of classes
num_classes = len(class_names)

# Load model
model = get_model(num_classes)
model = model.to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Epochs (keep small to save time)
epochs = 1

# Track best accuracy
best_accuracy = 0

for epoch in range(epochs):
    # TRAINING
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1}, Loss: {total_loss:.4f}")

    # VALIDATION
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # ✅ SAVE BEST MODEL
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "outputs/model.pth")
        print("✅ Best model saved!")

print("\nTraining finished!")