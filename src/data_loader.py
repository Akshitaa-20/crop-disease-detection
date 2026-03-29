import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Path to dataset
data_dir = "data/plantvillage"

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# Save class names (IMPORTANT for later)
class_names = full_dataset.classes

# Reduce dataset size (for faster training)
small_size = 4000
full_dataset, _ = random_split(full_dataset, [small_size, len(full_dataset) - small_size])

# Train-validation split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply validation transforms
val_dataset.dataset.transform = val_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Debug prints
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))

if __name__ == "__main__":
    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
