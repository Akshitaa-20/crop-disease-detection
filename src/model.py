import timm
import torch.nn as nn
def get_model(num_classes):
    #Load Pretrained EfficientNet 
    model = timm.create_model("efficientnet_b0",pretrained=True)
    #Replace final layer
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
if __name__ == "__main__":
    model=get_model(num_classes=38) #adjust if needed
    print(model)
    