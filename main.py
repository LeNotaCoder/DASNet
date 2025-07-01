import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
from torchvision.models import alexnet, resnet50, inception_v3, vgg16
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
from models import DASNet
from train import train_model
from data import MyDataset
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data Loading 

path = "/home/cs23b1055/data/"
os.makedirs(path, exist_ok=True) 


dr = []
normal = []
n = 23548

for i in range(1, n + 1):
    dr_path = os.path.join(path, f"DR/{i}.jpg")
    n_path = os.path.join(path, f"NORMAL/{i}.jpg")
    
    if os.path.exists(dr_path):
        dr_image = cv2.imread(dr_path)
        dr_image = cv2.cvtColor(dr_image, cv2.COLOR_BGR2RGB)
        dr.append(dr_image)
    if os.path.exists(n_path):
        n_image = cv2.imread(n_path)
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)
        normal.append(n_image)

print(f"Loaded {len(dr)} DR images and {len(normal)} NORMAL images. Total: {len(dr) + len(normal)}")

data = dr + normal
labels = [1] * len(dr) + [0] * len(normal)



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, random_state=42)  # Use 80% for training

# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets and DataLoaders
train_dataset = MyDataset(X_train, y_train, transform=transform)
test_dataset = MyDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

# Define models
models = {
    "AlexNet": alexnet(pretrained=False, num_classes=2),
    "ResNet50": resnet50(pretrained=False, num_classes=2),
    "InceptionV3": inception_v3(pretrained=False, aux_logits=False, num_classes=2),
    "VGG16": vgg16(pretrained=False, num_classes=2),
    "MinMaxPool": DASNet()
}


num_classes = 2
models["AlexNet"].classifier[6] = nn.Linear(4096, num_classes)
models["ResNet50"].fc = nn.Linear(2048, num_classes)
models["InceptionV3"].fc = nn.Linear(2048, num_classes)
models["VGG16"].classifier[6] = nn.Linear(4096, num_classes)

# Train models
results = []
for name, model in models.items():
    print(f"\n Training {name}...")
    acc, f1, precision, recall = train_model(model, train_loader, test_loader, name=name, device=device, epochs=25)
    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    })

# Save results
results_df = pd.DataFrame(results)
print("\nFinal Evaluation Results:")
results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))
results_df.to_csv("model_results.csv", index=False)

