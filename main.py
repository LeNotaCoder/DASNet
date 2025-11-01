import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from models import DASNet
from data import MyDataset
from train import train_model
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from torchvision.models import alexnet, resnet50, inception_v3, vgg16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Data Loading
# ------------------------------
path = ""  # Dataset Path
os.makedirs(path, exist_ok=True)

dr, normal = [], []
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

# ------------------------------
# Transforms
# ------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------
# Models
# ------------------------------
num_classes = 2
models = {
    "AlexNet": alexnet(pretrained=False, num_classes=num_classes),
    "ResNet50": resnet50(pretrained=False, num_classes=num_classes),
    "InceptionV3": inception_v3(pretrained=False, aux_logits=False, num_classes=num_classes),
    "VGG16": vgg16(pretrained=False, num_classes=num_classes),
    "DASNet": DASNet()
}

# Modify classifier layers
models["AlexNet"].classifier[6] = nn.Linear(4096, num_classes)
models["ResNet50"].fc = nn.Linear(2048, num_classes)
models["InceptionV3"].fc = nn.Linear(2048, num_classes)
models["VGG16"].classifier[6] = nn.Linear(4096, num_classes)

# ------------------------------
# 3-Fold Cross Validation
# ------------------------------
kf = KFold(n_splits=3, shuffle=True, random_state=42)
results = []

for model_name, base_model in models.items():
    print(f"\n{'='*20}\nTraining {model_name} with 3-Fold CV\n{'='*20}")

    fold_metrics = {"Accuracy": [], "F1": [], "Precision": [], "Recall": []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"\n--- Fold {fold + 1}/3 ---")

        X_train = [data[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test = [data[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]

        # Datasets and DataLoaders
        train_dataset = MyDataset(X_train, y_train, transform=transform)
        test_dataset = MyDataset(X_test, y_test, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)

        # Clone model for each fold to reset weights
        model = type(base_model)() if model_name != "DASNet" else DASNet()
        model = model.to(device)

        # Ensure classifier layers are redefined for each clone
        if model_name == "AlexNet":
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == "ResNet50":
            model.fc = nn.Linear(2048, num_classes)
        elif model_name == "InceptionV3":
            model.fc = nn.Linear(2048, num_classes)
        elif model_name == "VGG16":
            model.classifier[6] = nn.Linear(4096, num_classes)

        acc, f1, precision, recall = train_model(model, train_loader, test_loader, name=f"{model_name}_fold{fold+1}", device=device, epochs=25)

        fold_metrics["Accuracy"].append(acc)
        fold_metrics["F1"].append(f1)
        fold_metrics["Precision"].append(precision)
        fold_metrics["Recall"].append(recall)

    # Average metrics across folds
    avg_acc = np.mean(fold_metrics["Accuracy"])
    avg_f1 = np.mean(fold_metrics["F1"])
    avg_precision = np.mean(fold_metrics["Precision"])
    avg_recall = np.mean(fold_metrics["Recall"])

    results.append({
        "Model": model_name,
        "Accuracy": avg_acc,
        "F1 Score": avg_f1,
        "Precision": avg_precision,
        "Recall": avg_recall
    })

# ------------------------------
# Save and Print Final Results
# ------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print("\nFinal 3-Fold CV Results:")
print(results_df.to_string(index=False))
results_df.to_csv("model_results_3fold.csv", index=False)
