import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import UnidentifiedImageError

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
DATA_DIR = r"C:\Users\User\Desktop\ivan files\plant_disease_detection\dataset\PlantVillage_modified"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
SAVE_DIR = r"C:\Users\User\Desktop\ivan files\plant_disease_detection\models\architectures_data augmented\resnet"

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Safe image loader to handle missing/corrupt images
def safe_image_loader(path):
    try:
        return datasets.folder.default_loader(path)
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"⚠️ Skipping missing/corrupt image: {path} ({str(e)})")
        return None  # Return None to mark as an invalid sample

# Custom dataset class to skip missing images
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = safe_image_loader(path)

        # Skip invalid images by returning the next valid sample
        if sample is None:
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

# Load dataset using CustomImageFolder
full_dataset = CustomImageFolder(root=TRAIN_DIR, transform=transform)
labels = np.array([label for _, label in full_dataset.samples])

# Set up k-Fold Cross-Validation
k_folds = 2
epochs = 5
batch_size = 16
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store per-fold results
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n🔹 Fold {fold+1}/{k_folds}")

    # Create train/val subsets
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    # Define dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(full_dataset.classes))
    model = model.to(device)

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Track best model
    best_f1 = 0.0
    best_model_path = os.path.join(SAVE_DIR, f"resnet50_plantvillage_best_fold{fold+1}.pth")

    # Store per-epoch metrics
    epoch_results = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        epoch_start = time.time()

        with tqdm(train_loader, desc=f"Fold {fold+1} | Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() / len(train_loader)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=loss.item(), acc=correct/total)

        train_acc = correct / total
        epoch_time = time.time() - epoch_start
        print(f"Fold {fold+1} | Epoch {epoch+1} completed in {epoch_time:.2f}s, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Evaluate model
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Compute AUC score (if more than one class)
        if len(full_dataset.classes) > 2:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        else:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])

        print(f"Fold {fold+1} | Val Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # Save per-epoch results
        epoch_results.append([epoch+1, train_loss, train_acc, accuracy, precision, recall, f1, auc])

        # Save best model for this fold
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)

    # Save per-fold results
    fold_results.append([fold+1, accuracy, precision, recall, f1, auc])

    # Save per-epoch results to CSV
    epoch_df = pd.DataFrame(epoch_results, columns=["Epoch", "Train Loss", "Train Acc", "Val Acc", "Precision", "Recall", "F1", "AUC"])
    epoch_df.to_csv(os.path.join(SAVE_DIR, f"resnet50_plantvillage_fold{fold+1}_epochs.csv"), index=False)

# Convert results to DataFrame & Save
df_results = pd.DataFrame(fold_results, columns=["Fold", "Accuracy", "Precision", "Recall", "F1", "AUC"])
df_results.loc["Mean"] = df_results.mean()
timestamp = time.strftime("%Y%m%d-%H%M%S")
df_results.to_csv(os.path.join(SAVE_DIR, f"resnet50_plantvillage_results_{timestamp}.csv"), index=False)

print("🔹 Cross-validation complete. Results and best models saved.")
