import os
import torch
import torch.nn as nn
from torchvision import models 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# --- Config ---
ARCH_NAME = "regnet"
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
MODEL_PATH = "models/regnet/trained/regnet_best_model.pth"
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Model Loader ---
def load_regnet_model():
    model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()

# --- Inference ---
def run_base_inference(dataset_path, dataset_name):
    print(f"[INFO] Running base inference on {dataset_name}...")
    dataset = datasets.ImageFolder(dataset_path, transform=TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = load_regnet_model()
    y_true, y_pred = [], []

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    out_dir = f"results/regnet/base_model/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(out_dir, f"{ARCH_NAME}_inference_result.csv"), index=True)
    with open(os.path.join(out_dir, f"{ARCH_NAME}_inference_result.txt"), "w") as f:
        f.write(report_df.to_string())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{ARCH_NAME.upper()} Base Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ARCH_NAME}_confusion_matrix.png"))
    plt.close()

    print(f"[✅] Results saved to: {out_dir}")

# --- Main ---
if __name__ == "__main__":
    run_base_inference("dataset/plantvillage/test", "plantvillage")
    run_base_inference("dataset/plantdoc/test", "plantdoc")