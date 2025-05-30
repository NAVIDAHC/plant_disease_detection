import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Config ---
ARCH_NAME = "resnet50"
INFERENCE_TYPE = "base_model"
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Bacterial Spot", "Cercospora Leaf Spot", "Common Rust", "Early Blight",
    "Healthy", "Late Blight", "Northern Leaf Blight",
    "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]
MODEL_PATH = "models/resnet50/trained/resnet50_best_model.pth"
SAVE_BASE_DIR = "results/resnet50/base_model"

# --- Transform (same as training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load Model ---
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# --- Load and Process Image ---
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    rgb_img = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    return input_tensor, rgb_img

# --- Grad-CAM Visualization ---
def generate_and_save_gradcam(model, image_path, dataset_name):
    input_tensor, rgb_img = load_image(image_path)

    target_layer = model.layer4[-1]  # last conv layer in ResNet50
    cam = GradCAM(model=model, target_layers=[target_layer])

    outputs = model(input_tensor)
    pred_class = torch.argmax(outputs, dim=1).item()
    targets = [ClassifierOutputTarget(pred_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # --- Prepare Save Directory ---
    save_dir = os.path.join(SAVE_BASE_DIR, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- Save Outputs ---
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    plt.imsave(os.path.join(save_dir, f"{base_filename}_original.png"), rgb_img)
    plt.imsave(os.path.join(save_dir, f"{base_filename}_heatmap.png"), grayscale_cam, cmap='jet')
    plt.imsave(os.path.join(save_dir, f"{base_filename}_overlay.png"), cam_image)

    print(f"[INFO] Grad-CAM images saved to: {save_dir}")

# --- Main ---
if __name__ == "__main__":
    plantvillage_image = "dataset/plantvillage/test/Bacterial Spot/6aa07949-08b2-4945-a535-3f8b49f86599___GCREC_Bact.Sp 6344.JPG"  # replace with your image
    plantdoc_image = "dataset/plantdoc/test/Bacterial Spot/Leaf%2Bsymptoms%2BMDA.jpg"  # replace with your image

    model = load_model()

    print("[INFO] Generating Grad-CAM for PlantVillage image...")
    generate_and_save_gradcam(model, plantvillage_image, "plantvillage")

    print("[INFO] Generating Grad-CAM for PlantDoc image...")
    generate_and_save_gradcam(model, plantdoc_image, "plantdoc")
