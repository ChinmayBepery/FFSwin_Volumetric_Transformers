import os
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import SimpleOCTDataset

# --- CONFIGURATION ---
# Note: We check the CLASSIFIER on the CLEAN (Denoised) data
DATA_DIR =  r"C:\Users\CSE-AI-Lab\Desktop\ChinmayResearch\PseudoFFSwin_Train_vali_test\clean_data_train_valid_test\testdata"
MODEL_PATH =  r"checkpoints\classifier_best.pth" 
#model_path = r"checkpoints\classifier_best.pth"
#data_root = r"C:\Users\CSE-AI-Lab\Desktop\ChinmayResearch\PseudoFFSwin_Train_vali_test\clean_data_train_valid_test\testdata"

IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CLASS MAPPING ---
# Ensure this matches your training logic
CLASSES = ['DryAMD', 'WetAMD', 'NonAMD']
LABEL_MAP = {name: i for i, name in enumerate(CLASSES)}

# --- MODEL DEFINITION ---
# IMPORTANT: This must match exactly what you used in train_classification.py
# If you moved the class to model_architecture.py, import it instead.
class FFSwinTransformer(nn.Module):
    def __init__(self, num_classes=3):
        super(FFSwinTransformer, self).__init__()
        # Placeholder backbone matching your training script
        # Replace this with your actual FFSwin implementation when ready
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 16, 16)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(16*16*16, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def get_label_from_path(folder_path):
    # folder_path example: /content/local_data_clean/Dry/1060RNIOH...
    parent = os.path.dirname(folder_path)
    class_name = os.path.basename(parent)
    return LABEL_MAP.get(class_name, 2) # Default to NonAMD if unknown

def check_one_sample():
    print("--- Running Visual Spot Check ---")
    
    # 1. Load Data
    # We use SimpleOCTDataset because we want distinct patients, no oversampling
    dataset = SimpleOCTDataset(DATA_DIR, img_size=IMG_SIZE)
    if len(dataset) == 0:
        print("No clean data found! Did you run run_inference.py?")
        return

    # 2. Load Model
    model = FFSwinTransformer(num_classes=3).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded classifier weights from {MODEL_PATH}")
    else:
        print("⚠️ Warning: Model weights not found. Using random weights.")
    
    model.eval()

    # 3. Pick Random Patient
    idx = random.randint(0, len(dataset) - 1)
    vol_tensor, folder_path = dataset[idx] # Returns (1, D, H, W), path
    
    # Prepare Input
    input_tensor = vol_tensor.unsqueeze(0).to(DEVICE) # Add batch dim -> (1, 1, D, H, W)
    
    # Get True Label
    true_label_idx = get_label_from_path(folder_path)
    true_label_str = CLASSES[true_label_idx]

    # 4. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
    
    pred_label_str = CLASSES[pred_idx]
    confidence = probs[0][pred_idx].item() * 100

    # 5. Visualize
    print(f"Patient: {os.path.basename(folder_path)}")
    print(f"TRUE Label:      {true_label_str}")
    print(f"PREDICTED Label: {pred_label_str} ({confidence:.2f}%)")
    
    # Show middle slice
    mid_slice = vol_tensor[0, vol_tensor.shape[1]//2, :, :].numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mid_slice, cmap='gray')
    plt.title(f"True: {true_label_str} | Pred: {pred_label_str}")
    plt.axis('off')
    plt.show()

def evaluate_full_accuracy():
    print("\n--- Running Full Dataset Evaluation ---")
    dataset = SimpleOCTDataset(DATA_DIR, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = FFSwinTransformer(num_classes=3).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    correct = 0
    total = 0
    
    print(f"Evaluating {len(dataset)} patients...")
    
    with torch.no_grad():
        for vol, paths in dataloader:
            vol = vol.to(DEVICE)
            
            # Get True Labels
            # paths is a tuple of strings (batch_size=1)
            labels = [get_label_from_path(p) for p in paths]
            target = torch.tensor(labels).to(DEVICE)
            
            # Predict
            outputs = model(vol)
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100 * correct / total
    print(f"Accuracy of the network on clean data: {acc:.2f}%")

if __name__ == "__main__":
    # You can choose to run one or both
    check_one_sample()
    # evaluate_full_accuracy() # Uncomment this to scan the whole folder
