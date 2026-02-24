import os
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from model_architecture import OCT3DDenoisingAutoencoder
from dataset import SimpleOCTDataset

DATA_DIR = "/content/local_data"
MODEL_PATH = "oct_denoiser_final.pth"

def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data & Model
    dataset = SimpleOCTDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No data found to check.")
        return

    model = OCT3DDenoisingAutoencoder().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print("Model loaded.")
    else:
        print("Model weights not found. Running with random weights (Output will be garbage).")

    model.eval()
    
    # Pick Random Sample
    idx = random.randint(0, len(dataset)-1)
    vol, path = dataset[idx]
    vol = vol.unsqueeze(0).to(device) # Add batch dim

    with torch.no_grad():
        out = model(vol)

    # Visualize Middle Slice
    mid = vol.shape[2] // 2
    orig = vol[0, 0, mid].cpu().numpy()
    denoised = out[0, 0, mid].cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Patient: {os.path.basename(path)}")
    plt.subplot(1, 2, 1); plt.imshow(orig, cmap='gray'); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(denoised, cmap='gray'); plt.title("Denoised")
    plt.show()

if __name__ == "__main__":
    check()
