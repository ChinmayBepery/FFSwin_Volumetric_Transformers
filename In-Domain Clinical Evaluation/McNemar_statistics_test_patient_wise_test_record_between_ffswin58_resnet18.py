import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import chi2
from model_architecture import FFSwinClassifier
import torchvision.models.video as models
import torch.nn as nn

# -------------------------------------------------------------
# USER SETTINGS
# -------------------------------------------------------------
TEST_ROOT = r"G:\My Drive\PhD_BanglaOCT2025_data\BanglaOCT2025\FFSwin_classifi_Split_Train_Valid_Test_DenoisData\testdata"

FFSWIN_MODEL_PATH = r"G:\My Drive\OCT_Project_PhD_Implementation\Experiment_1_FFSwin_Train_vali_test_Clinical_Experiment\checkpoints\classifier_best_55_epoch.pth"
RESNET_MODEL_PATH = r"G:\My Drive\OCT_Project_PhD_Implementation\Experiment_4_3D_ResNet_Train_vali_test_Clinical_Experiment\checkpoints_resnet3d\resnet3d18_best.pth"

OUTPUT_CSV = "McNemar_Clinical_Comparison_FFSwin_55_Resnet18.csv"

IMG_SIZE = (256, 256)
CLASSES = ["DryAMD", "WetAMD", "NonAMD"]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

# -------------------------------------------------------------
# MODEL LOADERS
# -------------------------------------------------------------
def load_ffswin(path, device):
    model = FFSwinClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

class ResNet3D18_OCT(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.r3d_18(weights=None)
        self.model.stem[0] = nn.Conv3d(
            1, 64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

def load_resnet(path, device):
    model = ResNet3D18_OCT(num_classes=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# -------------------------------------------------------------
# LOAD VOLUME
# -------------------------------------------------------------
def load_volume(path):
    files = sorted(glob.glob(os.path.join(path, "*.bmp")))
    if len(files) == 0:
        return None

    vol = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        vol.append(img)

    if len(vol) == 0:
        return None

    vol = np.array(vol) / 255.0
    if vol.shape[0] % 2 != 0:
        vol = np.concatenate([vol, vol[-1:]], axis=0)

    tensor = torch.FloatTensor(vol).unsqueeze(0).unsqueeze(0)
    return tensor

# -------------------------------------------------------------
# CLASSIFY
# -------------------------------------------------------------
def predict(model, device, volume):
    with torch.no_grad():
        volume = volume.to(device)
        logits = model(volume)
        pred = torch.argmax(logits, dim=1).item()
    return pred

# -------------------------------------------------------------
# MAIN EVALUATION
# -------------------------------------------------------------
def run_mcnemar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ffswin = load_ffswin(FFSWIN_MODEL_PATH, device)
    resnet = load_resnet(RESNET_MODEL_PATH, device)

    results = []

    print("\n🔎 Running patient-wise comparison...\n")

    for cls in CLASSES:
        cls_path = os.path.join(TEST_ROOT, cls)

        for patient in sorted(os.listdir(cls_path)):
            patient_path = os.path.join(cls_path, patient)
            if not os.path.isdir(patient_path):
                continue

            vol = load_volume(patient_path)
            if vol is None:
                continue

            gt = CLASS_TO_ID[cls]

            pred_ffswin = predict(ffswin, device, vol)
            pred_resnet = predict(resnet, device, vol)

            correct_ffswin = int(pred_ffswin == gt)
            correct_resnet = int(pred_resnet == gt)

            results.append({
                "Patient": patient,
                "GT": cls,
                "FFSwin_Pred": CLASSES[pred_ffswin],
                "ResNet_Pred": CLASSES[pred_resnet],
                "FFSwin_Correct": correct_ffswin,
                "ResNet_Correct": correct_resnet
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    # ---------------------------------------------------------
    # McNemar calculation
    # ---------------------------------------------------------
    b = len(df[(df["FFSwin_Correct"] == 1) & (df["ResNet_Correct"] == 0)])
    c = len(df[(df["FFSwin_Correct"] == 0) & (df["ResNet_Correct"] == 1)])

    print("\nContingency Table:")
    print(f"FFSwin correct / ResNet wrong (b): {b}")
    print(f"FFSwin wrong / ResNet correct (c): {c}")

    if b + c == 0:
        print("\n⚠ Models have identical predictions. McNemar not applicable.")
        return

    chi_square = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - chi2.cdf(chi_square, df=1)

    print(f"\nMcNemar χ² = {chi_square:.4f}")
    print(f"p-value = {p_value:.6f}")

    if p_value < 0.05:
        print("✅ Statistically significant difference (p < 0.05)")
    else:
        print("⚠ No statistically significant difference (p ≥ 0.05)")

    print(f"\nResults saved to {OUTPUT_CSV}")

# -------------------------------------------------------------
if __name__ == "__main__":
    run_mcnemar()
