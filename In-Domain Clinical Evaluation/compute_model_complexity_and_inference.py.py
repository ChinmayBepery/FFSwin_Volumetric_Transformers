import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from thop import profile
from tqdm import tqdm

# ==============================
# IMPORT YOUR MODEL HERE
# ==============================
# For FFSwin:
from model_architecture import FFSwinClassifier

# For ResNet3D:
# from your_resnet_file import ResNet3D18_OCT

from dataset import SimpleOCTDataset


# ==============================
# USER SETTINGS
# ==============================
TEST_DATA_ROOT = r"G:\My Drive\PhD_BanglaOCT2025_data\BanglaOCT2025\FFSwin_classifi_Split_Train_Valid_Test_DenoisData\testdata"
MODEL_PATH = r"checkpoints\classifier_best_55_epoch.pth"
OUTPUT_DIR = "./Computational_complexity_reports"
BATCH_SIZE = 1
NUM_WARMUP = 10   # warm-up iterations for stable timing

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# LOAD MODEL
# ==============================
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFSwinClassifier(num_classes=3).to(device)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model, device


# ==============================
# COUNT PARAMETERS
# ==============================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ==============================
# COMPUTE FLOPs
# ==============================
def compute_flops(model, device):
    dummy_input = torch.randn(1, 1, 33, 256, 256).to(device)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    flops_g = flops / 1e9   # GFLOPs
    return flops_g


# ==============================
# MEASURE INFERENCE TIME
# ==============================
def measure_inference_time(model, device, test_loader):
    timings = []

    # Warm-up GPU
    with torch.no_grad():
        for i, (vol, _) in enumerate(test_loader):
            if i >= NUM_WARMUP:
                break
            vol = vol.to(device)
            _ = model(vol)

    torch.cuda.synchronize()

    # Real measurement
    with torch.no_grad():
        for vol, _ in tqdm(test_loader, desc="Measuring Inference"):
            vol = vol.to(device)

            start = time.time()
            _ = model(vol)
            torch.cuda.synchronize()
            end = time.time()

            timings.append(end - start)

    avg_time = np.mean(timings)
    std_time = np.std(timings)

    return avg_time, std_time


# ==============================
# MAIN
# ==============================
def main():
    print("🚀 Computing Computational Metrics...")

    model, device = load_model(MODEL_PATH)

    test_dataset = SimpleOCTDataset(
        root_dir=TEST_DATA_ROOT,
        return_label=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # Parameters
    total_params, trainable_params = count_parameters(model)

    # FLOPs
    flops_g = compute_flops(model, device)

    # Inference Time
    avg_time, std_time = measure_inference_time(model, device, test_loader)

    # ==============================
    # PRINT RESULTS
    # ==============================
    print("\n==============================")
    print("📊 Model Complexity Report")
    print("==============================")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"FLOPs per forward pass: {flops_g:.3f} GFLOPs")
    print(f"Average Inference Time per patient: {avg_time*1000:.2f} ms")
    print(f"Std Dev Inference Time: {std_time*1000:.2f} ms")

    # ==============================
    # SAVE LOG
    # ==============================
    output_file = os.path.join(OUTPUT_DIR, "complexity_report.txt")

    with open(output_file, "w") as f:
        f.write("Model Complexity Report\n")
        f.write("=======================\n")
        f.write(f"Total Parameters: {total_params}\n")
        f.write(f"Trainable Parameters: {trainable_params}\n")
        f.write(f"FLOPs (GFLOPs): {flops_g:.3f}\n")
        f.write(f"Avg Inference Time (ms): {avg_time*1000:.2f}\n")
        f.write(f"Std Inference Time (ms): {std_time*1000:.2f}\n")

    print(f"\n✅ Report saved to: {output_file}")


if __name__ == "__main__":
    main()
