import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast, GradScaler

from model_ffswin import FFSwinClassifier
from dataset import BalancedOCTVolumeDataset, SimpleOCTDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_patch', type=int, default=4, choices=[1,2,4],
                        help='Depth kernel size: 1 (1x4x4), 2 (2x4x4), or 4 (4x4x4)')
    parser.add_argument('--data_dir', type=str, 
                        default=r"D:/OCTData/BanglaOCT2025_Dataset/ANONYMIZED_Denoise_Split_Train_Valid_Test_Data",
                        help='Root directory containing traindata/ and validationdata/')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (4 works on A2000 with 40 slices)')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save_dir', type=str, default='./ablation_checkpoints')
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Paths
    data_root = os.path.normpath(args.data_dir)
    train_dir = os.path.join(data_root, "traindata")
    val_dir   = os.path.join(data_root, "validationdata")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Depth patch kernel: {args.depth_patch}x4x4")
    print(f"Batch size: {args.batch_size}")

    # Datasets (use 40‑slice padding from dataset.py)
    train_ds = BalancedOCTVolumeDataset(train_dir, return_label=True)
    val_ds = SimpleOCTDataset(val_dir, return_label=True)
    print(f"Training samples (balanced): {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # Model
    model = FFSwinClassifier(num_classes=3, depth_patch=args.depth_patch).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=10, min_lr=1e-6)

    # Class weights (Dry=3, Wet=3, Non=1)
    class_weights = torch.tensor([3.0, 3.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Mixed precision
    scaler = GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, f"training_log_depth{args.depth_patch}.txt")
    best_model_path = os.path.join(args.save_dir, f"classifier_best_depth{args.depth_patch}.pth")

    best_val_acc = 0.0
    log_file = open(log_path, "w", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        for vol, lbl in train_loader:
            vol = vol.to(device)
            lbl = lbl.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(vol)
                loss = criterion(logits, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)

        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for vol, lbl in val_loader:
                vol = vol.to(device)
                lbl = lbl.to(device)
                with autocast():
                    logits = model(vol)
                pred = logits.argmax(dim=1)
                val_preds.append(pred.item())
                val_labels.append(lbl.item())

        val_acc = (np.array(val_preds) == np.array(val_labels)).mean() * 100.0
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        msg = (f"Epoch [{epoch:2d}/{args.epochs}] | "
               f"Loss {running_loss/len(train_loader):.4f} | "
               f"TrainAcc {train_acc:5.2f}% | "
               f"ValAcc {val_acc:5.2f}% | "
               f"LR {current_lr:.2e}")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        if epoch % 5 == 0:
            cm = confusion_matrix(val_labels, val_preds, labels=[0,1,2])
            print("Validation Confusion Matrix:\n", cm)
            log_file.write(f"Confusion Matrix (Epoch {epoch}):\n{cm}\n")
            log_file.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved (ValAcc={val_acc:.2f}%)")

        torch.cuda.empty_cache()

    log_file.close()
    print(f"Training finished. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()