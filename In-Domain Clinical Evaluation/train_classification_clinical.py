import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model_architecture import FFSwinClassifier
from dataset import BalancedOCTVolumeDataset, SimpleOCTDataset


def main():
    parser = argparse.ArgumentParser(
        description="FFSwin-Net Training (Patient-wise, Class-weighted)"
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default=r"clean_data_train_valid_test",
        help="Root data directory containing traindata/ and validationdata/"
    )

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save_dir', type=str, default="./checkpoints")

    args = parser.parse_args()

    # --------------------------------------------------
    # Resolve paths (Windows-safe)
    # --------------------------------------------------
    data_root = os.path.normpath(args.data_dir)
    train_dir = os.path.join(data_root, r"C:\Users\CSE-AI-Lab\Desktop\ChinmayResearch\Dataset_FFSwin_classification_train_Vali_Test\traindata")
    val_dir   = os.path.join(data_root, r"C:\Users\CSE-AI-Lab\Desktop\ChinmayResearch\Dataset_FFSwin_classification_train_Vali_Test\validationdata")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"[ERROR] traindata not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"[ERROR] validationdata not found: {val_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --------------------------------------------------
    # DATASETS
    # --------------------------------------------------
    train_ds = BalancedOCTVolumeDataset(
        root_dir=train_dir,
        return_label=True
    )

    val_ds = SimpleOCTDataset(
        root_dir=val_dir,
        return_label=True
    )

    print(f"📊 Training volumes (augmented): {len(train_ds)}")
    print(f"📊 Validation volumes (real):   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    model = FFSwinClassifier(num_classes=3).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=10,   # ⬅️ important
    min_lr=1e-6
)
    # IMPORTANT:
    # class index: 0=DryAMD, 1=WetAMD, 2=NonAMD
    class_weights = torch.tensor([3.0, 3.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --------------------------------------------------
    # LOGGING & CHECKPOINTS
    # --------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)

    log_path = os.path.join(args.save_dir, "training_log.txt")
    best_model_path = os.path.join(args.save_dir, "classifier_best.pth")
    final_model_path = os.path.join(args.save_dir, "classifier_final.pth")

    best_val_acc = 0.0
    log_file = open(log_path, "w", encoding="utf-8")

    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        for vol, lbl in train_loader:
            vol = vol.to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad()
            logits = model(vol)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)

        train_acc = 100.0 * correct / total

        # ---------------- VALIDATION ----------------
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for vol, lbl in val_loader:
                vol = vol.to(device)
                lbl = lbl.to(device)
                logits = model(vol)
                pred = logits.argmax(dim=1)
                val_preds.append(pred.item())
                val_labels.append(lbl.item())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_acc = (val_preds == val_labels).mean() * 100.0

        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]['lr']

        msg = (
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Loss {running_loss / len(train_loader):.4f} | "
            f"TrainAcc {train_acc:.2f}% | "
            f"ValAcc {val_acc:.2f}% | "
            f"LR {lr:.2e}"
        )

        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        # Validation confusion matrix (every 5 epochs)
        if epoch % 5 == 0:
            cm = confusion_matrix(val_labels, val_preds, labels=[0, 1, 2])
            print("Validation Confusion Matrix:\n", cm)
            log_file.write(f"Confusion Matrix (Epoch {epoch}):\n{cm}\n")
            log_file.flush()

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved (ValAcc={val_acc:.2f}%)")

    # --------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------
    torch.save(model.state_dict(), final_model_path)
    print("🎯 Training finished. Final model saved.")
    log_file.close()


if __name__ == "__main__":
    main()
