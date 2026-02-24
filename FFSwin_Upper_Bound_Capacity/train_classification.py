import torch, argparse, os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model_architecture import FFSwinClassifier
from dataset import BalancedOCTVolumeDataset, SimpleOCTDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="G:\\My Drive\\PhD_BanglaOCT2025_data\\BanglaOCT2025\\ANONYMIZED_DATA_Macula_33_images_Denoised") 
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default="FFSwin_Upper_Bound_Capacity\models")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Classifier on {device}...")
    
    # 1. DATA PREPARATION
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data folder '{args.data_dir}' not found!")
        return

    # Train on Balanced (Virtual Oversampling)
    train_ds = BalancedOCTVolumeDataset(args.data_dir, return_label=True)
    # Validate on Real Imbalanced Data
    val_ds = SimpleOCTDataset(args.data_dir, return_label=True)
    
    if len(train_ds) == 0:
        print("❌ Error: Dataset is empty!")
        return

    print(f"✅ Balanced Training Samples: {len(train_ds)}")
    print(f"✅ Real Validation Samples:  {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # 2. MODEL & OPTIMIZER
    model = FFSwinClassifier(num_classes=3).to(device)
    
    # IMPROVEMENT 1: Use AdamW (Weight Decay) to prevent overfitting
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    
    # IMPROVEMENT 2: Add Scheduler to fix the "See-Saw" effect
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)

    crit = nn.CrossEntropyLoss()
    
    # 3. SETUP SAVING
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0
    best_path = os.path.join(args.save_dir, "classifier_best.pth")
    final_path = os.path.join(args.save_dir, "classifier_final.pth")

    print("Starting Training Loop...")

    for ep in range(args.epochs):
        model.train()
        loss_sum = 0; correct = 0; total = 0
        
        for vol, label in train_dl:
            vol, label = vol.to(device), label.to(device)
            opt.zero_grad()
            out = model(vol)
            loss = crit(out, label)
            loss.backward()
            opt.step()
            
            loss_sum += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            
        # VALIDATION
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for v_vol, v_lbl in val_dl:
                v_vol, v_lbl = v_vol.to(device), v_lbl.to(device)
                out = model(v_vol)
                _, p = torch.max(out, 1)
                all_preds.extend(p.cpu().numpy())
                all_labels.extend(v_lbl.cpu().numpy())

        # METRICS
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_acc = (all_preds == all_labels).mean() * 100
        
        # Scheduler Step
        #scheduler.step(val_acc)
        # Add this manual logging
        current_lr = opt.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        print(f"Ep {ep+1}/{args.epochs} | Loss: {loss_sum/len(train_dl):.4f} | Train Acc: {correct/total*100:.1f}% | Val Acc: {val_acc:.1f}%")
        
        # IMPROVEMENT 3: Print Confusion Matrix every 5 epochs
        if (ep+1) % 5 == 0:
            print("Confusion Matrix (True vs Pred):")
            print(confusion_matrix(all_labels, all_preds, labels=[0,1,2]))

        # IMPROVEMENT 4: Save BEST model, not just final
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"🎉 New Best Model Saved! ({val_acc:.1f}%)")

    # Save Final
    torch.save(model.state_dict(), final_path)
    print("Done.")

if __name__ == "__main__": main()