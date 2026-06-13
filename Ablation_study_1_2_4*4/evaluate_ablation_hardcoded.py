import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from math import sqrt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from model_ffswin import FFSwinClassifier
from dataset import SimpleOCTDataset

# ============================================================
# ⚙️ CHANGE THESE FOR EACH MODEL (1, 2, or 4)
# ============================================================
DEPTH_PATCH = 4                     # 1, 2, or 4
CHECKPOINT = "ablation_checkpoints\classifier_best_depth4.pth"
TEST_DIR = r"D:\OCTData\BanglaOCT2025_Dataset\ANONYMIZED_Denoise_Split_Train_Valid_Test_Data\testdata"
SAVE_DIR = r"./eval_results_depth4"
BATCH_SIZE = 4
NUM_WORKERS = 4
# ============================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"depth{DEPTH_PATCH}_{timestamp}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating depth_patch={DEPTH_PATCH} from {CHECKPOINT}")

    # Dataset (pad to 40 slices automatically)
    test_ds = SimpleOCTDataset(root_dir=TEST_DIR, return_label=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Test samples: {len(test_ds)}")

    # Model
    model = FFSwinClassifier(num_classes=3, depth_patch=DEPTH_PATCH).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs, all_features = [], [], [], []

    with torch.no_grad():
        for vol, label in test_loader:
            vol, label = vol.to(device), label.to(device)
            # Manual feature extraction (model has no return_features)
            x = model.patch_embed(vol)
            x = x.permute(0, 2, 3, 4, 1)
            x = model.layers(x)
            x = model.norm(x)
            B, D, H, W, C = x.shape
            x = x.view(B, -1, C).transpose(1, 2)
            features = model.avgpool(x).flatten(1)
            logits = model.head(features)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)
    class_names = ['DryAMD', 'WetAMD', 'NonAMD']

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    acc = np.trace(cm) / np.sum(cm)
    acc_pct = acc * 100
    n = len(all_labels)
    ci = 1.96 * sqrt((acc * (1 - acc)) / n)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    # Save report
    log_path = os.path.join(SAVE_DIR, f"{prefix}_report.txt")
    with open(log_path, 'w') as f:
        f.write(f"FFSwin depth_patch={DEPTH_PATCH}\n")
        f.write(f"Checkpoint: {CHECKPOINT}\n")
        f.write(f"Test samples: {n}\n")
        f.write(f"Accuracy: {acc_pct:.2f}%  (95% CI: ±{ci*100:.2f}%)\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n")
        f.write(f"Weighted-F1: {weighted_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
        f.write("Classification Report:\n" + report)

    print(f"\n✅ Accuracy: {acc_pct:.2f}%  CI: ±{ci*100:.2f}%  Macro-F1: {macro_f1:.4f}")

    # Confusion matrix figure
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"FFSwin depth_patch={DEPTH_PATCH}\nAcc={acc_pct:.2f}%")
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_confusion.png"), dpi=300)
    plt.close()

    # ROC curves
    y_bin = label_binarize(all_labels, classes=[0,1,2])
    plt.figure(figsize=(10,8))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:,i], all_probs[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_roc.png"), dpi=300)
    plt.close()

    # t-SNE
    if len(all_features) > 2:
        perplexity = min(20, len(all_features)-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_res = tsne.fit_transform(all_features)
        plt.figure(figsize=(10,8))
        for i, name in enumerate(class_names):
            idx = all_labels == i
            plt.scatter(tsne_res[idx,0], tsne_res[idx,1], s=120, label=name, alpha=0.7)
        plt.title(f"t-SNE depth_patch={DEPTH_PATCH}")
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_tsne.png"), dpi=300)
        plt.close()

    print(f"All results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()