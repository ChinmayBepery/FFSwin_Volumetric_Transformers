import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ======================= IEEE TMI FONT =======================
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Palatino Linotype"]
rcParams["font.size"] = 11
rcParams["axes.titlesize"] = 12
rcParams["axes.labelsize"] = 11
rcParams["legend.fontsize"] = 10
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10

# ======================= READ LOG FILE =======================
log_text ="""
Ep 1/100 | Loss: 1.1088 | Train Acc: 36.2% | Val Acc: 71.5%
Ep 2/100 | Loss: 1.0934 | Train Acc: 37.7% | Val Acc: 78.5%
Ep 3/100 | Loss: 1.0857 | Train Acc: 40.7% | Val Acc: 7.8%
Ep 4/100 | Loss: 1.0721 | Train Acc: 41.2% | Val Acc: 51.0%
Ep 5/100 | Loss: 1.0549 | Train Acc: 44.5% | Val Acc: 70.8%
Ep 6/100 | Loss: 1.0189 | Train Acc: 48.3% | Val Acc: 58.5%
Ep 7/100 | Loss: 0.9791 | Train Acc: 52.5% | Val Acc: 57.2%
Ep 8/100 | Loss: 0.9305 | Train Acc: 57.1% | Val Acc: 53.7%
Ep 9/100 | Loss: 0.9032 | Train Acc: 58.1% | Val Acc: 44.9%
Ep 10/100 | Loss: 0.8602 | Train Acc: 60.7% | Val Acc: 41.7%
Ep 11/100 | Loss: 0.8387 | Train Acc: 61.7% | Val Acc: 61.3%
Ep 12/100 | Loss: 0.8285 | Train Acc: 64.2% | Val Acc: 72.6%
Ep 13/100 | Loss: 0.8046 | Train Acc: 66.6% | Val Acc: 55.1%
Ep 14/100 | Loss: 0.7578 | Train Acc: 68.6% | Val Acc: 53.3%
Ep 15/100 | Loss: 0.7377 | Train Acc: 71.0% | Val Acc: 74.9%
Ep 16/100 | Loss: 0.7101 | Train Acc: 72.1% | Val Acc: 65.8%
Ep 17/100 | Loss: 0.6840 | Train Acc: 73.4% | Val Acc: 83.0%
Ep 18/100 | Loss: 0.6647 | Train Acc: 74.3% | Val Acc: 75.7%
Ep 19/100 | Loss: 0.6429 | Train Acc: 75.7% | Val Acc: 67.8%
Ep 20/100 | Loss: 0.6121 | Train Acc: 76.7% | Val Acc: 72.5%
Ep 21/100 | Loss: 0.5951 | Train Acc: 78.7% | Val Acc: 81.9%
Ep 22/100 | Loss: 0.5694 | Train Acc: 79.5% | Val Acc: 76.9%
Ep 23/100 | Loss: 0.5530 | Train Acc: 79.6% | Val Acc: 82.0%
Ep 24/100 | Loss: 0.5241 | Train Acc: 81.2% | Val Acc: 81.3%
Ep 25/100 | Loss: 0.5139 | Train Acc: 82.1% | Val Acc: 76.7%
Ep 26/100 | Loss: 0.4667 | Train Acc: 83.2% | Val Acc: 54.4%
Ep 27/100 | Loss: 0.4614 | Train Acc: 84.0% | Val Acc: 74.7%
Ep 28/100 | Loss: 0.4448 | Train Acc: 83.8% | Val Acc: 78.8%
Ep 29/100 | Loss: 0.4207 | Train Acc: 85.4% | Val Acc: 87.0%
Ep 30/100 | Loss: 0.4087 | Train Acc: 86.2% | Val Acc: 89.0%
Ep 31/100 | Loss: 0.3733 | Train Acc: 87.5% | Val Acc: 91.9%
Ep 32/100 | Loss: 0.3517 | Train Acc: 88.7% | Val Acc: 81.6%
Ep 33/100 | Loss: 0.3531 | Train Acc: 88.9% | Val Acc: 89.1%
Ep 34/100 | Loss: 0.3207 | Train Acc: 90.0% | Val Acc: 76.2%
Ep 35/100 | Loss: 0.3147 | Train Acc: 90.1% | Val Acc: 87.6%
Ep 36/100 | Loss: 0.2819 | Train Acc: 92.4% | Val Acc: 92.6%
Ep 37/100 | Loss: 0.2776 | Train Acc: 92.6% | Val Acc: 89.7%
Ep 38/100 | Loss: 0.2596 | Train Acc: 92.7% | Val Acc: 92.2%
Ep 39/100 | Loss: 0.2490 | Train Acc: 92.6% | Val Acc: 82.8%
Ep 40/100 | Loss: 0.2387 | Train Acc: 93.8% | Val Acc: 90.7%
Ep 41/100 | Loss: 0.2303 | Train Acc: 93.8% | Val Acc: 92.9%
Ep 42/100 | Loss: 0.2115 | Train Acc: 94.4% | Val Acc: 95.6%
Ep 43/100 | Loss: 0.2044 | Train Acc: 93.8% | Val Acc: 94.9%
Ep 44/100 | Loss: 0.1783 | Train Acc: 95.9% | Val Acc: 93.2%
Ep 45/100 | Loss: 0.1813 | Train Acc: 95.4% | Val Acc: 90.5%
Ep 46/100 | Loss: 0.1701 | Train Acc: 95.4% | Val Acc: 88.3%
Ep 47/100 | Loss: 0.1744 | Train Acc: 95.3% | Val Acc: 97.1%
Ep 48/100 | Loss: 0.1465 | Train Acc: 96.5% | Val Acc: 95.1%
Ep 49/100 | Loss: 0.1440 | Train Acc: 96.0% | Val Acc: 95.1%
Ep 50/100 | Loss: 0.1636 | Train Acc: 95.4% | Val Acc: 96.5%
Ep 51/100 | Loss: 0.1234 | Train Acc: 97.1% | Val Acc: 97.9%
Ep 52/100 | Loss: 0.1255 | Train Acc: 96.6% | Val Acc: 98.0%
Ep 53/100 | Loss: 0.1149 | Train Acc: 97.2% | Val Acc: 98.5%
Ep 54/100 | Loss: 0.1068 | Train Acc: 97.2% | Val Acc: 97.2%
Ep 55/100 | Loss: 0.0904 | Train Acc: 98.0% | Val Acc: 99.1%
Ep 56/100 | Loss: 0.1581 | Train Acc: 96.6% | Val Acc: 98.6%
Ep 57/100 | Loss: 0.0869 | Train Acc: 98.2% | Val Acc: 99.2%
Ep 58/100 | Loss: 0.0957 | Train Acc: 97.7% | Val Acc: 94.7%
Ep 59/100 | Loss: 0.1091 | Train Acc: 97.5% | Val Acc: 97.0%
Ep 60/100 | Loss: 0.0710 | Train Acc: 98.6% | Val Acc: 99.1%
Ep 61/100 | Loss: 0.0800 | Train Acc: 97.9% | Val Acc: 98.6%
Ep 62/100 | Loss: 0.0673 | Train Acc: 98.7% | Val Acc: 97.3%
Ep 63/100 | Loss: 0.0686 | Train Acc: 98.6% | Val Acc: 99.1%
Ep 64/100 | Loss: 0.0799 | Train Acc: 98.0% | Val Acc: 99.4%
Ep 65/100 | Loss: 0.0874 | Train Acc: 97.9% | Val Acc: 96.1%
Ep 66/100 | Loss: 0.0578 | Train Acc: 98.8% | Val Acc: 98.9%
Ep 67/100 | Loss: 0.0467 | Train Acc: 99.0% | Val Acc: 99.4%
Ep 68/100 | Loss: 0.0547 | Train Acc: 98.9% | Val Acc: 99.8%
Ep 69/100 | Loss: 0.0613 | Train Acc: 98.4% | Val Acc: 99.6%
Ep 70/100 | Loss: 0.0522 | Train Acc: 98.7% | Val Acc: 98.6%
Ep 71/100 | Loss: 0.0389 | Train Acc: 99.1% | Val Acc: 99.6%
Ep 72/100 | Loss: 0.0579 | Train Acc: 98.6% | Val Acc: 91.9%
Ep 73/100 | Loss: 0.1100 | Train Acc: 96.5% | Val Acc: 99.6%
Ep 74/100 | Loss: 0.0377 | Train Acc: 98.9% | Val Acc: 99.6%
Ep 75/100 | Loss: 0.0282 | Train Acc: 99.3% | Val Acc: 99.8%
Ep 76/100 | Loss: 0.0333 | Train Acc: 99.1% | Val Acc: 99.4%
Ep 77/100 | Loss: 0.0929 | Train Acc: 97.1% | Val Acc: 97.2%
Ep 78/100 | Loss: 0.0449 | Train Acc: 98.9% | Val Acc: 99.1%
Ep 79/100 | Loss: 0.0271 | Train Acc: 99.4% | Val Acc: 99.6%
Ep 80/100 | Loss: 0.0645 | Train Acc: 98.0% | Val Acc: 96.4%
Ep 81/100 | Loss: 0.0382 | Train Acc: 99.0% | Val Acc: 99.6%
Ep 82/100 | Loss: 0.0261 | Train Acc: 99.1% | Val Acc: 99.8%
Ep 83/100 | Loss: 0.0276 | Train Acc: 99.3% | Val Acc: 97.7%
Ep 84/100 | Loss: 0.0836 | Train Acc: 97.7% | Val Acc: 99.8%
Ep 85/100 | Loss: 0.0236 | Train Acc: 99.4% | Val Acc: 99.8%
Ep 86/100 | Loss: 0.0832 | Train Acc: 97.1% | Val Acc: 99.1%
Ep 87/100 | Loss: 0.0232 | Train Acc: 99.5% | Val Acc: 99.8%
Ep 88/100 | Loss: 0.0255 | Train Acc: 99.1% | Val Acc: 99.8%
Ep 89/100 | Loss: 0.0212 | Train Acc: 99.3% | Val Acc: 99.6%
Ep 90/100 | Loss: 0.0502 | Train Acc: 98.3% | Val Acc: 99.5%
Ep 91/100 | Loss: 0.0221 | Train Acc: 99.5% | Val Acc: 99.8%
Ep 92/100 | Loss: 0.0234 | Train Acc: 99.3% | Val Acc: 99.5%
Ep 93/100 | Loss: 0.0377 | Train Acc: 98.8% | Val Acc: 99.4%
Ep 94/100 | Loss: 0.0240 | Train Acc: 99.3% | Val Acc: 99.9%
Ep 95/100 | Loss: 0.0804 | Train Acc: 97.3% | Val Acc: 99.2%
Ep 96/100 | Loss: 0.0242 | Train Acc: 99.5% | Val Acc: 99.8%
Ep 97/100 | Loss: 0.0183 | Train Acc: 99.3% | Val Acc: 99.8%
Ep 98/100 | Loss: 0.0175 | Train Acc: 99.3% | Val Acc: 99.8%
Ep 99/100 | Loss: 0.1346 | Train Acc: 95.4% | Val Acc: 99.6%
Ep 100/100 | Loss: 0.0241 | Train Acc: 99.3% | Val Acc: 99.8%
"""

# ======================= PARSE EPOCH METRICS =======================
epochs, loss, train_acc, val_acc = [], [], [], []

# Updated regex to match the new format: "Ep 1/100 | Loss: 1.1088 | Train Acc: 36.2% | Val Acc: 71.5%"
epoch_re = re.compile(
    r"Ep\s+(\d+)/\d+\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Train Acc:\s*([\d.]+)%\s*\|\s*Val Acc:\s*([\d.]+)%"
)

for line in log_text.splitlines():
    m = epoch_re.search(line)
    if m:
        epochs.append(int(m.group(1)))
        loss.append(float(m.group(2)))
        train_acc.append(float(m.group(3)))
        val_acc.append(float(m.group(4)))

epochs = np.array(epochs)
train_acc = np.array(train_acc)
val_acc = np.array(val_acc)
loss = np.array(loss)

# ======================= BEST EPOCH =======================
best_idx = np.argmax(val_acc)
best_epoch = epochs[best_idx]
best_val = val_acc[best_idx]
best_loss = loss[best_idx]

# ======================= 95% CI (Bootstrap on validation) =======================
def bootstrap_ci(data, n=2000):
    means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

ci_lo, ci_hi = [], []
win = 5  # rolling window
for i in range(len(val_acc)):
    s = max(0, i-win)
    e = min(len(val_acc), i+win)
    lo, hi = bootstrap_ci(val_acc[s:e])
    ci_lo.append(lo)
    ci_hi.append(hi)

ci_lo = np.array(ci_lo)
ci_hi = np.array(ci_hi)

# ======================= FIGURE 1: TRAIN vs VAL ACC =======================
plt.figure(figsize=(6,4))
plt.plot(epochs, train_acc, linewidth=1.5, label="Train Accuracy")
plt.plot(epochs, val_acc, linewidth=1.8, label="Validation Accuracy")
plt.fill_between(epochs, ci_lo, ci_hi, alpha=0.2, color='orange', label="95% CI")
plt.scatter(best_epoch, best_val, s=60, color='red', zorder=5, 
            label=f"Best Epoch ({best_epoch}: {best_val:.1f}%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Fig1_Accuracy_CI.png", dpi=300)
plt.close()

# ======================= FIGURE 2: LOSS CURVE =======================
plt.figure(figsize=(6,4))
plt.plot(epochs, loss, color='blue', linewidth=1.8, label="Training Loss")
plt.scatter(best_epoch, best_loss, s=60, color='red', zorder=5, 
            label=f"Best Epoch ({best_epoch})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Fig2_Loss_Curve.png", dpi=300)
plt.close()

# Print some statistics
print(f"✅ Best validation accuracy: {best_val:.2f}% at epoch {best_epoch}")
print(f"✅ Training accuracy at best epoch: {train_acc[best_idx]:.2f}%")
print(f"✅ Loss at best epoch: {best_loss:.4f}")
print("✅ Figures generated: Fig1_Accuracy_CI.png and Fig2_Loss_Curve.png")