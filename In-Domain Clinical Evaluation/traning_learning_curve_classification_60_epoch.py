import matplotlib.pyplot as plt
import numpy as np
import re
from math import sqrt

# ============================================================
# SETTINGS
# ============================================================
log_file = r"checkpoints\training_log.txt"
val_size = 64  # number of validation patients

epoch_clinical = 55
epoch_best = 58

plt.rcParams["font.family"] = "Palatino Linotype"
plt.rcParams["axes.labelsize"] = 28
plt.rcParams["axes.titlesize"] = 28
plt.rcParams["xtick.labelsize"] = 28
plt.rcParams["ytick.labelsize"] = 28
plt.rcParams["legend.fontsize"] = 20

# ============================================================
# PARSE LOG FILE
# ============================================================
epochs = []
train_acc = []
val_acc = []
train_loss = []

with open(log_file, "r") as f:
    for line in f:
        if "Epoch [" in line:
            match = re.search(
                r"Epoch \[(\d+)/\d+\] \| Loss ([\d.]+) \| TrainAcc ([\d.]+)% \| ValAcc ([\d.]+)%",
                line
            )
            if match:
                epochs.append(int(match.group(1)))
                train_loss.append(float(match.group(2)))
                train_acc.append(float(match.group(3)))
                val_acc.append(float(match.group(4)))

epochs = np.array(epochs)
train_acc = np.array(train_acc)
val_acc = np.array(val_acc)
train_loss = np.array(train_loss)

# ============================================================
# WILSON CONFIDENCE INTERVAL FUNCTION
# ============================================================
def wilson_ci(p, n, z=1.96):
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    margin = (z * sqrt((p*(1-p)/n) + (z**2/(4*n**2)))) / denominator
    return center - margin, center + margin

# Convert accuracy (%) to proportion
val_prop = val_acc / 100.0

ci_lower = []
ci_upper = []

for p in val_prop:
    low, high = wilson_ci(p, val_size)
    ci_lower.append(low * 100)
    ci_upper.append(high * 100)

ci_lower = np.array(ci_lower)
ci_upper = np.array(ci_upper)

# ============================================================
# ACCURACY FIGURE WITH TRUE CI
# ============================================================
plt.figure(figsize=(10, 7))

plt.plot(epochs, train_acc, linewidth=3, label="Train Accuracy")
plt.plot(epochs, val_acc, linewidth=3, label="Validation Accuracy")

plt.fill_between(
    epochs,
    ci_lower,
    ci_upper,
    alpha=0.2,
    label="95% CI (Validation)"
)

# Clinical epoch marker
plt.scatter(epoch_clinical,
            val_acc[epoch_clinical - 1],
            s=250,
            marker='D',
            color='green',
            label=f"Clinical Epoch ({epoch_clinical})")

# Best epoch marker
plt.scatter(epoch_best,
            val_acc[epoch_best - 1],
            s=250,
            marker='o',
            color='red',
            label=f"Peak Accuracy ({epoch_best})")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("Accuracy_with_True_CI.png", dpi=600)
plt.show()

# ============================================================
# LOSS CURVE
# ============================================================
plt.figure(figsize=(10, 7))

plt.plot(epochs, train_loss, linewidth=3, label="Training Loss")

plt.scatter(epoch_clinical,
            train_loss[epoch_clinical - 1],
            s=250,
            marker='D',
            color='green',
            label=f"Clinical Epoch ({epoch_clinical})")

plt.scatter(epoch_best,
            train_loss[epoch_best - 1],
            s=250,
            marker='o',
            color='red',
            label=f"Peak Accuracy ({epoch_best})")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("Loss_with_ClinicalMarkers.png", dpi=600)
plt.show()

print("✅ Figures saved with true Wilson 95% CI.")
