import torch
import torch.nn as nn
import os
import glob
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             f1_score, precision_score, recall_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from itertools import cycle

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, D, H, W):
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)) 
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2,4,4), shift_size=(0,0,0)):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4.))
    def forward(self, x):
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if sum(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else: shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)
        if sum(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else: x = shifted_x
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class FFSwinClassifier(nn.Module):
    def __init__(self, num_classes=3, in_chans=1, embed_dim=96, patch_size=4, return_features=False):
        super().__init__()
        self.return_features = return_features
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
        self.layers = nn.Sequential(
            SwinTransformerBlock3D(embed_dim, num_heads=3, shift_size=(0,0,0)), 
            SwinTransformerBlock3D(embed_dim, num_heads=3, shift_size=(1,2,2)), 
            SwinTransformerBlock3D(embed_dim, num_heads=3, shift_size=(0,0,0)), 
            SwinTransformerBlock3D(embed_dim, num_heads=3, shift_size=(1,2,2))  
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 4, 1) 
        x = self.layers(x)
        x = self.norm(x)
        B, D, H, W, C = x.shape
        x = x.view(B, -1, C).transpose(1, 2) 
        x = self.avgpool(x).flatten(1)  
        
        logits = self.head(x)
        if self.return_features:
            return logits, x
        return logits

# ==========================================
# 2. DATASET
# ==========================================
class SimpleOCTDataset(Dataset):
    def __init__(self, root_dir, img_size=256, return_label=False):
        self.img_size = img_size
        self.return_label = return_label
        self.folders = []
        self.labels = []
        label_map = {'DryAMD': 0, 'WetAMD': 1, 'NonAMD': 2}
        target_folders = ['DryAMD', 'WetAMD', 'NonAMD']
        
        for cat in target_folders:
            p = os.path.join(root_dir, cat)
            if os.path.exists(p):
                subs = [f.path for f in os.scandir(p) if f.is_dir()]
                self.folders.extend(subs)
                self.labels.extend([label_map[cat]] * len(subs))

    def __len__(self): return len(self.folders)
    
    def __getitem__(self, idx):
        vol = self._load(self.folders[idx])
        if self.return_label:
            return vol, self.labels[idx]
        return vol, self.folders[idx]

    def _load(self, path):
        files = sorted(glob.glob(os.path.join(path, "*.bmp")))
        vol = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None: vol.append(cv2.resize(img, (self.img_size, self.img_size)))
        if not vol: return torch.zeros(1, 1, 16, self.img_size, self.img_size)
        if len(vol) % 2 != 0: vol.append(vol[-1]) 
        return torch.FloatTensor(np.array(vol)/255.0).unsqueeze(0)

# ==========================================
# 3. HELPER: SPECIFICITY
# ==========================================
def calculate_specificity(cm):
    specificities = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        fp = np.sum(np.delete(cm, i, 0)[:, i])
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)
    return specificities

# ==========================================
# 4. MASTER EVALUATION
# ==========================================
def run_master_thesis_evaluation():
    print("🎓 INITIALIZING MASTER THESIS EVALUATION...")
    print("="*50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. PATHS 
    data_dir = "G:\My Drive\OCT_Project_PhD_Implementation\BanglaOCT2025_Dataset\local_data_clean"
    best_path = "G:\My Drive\OCT_Project_PhD_Implementation\FFSwin_Pseudo_3D_Model_main\models\classifier_best_100_used_here_as_final.pth"
    final_path = "G:\My Drive\OCT_Project_PhD_Implementation\FFSwin_Pseudo_3D_Model_main\models\classifier_best_100_used_here_as_final.pth"
   
    model_path = best_path if os.path.exists(best_path) else final_path
    
    if not os.path.exists(model_path):
        print("❌ Error: No model file found.")
        return

    # Load Data & Model
    ds = SimpleOCTDataset(data_dir, return_label=True)
    if len(ds) == 0:
        print("❌ Error: Dataset empty.")
        return
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    model = FFSwinClassifier(num_classes=3, return_features=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Run Inference
    y_true = []
    y_pred = []
    y_probs = [] 
    features_list = []
    
    print(f"⏳ Processing {len(ds)} patients...")
    
    start_time = time.time()
    with torch.no_grad():
        for vol, label in dl:
            vol = vol.to(device)
            logits, feats = model(vol)
            probs = torch.softmax(logits, dim=1)
            _, pred = torch.max(logits, 1)
            
            y_pred.append(pred.item())
            y_true.append(label.item())
            y_probs.append(probs.cpu().numpy()[0])
            features_list.append(feats.cpu().numpy()[0])
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_inference_time = (total_time / len(ds)) * 1000 
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    features_list = np.array(features_list)
    
    # --- REPORT 1: COMPUTATIONAL STATS ---
    params = sum(p.numel() for p in model.parameters())
    file_size = os.path.getsize(model_path) / (1024*1024)
    print("\n" + "="*50)
    print("⚡ 1. COMPUTATIONAL EFFICIENCY")
    print("="*50)
    print(f"🔹 Total Parameters:   {params:,}")
    print(f"🔹 Model File Size:    {file_size:.2f} MB")
    print(f"🔹 Avg Inference Time: {avg_inference_time:.2f} ms / patient")

    # --- REPORT 2: METRICS ---
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    specs = calculate_specificity(cm)
    
    print("\n" + "="*50)
    print("📊 2. CLINICAL METRICS")
    print("="*50)
    print(f"🔹 Accuracy:          {acc*100:.2f}%")
    print(f"🔹 Macro F1:          {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"🔹 Weighted F1:       {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    classes = ['DryAMD', 'WetAMD', 'NonAMD']
    print("-" * 50)
    print(f"{'Class':<10} | {'Prec':<10} | {'Recall':<10} | {'Spec':<10}")
    print("-" * 50)
    
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, cls in enumerate(classes):
        print(f"{cls:<10} | {prec[i]:.4f}     | {rec[i]:.4f}     | {specs[i]:.4f}")
    
    print("\n[Confusion Matrix]")
    print(cm)

    # --- REPORT 3: ROC CURVES ---
    print("\nGenerating ROC Curves...")
    y_test_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    fpr = dict(); tpr = dict(); roc_auc = dict()
    
    plt.figure(figsize=(10, 6))
    colors = cycle(['orange', 'green', 'blue'])
    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves_thesis.png', dpi=300)
    print("✅ Saved 'roc_curves_thesis.png'")

    # --- REPORT 4: t-SNE ---
    print("\nGenerating t-SNE (Features)...")
    # FIX: Removed 'n_iter=1000' to assume default and avoid version conflicts
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(features_list)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, title="Classes")
    plt.title("t-SNE Feature Visualization")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.3)
    plt.savefig('tsne_plot_thesis.png', dpi=300)
    print("✅ Saved 'tsne_plot_thesis.png'")
    print("\n🎉 ALL DONE.")

if __name__ == "__main__":
    run_master_thesis_evaluation()