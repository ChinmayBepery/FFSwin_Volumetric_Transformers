import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
import glob
import cv2

# ==========================================
# 0. FONT CONFIGURATION (Palatino Linotype)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino Linotype']
plt.rcParams['font.size'] = 12

# ==========================================
# 1. DEFINE THE MODEL ARCHITECTURE
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
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
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
        self.dim = dim; self.window_size = window_size; self.num_heads = num_heads
        head_dim = dim // num_heads; self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        # --- FIX: Re-added this parameter to match saved weights ---
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

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
        self.window_size = window_size; self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim); self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim); self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4.))

    def forward(self, x):
        B, D, H, W, C = x.shape; shortcut = x
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
        x = shortcut + x; x = x + self.mlp(self.norm2(x))
        return x

class FFSwinClassifier(nn.Module):
    def __init__(self, num_classes=3, in_chans=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
        self.layers = nn.Sequential(
            SwinTransformerBlock3D(embed_dim, num_heads=3, shift_size=(0,0,0)), 
            SwinTransformerBlock3D(embed_dim, num_heads=3, shift_size=(1,2,2)), 
            SwinTransformerBlock3D(embed_dim, num_heads=3, shift_size=(0,0,0)), 
            SwinTransformerBlock3D(embed_dim, num_heads=3, shift_size=(1,2,2))  
        )
        self.norm = nn.LayerNorm(embed_dim); self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 4, 1) 
        x = self.layers(x); x = self.norm(x)
        B, D, H, W, C = x.shape
        x = x.view(B, -1, C).transpose(1, 2) 
        x = self.avgpool(x).flatten(1)       
        return self.head(x)

# ==========================================
# 2. DATASET HELPER
# ==========================================
class SimpleOCTDataset(Dataset):
    def __init__(self, root_dir, img_size=256, return_label=False):
        self.img_size = img_size
        self.return_label = return_label
        self.folders = []
        self.labels = []
        label_map = {'DryAMD': 0, 'WetAMD': 1, 'NonAMD': 2}
        for cat in ['DryAMD', 'WetAMD', 'NonAMD']:
            p = os.path.join(root_dir, cat)
            if os.path.exists(p):
                subs = [f.path for f in os.scandir(p) if f.is_dir()]
                self.folders.extend(subs)
                self.labels.extend([label_map[cat]] * len(subs))

    def __len__(self): return len(self.folders)
    
    def __getitem__(self, idx):
        vol = self._load(self.folders[idx])
        if self.return_label: return vol, self.labels[idx]
        return vol, self.folders[idx]

    def _load(self, path):
        files = sorted(glob.glob(os.path.join(path, "*.bmp")))
        vol = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None: vol.append(cv2.resize(img, (self.img_size, self.img_size)))
        if not vol: return torch.zeros(1, 1, 16, self.img_size, self.img_size)
        if len(vol) % 2 != 0: vol.append(vol[-1]) 
        return torch.FloatTensor(np.array(vol).astype(np.float32)/255.0).unsqueeze(0)

# ==========================================
# 3. RUN EVALUATION (COMPACT VISUALS)
# ==========================================
def evaluate_final_model():
    print("📊 Evaluating Final Model Results...Clean Data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------- CONFIGURATION ----------------
    #data_dir = r"D:\OCTData\BanglaOCT2025_Dataset\clean_data" 
    data_dir = r"G:\My Drive\OCT_Project_PhD_Implementation\3_AROI Dataset\clean_33_slices" 
    model_path = r"models\classifier_best_100_used_here_as_final.pth"
    # -----------------------------------------------

    if not os.path.exists(data_dir):
        print(f"❌ Error: {data_dir} not found.")
        return
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    # Load Data & Model
    ds = SimpleOCTDataset(data_dir, return_label=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    print(f"✅ Loading weights from: {model_path}")
    model = FFSwinClassifier(num_classes=3).to(device)
    
    # Safe Load
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"\n❌ LOAD ERROR: {e}"); return

    model.eval()
    
    all_preds = []; all_labels = []
    print(f"   Testing on {len(ds)} patients...")
    with torch.no_grad():
        for vol, label in dl:
            vol = vol.to(device)
            out = model(vol)
            _, pred = torch.max(out, 1)
            all_preds.append(pred.item())
            all_labels.append(label.item())

    # Metrics
    class_names = ['DryAMD', 'WetAMD', 'NonAMD']
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    accuracy = np.trace(cm) / np.sum(cm) * 100
    print(f"\n🏆 Final Accuracy: {accuracy:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ---------------- PLOTTING FOR QUARTER PAGE A4 ----------------
    # figsize=(6, 5) inches is approx quarter A4 width with margins
    plt.figure(figsize=(6, 5)) 
    
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=class_names, yticklabels=class_names,
                     # Size 20 Bold for the numbers inside the boxes
                     annot_kws={"size": 20, "weight": "bold", "family": "Palatino Linotype"}, 
                     cbar=False) # Turned off colorbar to save space for larger fonts
    
    # Large Axis Labels
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold', family='Palatino Linotype')
    plt.ylabel('True Class', fontsize=14, fontweight='bold', family='Palatino Linotype')
    plt.title(f'Confusion Matrix (Acc: {accuracy:.1f}%)', fontsize=16, fontweight='bold', family='Palatino Linotype', pad=15)
    
    # Large Ticks
    plt.xticks(fontsize=13, family='Palatino Linotype')
    plt.yticks(fontsize=13, family='Palatino Linotype', rotation=0)

    plt.tight_layout()
    plt.savefig('Confusion_Matrix_Compact.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Compact plot saved as 'Confusion_Matrix_Compact.png'")

if __name__ == '__main__':
    evaluate_final_model()