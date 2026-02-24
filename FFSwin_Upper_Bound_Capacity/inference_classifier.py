import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. FIXED MODEL ARCHITECTURE (Matches Saved Weights)
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
        
        # --- THIS WAS MISSING IN THE PREVIOUS CELL ---
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)) 
        # ---------------------------------------------

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
    def __init__(self, num_classes=3, in_chans=1, embed_dim=96, patch_size=4):
        super().__init__()
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
        return self.head(x)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def preprocess_patient(folder_path, img_size=256):
    files = sorted(glob.glob(os.path.join(folder_path, "*.bmp")))
    if len(files) == 0:
        print(f"❌ No images found in {folder_path}")
        return None
    vol = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            vol.append(img)
    if len(vol) == 0: return None
    if len(vol) % 2 != 0: vol.append(vol[-1])
    tensor = torch.FloatTensor(np.array(vol) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor

def predict_patient(patient_folder, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = preprocess_patient(patient_folder)
    if input_tensor is None: return

    # Load Model
    model = FFSwinClassifier(num_classes=3).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return
    model.eval()

    # Inference
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        
    classes = ['Dry AMD', 'Wet AMD', 'Non-AMD (Healthy)']
    prediction = classes[pred_idx.item()]
    
    print("-" * 40)
    print(f"📂 Patient: {os.path.basename(patient_folder)}")
    print(f"🩺 DIAGNOSIS: {prediction}")
    print(f"📊 Confidence: {confidence.item()*100:.2f}%")
    print("-" * 40)
    print(f"Probability Breakdown:")
    print(f" - Dry AMD: {probs[0][0]*100:.2f}%")
    print(f" - Wet AMD: {probs[0][1]*100:.2f}%")
    print(f" - Healthy: {probs[0][2]*100:.2f}%")
    print("-" * 40)
    
    mid_slice = input_tensor.cpu().numpy()[0, 0, input_tensor.shape[2]//2, :, :]
    plt.imshow(mid_slice, cmap='gray')
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show()

# ==========================================
# 3. RUN TEST
# ==========================================
model_path = "FFSwin_Upper_Bound_Capacity\models\classifier_best.pth"
data_root = "G:\\My Drive\\PhD_BanglaOCT2025_data\\BanglaOCT2025\\ANONYMIZED_DATA_Macula_33_images_Denoised"

test_category = 'NonAMD' 
category_path = os.path.join(data_root, test_category)

if os.path.exists(category_path):
    subfolders = [f.path for f in os.scandir(category_path) if f.is_dir()]
    if subfolders:
        random_patient = random.choice(subfolders)
        print(f"🧪 Testing on a random REAL '{test_category}' patient...")
        predict_patient(random_patient, model_path)
    else:
        print("No subfolders found.")
else:
    print("Data path not found.")