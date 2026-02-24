import torch
import torch.nn as nn
import os
import numpy as np

# ==========================================
# 1. DEFINE ARCHITECTURE (Must match saved model)
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
# 2. ANALYSIS FUNCTION
# ==========================================
def inspect_model_parameters(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔍 Analyzing Model: {os.path.basename(model_path)} ...")
    
    if not os.path.exists(model_path):
        print("❌ File not found.")
        return

    # Initialize Model
    model = FFSwinClassifier(num_classes=3)
    
    # Load Weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 1. Count Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Check File Size
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print("\n" + "="*40)
    print("📊 MODEL PARAMETER REPORT")
    print("="*40)
    print(f"🔹 Total Parameters:     {total_params:,}")
    print(f"🔹 Trainable Parameters: {trainable_params:,}")
    print(f"🔹 File Size on Disk:    {file_size_mb:.2f} MB")
    print("-" * 40)
    
    # 3. Layer Breakdown (Optional: First 5 and Last 5)
    print("\n🔹 Layer-by-Layer Breakdown (Summary):")
    params = list(model.named_parameters())
    
    print(f"{'Layer Name':<40} | {'Shape':<20} | {'Params'}")
    print("-" * 75)
    
    # Print logic: Show patch embed, then summarized blocks, then head
    for name, param in params:
        # Just printing the key layers to avoid a huge list
        if 'patch_embed' in name or 'head' in name or 'norm' in name:
            print(f"{name:<40} | {str(list(param.shape)):<20} | {param.numel():,}")
        elif 'layers.0.attn.qkv.weight' in name:
            print(f"{'... (Inside Swin Blocks) ...':<40} | {'...':<20} | ...")

# ==========================================
# 3. RUN ANALYSIS
# ==========================================
path = "G:\My Drive\OCT_Project\models\classifier_final.pth"
inspect_model_parameters(path)