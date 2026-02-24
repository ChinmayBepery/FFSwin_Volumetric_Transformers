import torch
import torch.nn as nn
import time
import os
import numpy as np
from thop import profile # The library for calculating FLOPs

# ==========================================
# 1. DEFINE YOUR FFSWIN ARCHITECTURE
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
# 2. BENCHMARKING FUNCTION
# ==========================================
def benchmark_model():
    print("⚡ STARTING COMPUTATIONAL EFFICIENCY BENCHMARK...")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 1. Instantiate Model
    model = FFSwinClassifier(num_classes=3).to(device)
    model.eval()
    
    # 2. Define Dummy Input (Batch=1, Channel=1, Depth=16, Height=256, Width=256)
    dummy_input = torch.randn(1, 1, 16, 256, 256).to(device)
    
    # ---------------------------------------------------------
    # A. FLOPs & Parameters
    # ---------------------------------------------------------
    print("\n[1] Calculating Complexity (FLOPs)...")
    # thop calculates MACs (Multiply-Accumulate Operations)
    # 1 MAC ≈ 2 FLOPs
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2
    
    print(f"✅ Total Parameters: {params / 1e6:.2f} Million")
    print(f"✅ GFLOPs:           {flops / 1e9:.2f} G (Billions of operations)")
    
    # ---------------------------------------------------------
    # B. Model Size (Storage)
    # ---------------------------------------------------------
    print("\n[2] Calculating Storage Size...")
    # Save temp model to check file size
    torch.save(model.state_dict(), "temp_model_size.pth")
    size_mb = os.path.getsize("temp_model_size.pth") / (1024 * 1024)
    os.remove("temp_model_size.pth")
    print(f"✅ Model File Size:  {size_mb:.2f} MB")
    
    # ---------------------------------------------------------
    # C. Inference Time & Throughput (with CUDA Events)
    # ---------------------------------------------------------
    print("\n[3] Measuring Latency (Inference Time)...")
    
    # Warm-up (Critical for GPU to reach clock speed)
    print("   Running warm-up...")
    for _ in range(20):
        _ = model(dummy_input)
    
    # Measurement Loop
    iterations = 100
    print(f"   Averaging over {iterations} runs...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    end_event.record()
    
    torch.cuda.synchronize() # Wait for GPU to finish
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / iterations
    throughput = 1000 / avg_latency_ms # Samples per second
    
    print(f"✅ Latency:          {avg_latency_ms:.2f} ms per patient")
    print(f"✅ Throughput:       {throughput:.2f} patients per second")
    
    print("\n" + "="*50)
    print("🏁 BENCHMARK COMPLETE")

if __name__ == "__main__":
    benchmark_model()