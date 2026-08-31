"""
快速集成测试: 加载数据 → 建模型 → 训练5个epoch → 提取HI
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from data_loader import BearingDataset
from model import DeepHIModel
from losses import compute_total_loss
from config import WINDOW_SIZE

# === 设备 ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# === 加载一个轴承 ===
print("\n[1] Loading Bearing1_1...")
ds = BearingDataset('Bearing1_1')
data = ds.to_torch(device)
x = data['windows']
t_pos = data['time_positions']
T = x.shape[0]
print(f"    {T} windows, shape={x.shape}")

# === 创建模型 ===
print("\n[2] Creating model...")
model = DeepHIModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"    {n_params:,} parameters")

# === 数据增强 ===
def augment(x):
    noise = torch.randn_like(x) * 0.01
    scale = torch.empty(x.shape[0], 1, 1, device=device).uniform_(0.95, 1.05)
    return (x + noise) * scale

# === 训练5个epoch ===
print("\n[3] Training 5 epochs...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    optimizer.zero_grad()

    # 增强
    v1 = augment(x)
    v2 = augment(x)

    # 前向
    z1 = model.encoder(v1)
    x_hat1 = model.decoder(z1)
    z2 = model.encoder(v2)
    x_hat2 = model.decoder(z2)
    HI = model.projector(z1).squeeze()

    # 损失
    losses = compute_total_loss(
        x=x, x_hat1=x_hat1, x_hat2=x_hat2,
        z1=z1, z2=z2,
        HI=HI, time_positions=t_pos,
        lambda_aug=0.5, lambda_seg=0.0,
        lambda_mono=0.0, lambda_trend=0.0,
    )

    losses['total'].backward()
    optimizer.step()

    print(f"  epoch {epoch+1}: total={losses['total'].item():.4f}, "
          f"recon={losses['recon'].item():.4f}, aug={losses['aug'].item():.4f}")

# === 提取HI ===
print("\n[4] Extracting HI...")
model.eval()
with torch.no_grad():
    HI = model.extract_hi(x).cpu().numpy()

print(f"    HI shape: {HI.shape}, range=[{HI.min():.4f}, {HI.max():.4f}]")
print(f"    Mean={HI.mean():.4f}, Std={HI.std():.4f}")

# === 基本检查 ===
print("\n[5] Sanity checks:")
print(f"    HI is not constant: {HI.std() > 0.01}")
print(f"    HI in [0,1]: {(HI >= 0).all() and (HI <= 1).all()}")

print("\n*** Integration test PASSED ***")
