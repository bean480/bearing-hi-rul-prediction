"""HI visualization: train + extract + plot. CUDA_VISIBLE_DEVICES=1 python viz_hi.py"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import load_all_bearings
from config import BEARINGS_BY_CONDITION, SG_WINDOW
from train import train_model

def _smooth(HI, w=11):
    if len(HI) < w: return HI.copy()
    return np.convolve(HI, np.ones(w)/w, mode='same')

def _pearson(x, y):
    xc, yc = x-x.mean(), y-y.mean()
    return (xc*yc).mean() / (xc.std()*yc.std() + 1e-10)

print("[1] Training 100 epochs...", flush=True)
all_ds = load_all_bearings()
b = BEARINGS_BY_CONDITION[1]
train_ds = {n: all_ds[n] for n in b[1:]}
test_name = b[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = train_model(train_ds, epochs=100, ablation=None, device=device, verbose=True)

print("[2] Extracting HI...", flush=True)
model.eval()
test_ds = all_ds[test_name]
data = test_ds.to_torch(device)
with torch.no_grad():
    HI = model.extract_hi(data['windows']).cpu().numpy()

T = len(HI); t = np.arange(T)
smooth = _smooth(HI, SG_WINDOW*2-1) if T>=SG_WINDOW else HI
up = (np.diff(HI)>0).sum(); down = (np.diff(HI)<0).sum()
trend_val = max(0.0, (_pearson(HI, np.linspace(0,1,T))+1)/2)
print(f"HI: [{HI.min():.4f}, {HI.max():.4f}]  Mon={up/(T-1):.4f}  Trend={trend_val:.4f}")

fig, axes = plt.subplots(3, 1, figsize=(16, 10))
fig.suptitle(f'Deep HI - {test_name} (100 epochs + anchor)', fontsize=14)

axes[0].plot(t, HI, alpha=0.3, lw=0.3, color='steelblue'); axes[0].plot(t, smooth, 'r-', lw=1.5)
axes[0].set_ylabel('HI'); axes[0].set_ylim(-0.05, 1.05); axes[0].legend(['Raw','Smoothed'])
axes[0].grid(True, alpha=0.3)
axes[0].text(0.02, 0.95, f'Range=[{HI.min():.3f},{HI.max():.3f}] Mon={up/(T-1):.3f} Trend={trend_val:.3f}',
             transform=axes[0].transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat'))

k = min(500, T)
axes[1].plot(t[:k], HI[:k], lw=0.8, color='steelblue', alpha=0.7)
axes[1].plot(t[:k], smooth[:k], 'r-', lw=1.5)
axes[1].set_ylabel('HI'); axes[1].set_xlabel('Window')
axes[1].set_title(f'Zoom: First {k} windows'); axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(True, alpha=0.3)

axes[2].hist(HI, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
axes[2].set_xlabel('HI Value'); axes[2].set_ylabel('Count')
axes[2].set_title(f'Distribution (mean={HI.mean():.3f}, std={HI.std():.3f})')
axes[2].axvline(HI.mean(), color='red', ls='--', lw=2, label=f'Mean={HI.mean():.3f}')
axes[2].axvline(np.median(HI), color='orange', ls='--', lw=2, label=f'Median={np.median(HI):.3f}')
axes[2].legend()

plt.tight_layout()
fname = 'hi_vis.png'
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()

# Normalize timestamp to avoid SFTP SSH2 "isDate is not a function" crash
now = time.time()
os.utime(fname, (now, now))
print(f"[3] Saved: {fname} (timestamp fixed)")

# Workaround for SFTP timestamp bug: compress to .gz
import gzip, shutil
with open(fname, 'rb') as f_in:
    with gzip.open(fname + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
os.utime(fname + '.gz', (now, now))
print(f"[5] Compressed: {fname}.gz (download this instead)")
