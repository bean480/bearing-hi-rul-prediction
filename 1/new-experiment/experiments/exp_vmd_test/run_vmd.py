"""实验一: VMD 分解与故障频带重构"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr, kurtosis
sys.path.insert(0, '.')
from vmd_core import vmd

# === 参数 ===
fs = 25600
fr = 35.0
BPFO = (8/2)*fr*(1-7.92/34.55)   # 107.9
BPFI = (8/2)*fr*(1+7.92/34.55)   # 172.1
fault_freq = BPFO  # Bearing1_1 外圈故障

# 选 Bearing1_1 的 CSV#70 (约 t=0.57, 退化中期)
csv_path = '../../data/XJTU-SY/Bearing1_1/70.csv'
df = pd.read_csv(csv_path, header=0)
sig = df.iloc[:, 0].values.astype(np.float64)
# 去均值
sig = sig - sig.mean()
N = len(sig)  # 32768

print(f'Signal: {N} points, fs={fs}Hz')
print(f'Fault freq: BPFO={BPFO:.1f}Hz')

# === VMD 分解 ===
K = 6  # 模态数
print(f'\nVMD decomposing (K={K}, alpha=2000)...')
u, u_hat, omega, omega_hz = vmd(sig, alpha=2000, tau=0, K=K, DC=False, init=1, tol=1e-7)

print('\nIMF center frequencies:')
for k in range(K):
    print(f'  IMF{k+1}: ω = {omega_hz[k]:.1f} Hz')

# === IMF 筛选 ===
# 1. 与原始信号的相关系数
r_scores = [abs(pearsonr(u[k], sig)[0]) for k in range(K)]

# 2. 包络谱峭度 (在故障频带附近)
def envelope_kurtosis(signal, fault_freq, fs, bandwidth=20):
    env = np.abs(hilbert(signal))
    env_fft = np.abs(fft(env - np.mean(env)))
    efreq = fftfreq(len(env), 1/fs)
    mask = (efreq >= fault_freq - bandwidth) & (efreq <= fault_freq + bandwidth)
    if mask.sum() > 0:
        return kurtosis(env_fft[mask])
    return 0

esk_scores = [envelope_kurtosis(u[k], fault_freq, fs) for k in range(K)]

# 3. 故障频率能量比 FCER
def fault_energy_ratio(signal, fault_freq, fs, bandwidth=15):
    env = np.abs(hilbert(signal))
    env_fft = np.abs(fft(env - np.mean(env)))
    efreq = fftfreq(len(env), 1/fs)
    N2 = len(env_fft)//2
    mask_fault = (efreq[:N2] >= fault_freq - bandwidth) & (efreq[:N2] <= fault_freq + bandwidth)
    fault_energy = env_fft[:N2][mask_fault].sum()
    total_energy = env_fft[:N2].sum()
    return fault_energy / (total_energy + 1e-10)

fcer_scores = [fault_energy_ratio(u[k], fault_freq, fs) for k in range(K)]

# Normalize and combine
esk_n = np.array(esk_scores); esk_n = (esk_n - esk_n.min()) / (esk_n.max() - esk_n.min() + 1e-10)
fcer_n = np.array(fcer_scores); fcer_n = (fcer_n - fcer_n.min()) / (fcer_n.max() - fcer_n.min() + 1e-10)
combined = 0.5 * esk_n + 0.5 * fcer_n

# Selection: r > 0.3 AND combined > median
th_r = 0.3
th_c = np.median(combined)
selected = [k for k in range(K) if r_scores[k] > th_r and combined[k] > th_c]

print('\nIMF screening:')
for k in range(K):
    status = '★ SELECTED' if k in selected else '  skipped'
    print(f'  IMF{k+1}: r={r_scores[k]:.3f}, ESK={esk_scores[k]:.1f}, FCER={fcer_scores[k]:.4f}, combined={combined[k]:.3f} {status}')

# === 加权重构 ===
if len(selected) > 0:
    sel_weights = np.array([combined[k] for k in selected])
    sel_weights = sel_weights / sel_weights.sum()
    recon = np.sum([sel_weights[i] * u[selected[i]] for i in range(len(selected))], axis=0)
else:
    recon = sig.copy()
    print('WARNING: No IMF selected, using original signal')

# === 可视化 ===
fig = plt.figure(figsize=(22, 14))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Row 0: 各 IMF 时域 (前 2000 点)
t_ms = np.arange(2000) / fs * 1000
colors = plt.cm.tab10(np.linspace(0, 1, K))
for k in range(K):
    ax = fig.add_subplot(gs[0, k])
    ax.plot(t_ms, u[k, :2000], color=colors[k], lw=0.8)
    status = '★' if k in selected else ''
    ax.set_title(f'IMF{k+1} {status} (r={r_scores[k]:.2f}, ω={omega_hz[k]:.0f}Hz)', fontsize=9,
                color='red' if k in selected else 'black')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amp')
    ax.grid(True, alpha=0.25)

# Row 1: 各 IMF 频谱 (0-2000 Hz)
freqs = fftfreq(N, 1/fs)[:N//2]
for k in range(K):
    ax = fig.add_subplot(gs[1, k])
    spec = np.abs(fft(u[k]))[:N//2]
    spec = spec / (spec.max() + 1e-10)
    mask = freqs <= 2000
    ax.plot(freqs[mask], spec[mask], color=colors[k], lw=0.8)
    ax.axvline(fault_freq, color='red', ls='--', lw=1, alpha=0.5, label=f'BPFO={fault_freq:.0f}Hz')
    ax.set_title(f'IMF{k+1} Spectrum', fontsize=9)
    ax.set_xlabel('Freq (Hz)'); ax.set_ylabel('Norm Amp')
    ax.set_xlim(0, 2000); ax.grid(True, alpha=0.25)
    if k == 0: ax.legend(fontsize=7)

# Row 2: 重构前后包络谱对比
ax = fig.add_subplot(gs[2, 0])
# 原始信号包络谱
env_orig = np.abs(hilbert(sig))
env_orig_fft = np.abs(fft(env_orig - np.mean(env_orig)))[:N//2]
env_orig_fft /= env_orig_fft.max()
mask = freqs <= 500
ax.plot(freqs[mask], env_orig_fft[mask], 'gray', lw=1.5, alpha=0.7, label='Original')
# 重构信号包络谱
env_recon = np.abs(hilbert(recon))
env_recon_fft = np.abs(fft(env_recon - np.mean(env_recon)))[:N//2]
env_recon_fft /= env_recon_fft.max()
ax.plot(freqs[mask], env_recon_fft[mask], 'red', lw=2, label='Reconstructed')
ax.axvline(BPFO, color='blue', ls='--', lw=1.5, alpha=0.7)
ax.text(BPFO, 0.95, f'BPFO={BPFO:.0f}Hz', fontsize=9, color='blue', rotation=90, va='top')
ax.set_xlim(0, 500); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Norm Amp')
ax.set_title('Envelope Spectrum: Original vs Reconstructed', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

# Row 2 col 1: 故障频带局部放大对比
ax = fig.add_subplot(gs[2, 1])
f_lo, f_hi = BPFO-40, BPFO+40
mask_zoom = (freqs >= f_lo) & (freqs <= f_hi)
ax.plot(freqs[mask_zoom], env_orig_fft[mask_zoom], 'gray', lw=2, alpha=0.7, label='Original')
ax.plot(freqs[mask_zoom], env_recon_fft[mask_zoom], 'red', lw=2.5, label='Reconstructed')
ax.axvline(BPFO, color='blue', ls='--', lw=1.5)
# Calculate enhancement
bpfo_orig = env_orig_fft[(freqs >= BPFO-5) & (freqs <= BPFO+5)].max()
bpfo_recon = env_recon_fft[(freqs >= BPFO-5) & (freqs <= BPFO+5)].max()
enhance = bpfo_recon / (bpfo_orig + 1e-10)
ax.set_xlim(f_lo, f_hi); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Norm Amp')
ax.set_title(f'Zoom: BPFO ± 40Hz\nEnhancement: {enhance:.1f}×', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

# Row 2 col 2: 筛选指标
ax = fig.add_subplot(gs[2, 2])
x_pos = np.arange(K)
w = 0.25
ax.bar(x_pos - w, r_scores, w, color='steelblue', label='Correlation')
ax.bar(x_pos, esk_n, w, color='coral', label='ESK (norm)')
ax.bar(x_pos + w, fcer_n, w, color='seagreen', label='FCER (norm)')
ax.axhline(th_r, color='steelblue', ls=':', lw=1, alpha=0.7)
ax.axhline(th_c, color='gray', ls='--', lw=1.5, alpha=0.7, label=f'Threshold={th_c:.2f}')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'IMF{k+1}' for k in range(K)])
ax.set_ylabel('Score'); ax.set_title('IMF Selection Scores', fontsize=11, fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=0.25, axis='y')

plt.suptitle(f'VMD Decomposition & Fault-Band Reconstruction\n'
             f'Bearing1_1 CSV#70 (t≈0.57) | K={K}, α=2000 | {len(selected)}/{K} IMFs selected',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fname = 'figures/vmd_decompose.png'
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: {fname}')

# Print enhancement details
print(f'\nEnhancement: BPFO peak {enhance:.2f}×')
print(f'Original BPFO peak: {bpfo_orig:.4f}')
print(f'Reconstructed BPFO peak: {bpfo_recon:.4f}')
