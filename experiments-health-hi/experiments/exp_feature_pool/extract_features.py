"""特征池提取：Pool A (重构信号) + Pool B (原始信号)

每个特征 = 一条时间序列，长度 = CSV 文件数

输出：
  features/pool_A.npz   — 故障频带特征 (~16 个)
  features/pool_B.npz   — 全局统计特征 (~25 个)
  features/meta.json    — 特征名列表和来源标注
"""

import sys, numpy as np, pandas as pd, json, os, glob
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../exp_vmd_test'))
from vmd_core import vmd

fs = 25600; fr = 35.0
BPFO = (8/2)*fr*(1-7.92/34.55)
BPFI = (8/2)*fr*(1+7.92/34.55)

def adaptive_K(sig):
    prev = None
    for K in range(2, 11):
        _, _, omega, _ = vmd(sig, alpha=2000, tau=0, K=K, DC=False, init=1, tol=1e-7)
        os_ = np.sort(omega)
        if prev is not None and len(os_) == len(prev) + 1:
            if abs(os_[-1] - os_[-2]) < 0.05: return K - 1
        prev = os_
    return 10

def env_pf(imf):
    env = np.abs(hilbert(imf))
    ef = np.abs(fft(env - np.mean(env)))
    ef = ef[:len(ef)//2]
    return ef.max() / (np.median(ef) + 1e-10)

# ========== 特征提取函数 ==========

def time_features(sig):
    """提取时域特征，返回 dict"""
    s = np.asarray(sig, dtype=np.float64)
    mean_abs = np.mean(np.abs(s))
    rms = np.sqrt(np.mean(s**2))
    peak = np.max(np.abs(s))
    return {
        'RMS': rms,
        'Peak': peak,
        'Kurtosis': float(kurtosis(s)),
        'Skewness': float(skew(s)),
        'CrestFactor': peak / (rms + 1e-10),
        'ImpulseFactor': peak / (mean_abs + 1e-10),
        'ShapeFactor': rms / (mean_abs + 1e-10),
        'PeakToPeak': np.max(s) - np.min(s),
        'Variance': float(np.var(s)),
        'Std': float(np.std(s)),
        'MeanAbs': mean_abs,
        'Energy': float(np.sum(s**2)),
    }

def freq_features(sig, fs):
    """提取频域特征"""
    s = np.asarray(sig, dtype=np.float64)
    N = len(s); spec = np.abs(fft(s))[:N//2]; freqs = fftfreq(N, 1/fs)[:N//2]
    power = spec**2; total_power = power.sum() + 1e-10
    
    # Spectral moments
    centroid = np.sum(freqs * power) / total_power
    spread = np.sqrt(np.sum(((freqs - centroid)**2) * power) / total_power)
    
    # Sub-band energy ratios
    b0 = (freqs >= 0) & (freqs < 1000)
    b1 = (freqs >= 1000) & (freqs < 3000)
    b2 = (freqs >= 3000) & (freqs < 6000)
    b3 = (freqs >= 6000) & (freqs < 12800)
    
    return {
        'FreqCentroid': centroid,
        'FreqSpread': spread,
        'RMSFreq': np.sqrt(np.sum(freqs**2 * power) / total_power),
        'FreqVariance': float(np.var(freqs.repeat((power/power.max()*100).astype(int)+1)[:len(freqs)]) if total_power > 0 else 0),
        'BandEnergy_0_1kHz': float(power[b0].sum() / total_power),
        'BandEnergy_1_3kHz': float(power[b1].sum() / total_power),
        'BandEnergy_3_6kHz': float(power[b2].sum() / total_power),
        'BandEnergy_6_12_8kHz': float(power[b3].sum() / total_power),
    }

def envelope_features(sig, fs):
    """提取包络谱相关特征 (Pool A 特有)"""
    env = np.abs(hilbert(sig))
    env = env - np.mean(env)
    ef = np.abs(fft(env))[:len(sig)//2]
    efreq = fftfreq(len(sig), 1/fs)[:len(sig)//2]
    total = ef.sum() + 1e-10
    
    m_bpfo = (efreq >= BPFO-10) & (efreq <= BPFO+10)
    m_bpfi = (efreq >= BPFI-10) & (efreq <= BPFI+10)
    
    return {
        'EnvPeakFactor': ef.max() / (np.median(ef) + 1e-10),
        'BPFO_Energy': float(ef[m_bpfo].sum() / total),
        'BPFI_Energy': float(ef[m_bpfi].sum() / total),
        'EnvKurtosis': float(kurtosis(ef)),
        'EnvRMS': float(np.sqrt(np.mean(ef**2))),
    }

# ========== 主程序 ==========
print('Loading Bearing1_1 data...')
csv_files = sorted(glob.glob('../../data/XJTU-SY/Bearing1_1/*.csv'),
                   key=lambda x: int(x.split('/')[-1].split('.')[0]))
n_csv = len(csv_files)

# 尝试加载已保存的重构信号（如果实验一已运行）
recon_path = '../../exp_vmd_test/reconstructed/Bearing1_1_recon.npz'
recon_precomputed = os.path.exists(recon_path)

if recon_precomputed:
    print('Loading precomputed reconstructed signals...')
    recon_data = np.load(recon_path, allow_pickle=True)
    recon_all = recon_data.get('recon', None)
    # Check if we can use precomputed
    if recon_all is not None:
        print(f'  Using {len(recon_all)} precomputed reconstructions')
else:
    print('No precomputed reconstruction found, will compute fresh')

# Initialize feature arrays
pool_A_names = []
pool_A_data = []  # list of (name, array)
pool_B_names = []
pool_B_data = []

# 第一遍：收集原始信号 + 重构信号
all_sig = []  # all CSVs original signals
all_recon = []  # all CSVs reconstructed signals

for ci, csv_path in enumerate(csv_files):
    df = pd.read_csv(csv_path, header=0)
    sig = df.iloc[:, 0].values.astype(np.float64)
    sig = sig - np.mean(sig)
    all_sig.append(sig)
    
    if recon_precomputed and recon_all is not None and ci < len(recon_all):
        recon = recon_all[ci]
    else:
        # Compute VMD + reconstruction
        K = adaptive_K(sig)
        u, _, omega, _ = vmd(sig, alpha=2000, tau=0, K=K, DC=False, init=1, tol=1e-7)
        scores = np.array([env_pf(u[k]) for k in range(K)])
        ss = np.sort(scores)[::-1]
        gaps = np.diff(ss)
        gap_idx = np.argmin(gaps)
        max_gap = abs(gaps[gap_idx])
        med = np.median(scores)
        if max_gap < 1.5 * med:
            sel = list(range(K))
        else:
            sel = list(np.argsort(scores)[::-1][:gap_idx+1])
            sel.sort()
        recon = np.sum([u[k] for k in sel], axis=0)
    all_recon.append(recon)

# 第二遍：提取特征
print(f'Extracting features from {n_csv} CSVs...')
pool_A_feats = {}  # name -> list of values
pool_B_feats = {}

for ci in range(n_csv):
    sig = all_sig[ci]
    recon = all_recon[ci]
    
    # Pool A: from reconstructed signal
    feats_a = {}
    feats_a.update(time_features(recon))
    feats_a.update(freq_features(recon, fs))
    feats_a.update(envelope_features(recon, fs))
    for k, v in feats_a.items():
        if k not in pool_A_feats: pool_A_feats[k] = []
        pool_A_feats[k].append(v)
    
    # Pool B: from original signal
    feats_b = {}
    feats_b.update(time_features(sig))
    feats_b.update(freq_features(sig, fs))
    for k, v in feats_b.items():
        if k not in pool_B_feats: pool_B_feats[k] = []
        pool_B_feats[k].append(v)

    if (ci+1) % 20 == 0:
        print(f'  {ci+1}/{n_csv} CSV processed')

# Convert to arrays and normalize
pool_A = {}
pool_B = {}
for name, vals in pool_A_feats.items():
    arr = np.array(vals, dtype=np.float64)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    pool_A[name] = arr

for name, vals in pool_B_feats.items():
    arr = np.array(vals, dtype=np.float64)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    pool_B[name] = arr

# Save
os.makedirs('features', exist_ok=True)
np.savez_compressed('features/pool_A.npz', **pool_A)
np.savez_compressed('features/pool_B.npz', **pool_B)

meta = {
    'n_csv': n_csv,
    'bearing': 'Bearing1_1',
    'fs': fs, 'BPFO': BPFO, 'BPFI': BPFI,
    'pool_A_names': list(pool_A.keys()),
    'pool_B_names': list(pool_B.keys()),
    'pool_A_description': 'Features from VMD fault-band reconstructed signal',
    'pool_B_description': 'Features from original vibration signal',
}

with open('features/meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f'\nPool A: {len(pool_A)} features × {n_csv} CSVs')
for name in sorted(pool_A.keys()):
    print(f'  {name}: [{pool_A[name].min():.3f}, {pool_A[name].max():.3f}]')

print(f'\nPool B: {len(pool_B)} features × {n_csv} CSVs')
for name in sorted(pool_B.keys()):
    print(f'  {name}: [{pool_B[name].min():.3f}, {pool_B[name].max():.3f}]')

print(f'\nSaved: features/pool_A.npz, features/pool_B.npz, features/meta.json')
