"""
诊断FPT检测问题
可视化每个轴承的RMS曲线和检测到的FPT点
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

def extract_rms(file_path):
    """提取RMS特征"""
    try:
        df = pd.read_csv(file_path, header=None)
        signal = df.iloc[:, 4].values
    except:
        try:
            df = pd.read_csv(file_path, header=None, sep=r'\s+')
            signal = df.iloc[:, 4].values
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    return np.sqrt(np.mean(signal**2))

def process_bearing(dir_path):
    """处理单个轴承，返回RMS序列"""
    files = sorted(glob.glob(os.path.join(dir_path, "*.*")))
    files = [f for f in files if "acc_" in os.path.basename(f)]

    rms_list = []
    for f in files:
        rms = extract_rms(f)
        if rms is not None:
            rms_list.append(rms)
    return np.array(rms_list)

def find_fpt_slope_change(rms_array, healthy_samples=200):
    """斜率变化法检测FPT"""
    from scipy.ndimage import gaussian_filter1d

    smoothed_rms = gaussian_filter1d(rms_array, sigma=10)
    slope = np.gradient(smoothed_rms)
    smoothed_slope = gaussian_filter1d(slope, sigma=5)

    healthy_slope_mean = np.mean(smoothed_slope[:healthy_samples])
    healthy_slope_std = np.std(smoothed_slope[:healthy_samples])
    slope_threshold = healthy_slope_mean + 3 * healthy_slope_std

    sustain_steps = 30
    consecutive_count = 0

    for i in range(healthy_samples, len(smoothed_slope)):
        if smoothed_slope[i] > slope_threshold:
            consecutive_count += 1
            if consecutive_count >= sustain_steps:
                return i - sustain_steps + 1, slope_threshold, smoothed_slope
        else:
            consecutive_count = 0

    return int(len(rms_array) * 0.7), slope_threshold, smoothed_slope

def find_fpt_macro_amplitude(rms_array, healthy_samples=200):
    """当前的FPT检测算法"""
    healthy_mean = np.mean(rms_array[:healthy_samples])
    healthy_std = np.std(rms_array[:healthy_samples])
    failure_amplitude = np.mean(rms_array[-50:])

    # 新的保守阈值
    threshold = healthy_mean + 3 * healthy_std
    amplitude_threshold = healthy_mean + 0.15 * (failure_amplitude - healthy_mean)
    threshold = max(threshold, amplitude_threshold)

    sustain_steps = 20
    consecutive_count = 0

    for i in range(healthy_samples, len(rms_array)):
        if rms_array[i] > threshold:
            consecutive_count += 1
            if consecutive_count >= sustain_steps:
                return i - sustain_steps + 1, threshold
        else:
            consecutive_count = 0

    return int(len(rms_array) * 0.7), threshold

def main():
    # 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    all_dirs = config['data']['train_dirs'] + config['data']['test_dirs']

    fig, axes = plt.subplots(len(all_dirs), 2, figsize=(18, 4*len(all_dirs)))
    if len(all_dirs) == 1:
        axes = axes.reshape(1, -1)

    for idx, dir_path in enumerate(all_dirs):
        print(f"\n处理: {dir_path}")
        rms_array = process_bearing(dir_path)

        # 归一化（Z-score）
        rms_norm = (rms_array - rms_array.mean()) / (rms_array.std() + 1e-8)

        # 方法1：幅值法
        fpt_amp, threshold_amp = find_fpt_macro_amplitude(rms_norm,
                                                          config['data']['healthy_samples'])

        # 方法2：斜率法
        fpt_slope, threshold_slope, smoothed_slope = find_fpt_slope_change(rms_norm,
                                                                           config['data']['healthy_samples'])

        print(f"  RMS范围: {rms_norm.min():.3f} ~ {rms_norm.max():.3f}")
        print(f"  幅值法FPT: {fpt_amp} / {len(rms_norm)} ({100*fpt_amp/len(rms_norm):.1f}%)")
        print(f"  斜率法FPT: {fpt_slope} / {len(rms_norm)} ({100*fpt_slope/len(rms_norm):.1f}%)")

        # 左图：幅值法
        ax1 = axes[idx, 0]
        ax1.plot(rms_norm, 'b-', linewidth=1, label='RMS (normalized)')
        ax1.axhline(threshold_amp, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold_amp:.3f}')
        ax1.axvline(fpt_amp, color='orange', linestyle='--', linewidth=2, label=f'FPT={fpt_amp}')
        ax1.axvspan(0, config['data']['healthy_samples'], alpha=0.2, color='green', label='Healthy Period')
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Normalized RMS', fontsize=11)
        ax1.set_title(f'{os.path.basename(dir_path)} - Amplitude Method', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 右图：斜率法
        ax2 = axes[idx, 1]
        ax2_twin = ax2.twinx()
        ax2.plot(rms_norm, 'b-', linewidth=1, label='RMS (normalized)')
        ax2_twin.plot(smoothed_slope, 'g-', linewidth=1, alpha=0.6, label='Slope (smoothed)')
        ax2_twin.axhline(threshold_slope, color='r', linestyle='--', linewidth=2, label=f'Slope Threshold')
        ax2.axvline(fpt_slope, color='purple', linestyle='--', linewidth=2, label=f'FPT={fpt_slope}')
        ax2.axvspan(0, config['data']['healthy_samples'], alpha=0.2, color='green')
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Normalized RMS', fontsize=11, color='b')
        ax2_twin.set_ylabel('Slope', fontsize=11, color='g')
        ax2.set_title(f'{os.path.basename(dir_path)} - Slope Method', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./results/fpt_diagnosis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 诊断图已保存: ./results/fpt_diagnosis.png")

if __name__ == '__main__':
    main()
