"""
实验1：健康指标构建 - 特征提取模块
提取时域、频域和包络谱特征
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """振动信号特征提取器"""

    def __init__(self, sampling_rate: int = 25600):
        """
        初始化特征提取器

        Args:
            sampling_rate: 采样频率 (Hz)
        """
        self.fs = sampling_rate

    def extract_time_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        提取时域特征

        Args:
            data: 振动信号，shape = (n_samples,)

        Returns:
            时域特征字典
        """
        features = {}

        # 1. RMS (均方根)
        features['rms'] = np.sqrt(np.mean(data ** 2))

        # 2. 峰度 (Kurtosis)
        features['kurtosis'] = np.mean((data - np.mean(data)) ** 4) / (np.std(data) ** 4)

        # 3. 峰峰值 (Peak-to-Peak)
        features['peak_to_peak'] = np.max(data) - np.min(data)

        # 4. 波形因子 (Shape Factor)
        mean_abs = np.mean(np.abs(data))
        if mean_abs > 1e-10:
            features['shape_factor'] = features['rms'] / mean_abs
        else:
            features['shape_factor'] = 0.0

        # 5. 峰值 (Peak)
        features['peak'] = np.max(np.abs(data))

        return features

    def extract_frequency_features(self,
                                   data: np.ndarray,
                                   fault_freqs: Dict[str, float] = None) -> Dict[str, float]:
        """
        提取频域特征

        Args:
            data: 振动信号
            fault_freqs: 故障特征频率字典 (可选)

        Returns:
            频域特征字典
        """
        features = {}

        # FFT计算
        n = len(data)
        fft_vals = fft(data)
        fft_freqs = fftfreq(n, 1/self.fs)

        # 只取正频率部分
        positive_freq_idx = fft_freqs > 0
        freqs = fft_freqs[positive_freq_idx]
        amplitudes = np.abs(fft_vals[positive_freq_idx])

        # 1. 功率谱密度峰值
        psd = amplitudes ** 2
        features['psd_peak'] = np.max(psd)

        # 2. 频谱重心
        features['spectral_centroid'] = np.sum(freqs * amplitudes) / np.sum(amplitudes)

        # 3. 如果提供了故障频率，提取对应频率处的幅值
        if fault_freqs is not None:
            for fault_type, fault_freq in fault_freqs.items():
                # 找到最接近故障频率的索引
                idx = np.argmin(np.abs(freqs - fault_freq))
                features[f'{fault_type}_amplitude'] = amplitudes[idx]

        return features

    def extract_envelope_features(self,
                                  data: np.ndarray,
                                  highpass_cutoff: float = 1000,
                                  filter_order: int = 2) -> Dict[str, float]:
        """
        提取包络谱特征

        Args:
            data: 振动信号
            highpass_cutoff: 高通滤波截止频率 (Hz)
            filter_order: 滤波器阶数

        Returns:
            包络谱特征字典
        """
        features = {}

        # 1. 高通滤波
        nyquist = 0.5 * self.fs
        normal_cutoff = highpass_cutoff / nyquist
        b, a = signal.butter(filter_order, normal_cutoff, btype='high', analog=False)
        filtered_signal = signal.filtfilt(b, a, data)

        # 2. Hilbert变换提取包络
        analytic_signal = signal.hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)

        # 3. 包络谱峰值
        features['envelope_peak'] = np.max(envelope)

        # 4. 包络平方幅值谱 (ESAS)
        envelope_squared = envelope ** 2
        envelope_fft = fft(envelope_squared)
        envelope_spectrum = np.abs(envelope_fft[:len(envelope_fft)//2])
        features['esas_peak'] = np.max(envelope_spectrum)

        # 5. 包络谱均值
        features['envelope_mean'] = np.mean(envelope)

        return features

    def extract_all_features(self,
                            data: np.ndarray,
                            fault_freqs: Dict[str, float] = None,
                            highpass_cutoff: float = 1000) -> Dict[str, float]:
        """
        提取所有特征（时域+频域+包络谱）

        Args:
            data: 振动信号
            fault_freqs: 故障特征频率字典
            highpass_cutoff: 高通滤波截止频率

        Returns:
            所有特征的字典
        """
        all_features = {}

        # 时域特征
        time_features = self.extract_time_features(data)
        all_features.update(time_features)

        # 频域特征
        freq_features = self.extract_frequency_features(data, fault_freqs)
        all_features.update(freq_features)

        # 包络谱特征
        envelope_features = self.extract_envelope_features(data, highpass_cutoff)
        all_features.update(envelope_features)

        return all_features

    def extract_features_from_windows(self,
                                     windows: np.ndarray,
                                     fault_freqs: Dict[str, float] = None,
                                     highpass_cutoff: float = 1000) -> Tuple[np.ndarray, list]:
        """
        从多个窗口批量提取特征

        Args:
            windows: 窗口数组，shape = (n_windows, window_size)
            fault_freqs: 故障特征频率字典
            highpass_cutoff: 高通滤波截止频率

        Returns:
            feature_matrix: 特征矩阵，shape = (n_windows, n_features)
            feature_names: 特征名称列表
        """
        n_windows = windows.shape[0]

        # 提取第一个窗口的特征，获取特征名称
        first_features = self.extract_all_features(windows[0], fault_freqs, highpass_cutoff)
        feature_names = list(first_features.keys())
        n_features = len(feature_names)

        # 初始化特征矩阵
        feature_matrix = np.zeros((n_windows, n_features))
        feature_matrix[0] = list(first_features.values())

        # 批量提取特征
        print(f"提取特征: {n_windows} 个窗口, {n_features} 个特征...")

        for i in range(1, n_windows):
            if (i + 1) % 1000 == 0:
                print(f"  进度: {i+1}/{n_windows}")

            features = self.extract_all_features(windows[i], fault_freqs, highpass_cutoff)
            feature_matrix[i] = [features[name] for name in feature_names]

        print(f"特征提取完成！")

        return feature_matrix, feature_names


if __name__ == '__main__':
    # 测试特征提取
    from data_loader import XJTUDataLoader
    from config import DATA_PATHS, FEATURE_CONFIG, get_fault_frequencies

    print("=" * 60)
    print("测试特征提取模块")
    print("=" * 60)

    # 加载数据
    loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])
    windows, time_idx = loader.load_bearing_data_windowed(
        'Bearing1_1',
        window_size=FEATURE_CONFIG['window_size'],
        hop_length=FEATURE_CONFIG['hop_length']
    )

    print(f"\n窗口数据形状: {windows.shape}")

    # 获取工况1的故障频率
    fault_freqs = get_fault_frequencies(1)
    print(f"\n工况1故障频率: {fault_freqs}")

    # 初始化特征提取器
    extractor = FeatureExtractor(sampling_rate=25600)

    # 测试单个窗口
    print(f"\n测试单个窗口特征提取...")
    single_features = extractor.extract_all_features(windows[0], fault_freqs)
    print(f"特征数量: {len(single_features)}")
    print(f"特征名称: {list(single_features.keys())}")
    print(f"\n前5个特征值:")
    for i, (name, value) in enumerate(list(single_features.items())[:5]):
        print(f"  {name}: {value:.6f}")

    # 测试批量提取（只提取前100个窗口，避免太慢）
    print(f"\n测试批量特征提取（前100个窗口）...")
    feature_matrix, feature_names = extractor.extract_features_from_windows(
        windows[:100],
        fault_freqs,
        FEATURE_CONFIG['highpass_cutoff']
    )

    print(f"\n特征矩阵形状: {feature_matrix.shape}")
    print(f"特征名称: {feature_names}")
