"""
健康指标计算模块
方案A：时域 + 频域组合
HI1 = RMS（时域能量指标）
HI2 = 包络谱BPFO幅值（频域故障特征）
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from typing import Dict, List, Tuple
from config import get_fault_frequencies, DATA_CONFIG


class HealthIndicatorComputer:
    """健康指标计算器"""

    def __init__(self, working_condition: int, fs: int = 25600):
        """
        初始化

        Args:
            working_condition: 工况编号 (1, 2, 或 3)
            fs: 采样频率 (Hz)
        """
        self.working_condition = working_condition
        self.fs = fs
        self.fault_freqs = get_fault_frequencies(working_condition)

    def _highpass_filter(self, signal: np.ndarray, cutoff: float = 1000, order: int = 2) -> np.ndarray:
        """
        高通滤波

        Args:
            signal: 输入信号
            cutoff: 截止频率 (Hz)
            order: 滤波器阶数

        Returns:
            滤波后的信号
        """
        nyquist = self.fs / 2
        cutoff_norm = cutoff / nyquist
        b, a = butter(order, cutoff_norm, btype='high')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def _compute_envelope_spectrum(self, signal: np.ndarray, highpass_cutoff: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算包络谱

        Args:
            signal: 输入信号
            highpass_cutoff: 高通滤波截止频率 (Hz)

        Returns:
            freqs: 频率轴
            envelope_spectrum: 包络谱幅值
        """
        filtered_signal = self._highpass_filter(signal, highpass_cutoff)
        analytic_signal = hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        envelope_fft = np.fft.fft(envelope)
        envelope_spectrum = np.abs(envelope_fft[:len(envelope_fft)//2])
        freqs = np.fft.fftfreq(len(envelope), 1/self.fs)[:len(envelope_fft)//2]
        return freqs, envelope_spectrum

    def _extract_frequency_amplitude(self, freqs: np.ndarray, spectrum: np.ndarray,
                                    target_freq: float, tolerance: float = 2.0) -> float:
        """
        从频谱中提取特定频率处的幅值

        Args:
            freqs: 频率轴
            spectrum: 频谱幅值
            target_freq: 目标频率 (Hz)
            tolerance: 容差范围 (Hz)

        Returns:
            该频率处的幅值
        """
        freq_mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)
        if np.any(freq_mask):
            amplitude = np.max(spectrum[freq_mask])
        else:
            amplitude = 0.0
        return amplitude

    def compute_hi1(self, signal: np.ndarray) -> float:
        """
        计算HI1：RMS（时域能量指标）

        Args:
            signal: 输入信号

        Returns:
            HI1值（RMS）
        """
        rms = np.sqrt(np.mean(signal ** 2))
        return rms

    def compute_hi2(self, signal: np.ndarray, fault_type: str = 'BPFO',
                   num_harmonics: int = 3, highpass_cutoff: float = 1000) -> float:
        """
        计算HI2：包络谱中故障频率的幅值（含谐波）

        Args:
            signal: 输入信号
            fault_type: 故障类型 ('BPFO', 'BPFI', 'BSF', 'FTF')
            num_harmonics: 谐波数量（包括基频）
            highpass_cutoff: 高通滤波截止频率 (Hz)

        Returns:
            HI2值（基频 + 谐波幅值之和）
        """
        freqs, envelope_spectrum = self._compute_envelope_spectrum(signal, highpass_cutoff)
        fault_freq = self.fault_freqs[fault_type]

        hi2 = 0.0
        for k in range(1, num_harmonics + 1):
            harmonic_freq = k * fault_freq
            amplitude = self._extract_frequency_amplitude(freqs, envelope_spectrum, harmonic_freq)
            hi2 += amplitude

        return hi2

    def compute_hi_for_bearing(self, data_list: List[np.ndarray],
                              fault_type: str = 'BPFO') -> Dict[str, np.ndarray]:
        """
        计算整个轴承的健康指标序列

        Args:
            data_list: 信号列表（每个元素是一个CSV文件的数据）
            fault_type: 故障类型 ('BPFO', 'BPFI', 'BSF', 'FTF')

        Returns:
            包含HI1和HI2序列的字典
        """
        hi1_values = []
        hi2_values = []

        for signal in data_list:
            hi1 = self.compute_hi1(signal)
            hi2 = self.compute_hi2(signal, fault_type=fault_type)
            hi1_values.append(hi1)
            hi2_values.append(hi2)

        return {
            'hi1': np.array(hi1_values),
            'hi2': np.array(hi2_values)
        }


if __name__ == '__main__':
    # 测试代码
    from data_loader import XJTUDataLoader
    from config import DATA_PATHS

    print("测试HI计算模块...")
    print("=" * 50)

    # 加载Bearing1_1数据
    loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])
    data_list, file_names = loader.load_bearing_data_by_file('Bearing1_1')
    print(f"\n加载Bearing1_1数据: {len(data_list)}个文件")

    # 初始化HI计算器（工况1，外圈故障）
    hi_computer = HealthIndicatorComputer(working_condition=1)

    # 计算健康指标
    print("\n计算健康指标...")
    hi_dict = hi_computer.compute_hi_for_bearing(data_list, fault_type='BPFO')

    print(f"\nHI1 (RMS):")
    print(f"  长度: {len(hi_dict['hi1'])}")
    print(f"  范围: [{hi_dict['hi1'].min():.2f}, {hi_dict['hi1'].max():.2f}]")
    print(f"  前5个值: {hi_dict['hi1'][:5]}")
    print(f"  后5个值: {hi_dict['hi1'][-5:]}")

    print(f"\nHI2 (包络谱BPFO幅值):")
    print(f"  长度: {len(hi_dict['hi2'])}")
    print(f"  范围: [{hi_dict['hi2'].min():.2f}, {hi_dict['hi2'].max():.2f}]")
    print(f"  前5个值: {hi_dict['hi2'][:5]}")
    print(f"  后5个值: {hi_dict['hi2'][-5:]}")

    print("\n" + "=" * 50)
    print("测试完成！")


