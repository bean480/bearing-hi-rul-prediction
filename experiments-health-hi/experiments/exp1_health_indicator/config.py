"""
实验1：健康指标构建 - 配置文件
包含轴承参数、数据路径、特征提取参数等
"""

import numpy as np

# ==================== 轴承参数 ====================
BEARING_PARAMS = {
    'name': 'LDK UER204',
    'inner_diameter': 29.30,      # 内圈滚道直径 (mm)
    'outer_diameter': 39.80,      # 外圈滚道直径 (mm)
    'pitch_diameter': 34.55,      # 节圆直径 Pd (mm)
    'ball_diameter': 7.92,        # 滚珠直径 Bd (mm)
    'num_balls': 8,               # 滚珠个数 N
    'contact_angle': 0,           # 接触角 α (度)
}

# ==================== 工况参数 ====================
WORKING_CONDITIONS = {
    1: {'rpm': 2100, 'load_kN': 12, 'freq_hz': 35.0},
    2: {'rpm': 2250, 'load_kN': 11, 'freq_hz': 37.5},
    3: {'rpm': 2400, 'load_kN': 10, 'freq_hz': 40.0},
}

# ==================== 故障特征频率系数 ====================
# 相对于转速频率 fr 的倍数
FAULT_FREQ_COEFFS = {
    'BPFO': 3.083,   # 外圈故障频率系数
    'BPFI': 4.917,   # 内圈故障频率系数
    'BSF': 2.066,    # 滚珠故障频率系数
    'FTF': 0.385,    # 保持架频率系数
}

# ==================== 数据参数 ====================
DATA_CONFIG = {
    'sampling_rate': 25600,       # 采样频率 (Hz)
    'points_per_file': 32768,     # 每个文件的采样点数
    'duration_per_file': 1.28,    # 每个文件时长 (秒)
}

# ==================== 特征提取参数 ====================
FEATURE_CONFIG = {
    'window_size': 1024,          # 滑动窗口大小
    'overlap_ratio': 0.5,         # 重叠比例 (50%)
    'hop_length': 512,            # 跳跃长度 = window_size * (1 - overlap_ratio)

    # 滤波参数（用于包络分析）
    'highpass_cutoff': 1000,      # 高通滤波截止频率 (Hz)
    'filter_order': 2,            # 滤波器阶数
}

# ==================== 数据路径 ====================
DATA_PATHS = {
    'xjtu_root': 'c:/Users/53031/Desktop/new-exp/data/XJTU-SY',
    'phm2012_root': 'c:/Users/53031/Desktop/new-exp/data/PHM2012',
    'output_root': 'c:/Users/53031/Desktop/new-exp/results/exp1',
}

# ==================== 评估指标阈值 ====================
EVALUATION_THRESHOLDS = {
    'monotonicity_min': 0.7,      # 单调性最小值
    'trendability_min': 0.9,      # 趋势性最小值
    'robustness_min': 0.6,        # 鲁棒性最小值
}

# ==================== 可视化参数 ====================
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
}


def get_fault_frequencies(working_condition: int) -> dict:
    """
    计算指定工况下的故障特征频率

    Args:
        working_condition: 工况编号 (1, 2, 或 3)

    Returns:
        包含各故障频率的字典 (Hz)
    """
    fr = WORKING_CONDITIONS[working_condition]['freq_hz']

    return {
        'BPFO': FAULT_FREQ_COEFFS['BPFO'] * fr,
        'BPFI': FAULT_FREQ_COEFFS['BPFI'] * fr,
        'BSF': FAULT_FREQ_COEFFS['BSF'] * fr,
        'FTF': FAULT_FREQ_COEFFS['FTF'] * fr,
        'fr': fr,  # 转速频率
    }


if __name__ == '__main__':
    # 测试：打印各工况的故障频率
    print("=" * 50)
    print("XJTU-SY轴承故障特征频率")
    print("=" * 50)

    for condition in [1, 2, 3]:
        freqs = get_fault_frequencies(condition)
        print(f"\n工况{condition} (转速: {WORKING_CONDITIONS[condition]['rpm']} rpm)")
        print(f"  转速频率 (fr):  {freqs['fr']:.1f} Hz")
        print(f"  外圈故障 (BPFO): {freqs['BPFO']:.1f} Hz")
        print(f"  内圈故障 (BPFI): {freqs['BPFI']:.1f} Hz")
        print(f"  滚珠故障 (BSF):  {freqs['BSF']:.1f} Hz")
        print(f"  保持架 (FTF):   {freqs['FTF']:.1f} Hz")
