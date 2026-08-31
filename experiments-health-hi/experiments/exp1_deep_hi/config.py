"""
实验1 深度HI — 配置文件
基于分段时序对比学习 + 值感知单调性约束
"""

import os

# ==================== 路径 ====================
import os as _os
DATA_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../data/XJTU-SY')
OUTPUT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../results/exp1_deep')

# ==================== 信号参数 ====================
SAMPLING_RATE = 25600       # Hz
POINTS_PER_FILE = 32768     # 每个CSV采样点数

# ==================== 工况参数 ====================
WORKING_CONDITIONS = {
    1: {'rpm': 2100, 'load_kN': 12, 'freq_hz': 35.0},
    2: {'rpm': 2250, 'load_kN': 11, 'freq_hz': 37.5},
    3: {'rpm': 2400, 'load_kN': 10, 'freq_hz': 40.0},
}

def get_condition(bearing_name):
    """Bearing1_1 → 1, Bearing2_3 → 2, Bearing3_5 → 3"""
    return int(bearing_name.split('_')[0][-1])

# ==================== 窗口参数 ====================
WINDOW_SIZE = 1024           # 窗口长度 (采样点)
HOP_LENGTH  = 512            # 跳跃长度 (50%重叠)
# 每个CSV产出: (32768 - 1024) // 512 + 1 = 63 个窗口

# ==================== 轴承列表 ====================
BEARINGS_BY_CONDITION = {
    1: ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_4', 'Bearing1_5'],
    2: ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5'],
    3: ['Bearing3_1', 'Bearing3_2', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5'],
}

ALL_BEARINGS = [f'Bearing{c}_{i}' for c in [1,2,3] for i in [1,2,3,4,5]]

# ==================== 模型参数 ====================
ENCODER_CHANNELS = [16, 32, 64, 128]        # 逐层通道数
KERNEL_SIZES     = [64, 32, 8, 4]            # 逐层卷积核 (最后一层小核避免超界)
STRIDES          = [8, 4, 2, 1]              # 逐层步长
"""
输出维度验证:
  L=1024:
    Conv1(k=64,s=8) → (1024-64)//8+1 = 121
    Conv2(k=32,s=4) → (121-32)//4+1  = 23
    Conv3(k=8, s=2) → (23-8)//2+1    = 8
    Conv4(k=4, s=1) → (8-4)//1+1     = 5
    Flatten: 128*5 = 640
"""
LATENT_DIM       = 128                       # 表征向量维度
PROJECTOR_HIDDEN = 32                        # 投影头隐藏层维度

# ==================== 训练参数 ====================
EPOCHS           = 300
LR_WARMUP        = 1e-4
LR_MAX           = 1e-3
LR_MIN           = 1e-6
WARMUP_EPOCHS    = 10
PHYSICS_START    = 20                        # 从第几个epoch开始引入物理约束
TAU              = 0.1                       # 对比学习温度
DELTA_POS        = 5                         # 段内正样本对最大窗口间隔
SEG_WINDOWS_MIN  = 500                       # 每段最少窗口数（用于自适应K）

# ==================== 损失权重（warmup后）====================
LAMBDA_AUG        = 0.5    # 增强对比
LAMBDA_SEG_TEMP   = 0.5    # 分段时序对比
LAMBDA_VALUE_MONO = 2.0    # 值感知单调性 (加大)
LAMBDA_TREND      = 0.3    # 趋势一致性
LAMBDA_ANCHOR     = 0.5    # 锚点损失 (早期→0, 末期→1)
MONO_EPS          = 1e-3   # 单调性容忍微小下降

# ==================== 时序上下文 ====================
TEMPORAL_KERNEL_SIZE = 7          # 时间维1D卷积核大小 (覆盖前后各3个邻窗)
TEMPORAL_LAYERS      = 2           # 时间卷积层数
TIME_EMBED_DIM       = 16          # 时间位置嵌入维度

# ==================== 训练子采样 ====================
TRAIN_WINDOWS_PER_BEARING = 2000  # 每epoch每轴承随机采样的窗口数
# Bearing1_1: 7749窗→采2000, Bearing3_1: 160k窗→采2000
# 保证每个轴承每epoch计算量一致，大轴承不拖慢训练

# ==================== 增强参数 ====================
AUG_NOISE_STD  = 0.01
AUG_SCALE_MIN  = 0.95
AUG_SCALE_MAX  = 1.05

# ==================== 评估参数 ====================
SG_WINDOW  = 11         # Savitzky-Golay平滑窗口
SG_POLY    = 2
MON_WEIGHT = 0.5        # 综合评分权重
TREND_WEIGHT = 0.3
ROB_WEIGHT   = 0.1
SNR_WEIGHT   = 0.1

# ==================== 可视化 ====================
FIGSIZE = (12, 8)
DPI     = 150
os.makedirs(OUTPUT_ROOT, exist_ok=True)
