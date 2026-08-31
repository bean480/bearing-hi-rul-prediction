"""
门控机制高级可视化分析
包含：聚类分析、特征相关性、单轴承轨迹
"""
import torch
import numpy as np
import yaml
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import BearingRULModel
from gate_visualizer import GateVisualizer

def load_config():
    with open('./config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("="*70)
    print("门控机制高级可视化分析")
    print("="*70)

    # 加载配置
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载测试数据...")
    X_test = np.load('./data/processed/X_test.npy')
    Y_test = np.load('./data/processed/Y_test.npy')
    print(f"测试集: X_test={X_test.shape}, Y_test={Y_test.shape}")

    # 加载模型
    print("\n加载训练好的模型...")
    model = BearingRULModel(
        input_features=X_test.shape[2],  # 特征数
        seq_len=X_test.shape[1]          # 序列长度
    ).to(device)

    checkpoint = torch.load('./results/checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ 模型加载成功")

    # 创建可视化器
    visualizer = GateVisualizer(model, device, save_dir='./results/gates')

    # ========== 1. 门控模式聚类分析 ==========
    print("\n" + "="*70)
    print("1. 门控激活模式聚类分析")
    print("="*70)

    # 使用全部测试集进行聚类
    clusters, cluster_info = visualizer.analyze_gate_patterns(
        X_test, Y_test, n_clusters=3, save_name='gate_patterns.png'
    )

    # ========== 2. 门控与振动特征的相关性分析 ==========
    print("\n" + "="*70)
    print("2. 门控值与振动特征的相关性分析")
    print("="*70)

    # 定义特征名称（根据你的实际特征）
    feature_names = [
        'RMS', 'Peak', 'Kurtosis', 'Skewness', 'Crest_Factor',
        'Shape_Factor', 'Impulse_Factor', 'Clearance_Factor',
        'Freq_Mean', 'Freq_Std', 'Freq_Peak', 'Freq_Energy'
    ]

    # 如果特征数量不匹配，使用默认名称
    if len(feature_names) != X_test.shape[2]:
        feature_names = None

    correlations = visualizer.correlate_gates_with_features(
        X_test, Y_test, feature_names=feature_names
    )

    # ========== 3. 单个轴承的生命周期轨迹 ==========
    print("\n" + "="*70)
    print("3. 单个轴承生命周期轨迹可视化")
    print("="*70)

    # 选择一个从健康到故障的完整样本序列
    # 找到RUL从高到低的连续样本
    print("\n寻找完整的退化序列...")

    # 方法：找到RUL单调递减的最长子序列
    best_start = 0
    best_length = 0

    for start in range(len(Y_test) - 50):
        # 检查从start开始的单调性
        length = 1
        for i in range(start, len(Y_test) - 1):
            if Y_test[i] >= Y_test[i + 1]:  # 单调递减或相等
                length += 1
            else:
                break

        if length > best_length:
            best_length = length
            best_start = start

    print(f"找到最长退化序列: 起始索引={best_start}, 长度={best_length}")
    print(f"RUL范围: {Y_test[best_start]:.3f} → {Y_test[best_start + best_length - 1]:.3f}")

    # 提取这段数据
    bearing_data = X_test[best_start:best_start + best_length]
    bearing_rul = Y_test[best_start:best_start + best_length]

    visualizer.plot_single_bearing_trajectory(
        bearing_data, bearing_rul, save_name='single_bearing_trajectory.png'
    )

    print("\n" + "="*70)
    print("✅ 所有高级可视化分析完成！")
    print("="*70)
    print("\n生成的图片:")
    print("  1. ./results/gates/gate_patterns.png - 门控模式聚类分析")
    print("  2. ./results/gates/single_bearing_trajectory.png - 单轴承生命周期轨迹")
    print("\n相关性分析结果已输出到控制台")

if __name__ == '__main__':
    main()

