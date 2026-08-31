"""
门控机制可视化脚本

使用方法：
    python visualize_gates.py

功能：
1. 加载训练好的模型
2. 提取测试集的门控激活值
3. 生成4种可视化图表：
   - 门控激活值热力图
   - 门控-RUL演化曲线
   - 健康期vs故障期对比
   - 门控统计分布
"""

import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import BearingRULModel
from src.gate_visualizer import GateVisualizer


def main():
    print("="*60)
    print("🔍 门控机制可视化分析")
    print("="*60)

    # 1. 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 使用设备: {device}")

    model_path = "./results/checkpoints/best_model.pth"
    data_dir = "./data/processed"

    # 2. 加载数据
    print(f"\n📂 加载测试数据...")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    Y_test = np.load(os.path.join(data_dir, "Y_test.npy"))
    print(f"   测试集形状: X={X_test.shape}, Y={Y_test.shape}")
    print(f"   Y_test分布: min={Y_test.min():.3f}, max={Y_test.max():.3f}, mean={Y_test.mean():.3f}")
    print(f"   Y<0.5样本数: {(Y_test < 0.5).sum()}, Y<0.2样本数: {(Y_test < 0.2).sum()}")

    # 3. 加载模型
    print(f"\n🤖 加载模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在！")
        print(f"   请先运行训练: python main.py --mode train")
        return

    model = BearingRULModel(input_features=7, seq_len=30, hidden_dim=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("   ✅ 模型加载成功")

    # 4. 创建可视化器
    print(f"\n🎨 初始化可视化器...")
    visualizer = GateVisualizer(model, device, save_dir='./results/gates')

    # 5. 生成可视化
    print(f"\n" + "="*60)
    print("开始生成可视化图表...")
    print("="*60)

    # 5.1 门控激活值热力图（选择几个代表性样本）
    print(f"\n[1/4] 生成门控激活值热力图...")
    # 按分位数选择健康期、中期、故障期代表样本，避免固定阈值导致空集合
    p80 = np.percentile(Y_test, 80)
    p50 = np.percentile(Y_test, 50)
    p20 = np.percentile(Y_test, 20)
    print(f"   分位阈值: p80={p80:.3f}, p50={p50:.3f}, p20={p20:.3f}")

    healthy_candidates = np.where(Y_test >= p80)[0]
    mid_candidates = np.where((Y_test >= p50) & (Y_test < p80))[0]
    fault_candidates = np.where(Y_test <= p20)[0]

    healthy_idx = healthy_candidates[0] if len(healthy_candidates) > 0 else 0
    mid_idx = mid_candidates[0] if len(mid_candidates) > 0 else len(Y_test)//2
    fault_idx = fault_candidates[0] if len(fault_candidates) > 0 else len(Y_test)-1

    visualizer.plot_gate_heatmap(X_test, sample_idx=healthy_idx,
                                 save_name='gate_heatmap_healthy.png')
    visualizer.plot_gate_heatmap(X_test, sample_idx=mid_idx,
                                 save_name='gate_heatmap_mid.png')
    visualizer.plot_gate_heatmap(X_test, sample_idx=fault_idx,
                                 save_name='gate_heatmap_fault.png')

    # 5.2 门控-RUL演化曲线
    print(f"\n[2/4] 生成门控-RUL演化曲线...")
    visualizer.plot_gate_evolution(X_test, Y_test, n_samples=500)

    # 5.3 健康期vs故障期对比
    print(f"\n[3/4] 生成健康期vs故障期对比图...")
    visualizer.plot_gate_comparison(X_test, Y_test, threshold=0.5)

    # 5.4 门控统计分布
    print(f"\n[4/4] 生成门控统计分布图...")
    visualizer.plot_gate_statistics(X_test, Y_test)

    print(f"\n" + "="*60)
    print("✅ 所有可视化图表生成完成！")
    print(f"📁 保存位置: ./results/gates/")
    print("="*60)

    # 6. 生成分析报告
    print(f"\n📊 生成分析报告...")
    generate_analysis_report(X_test, Y_test, model, device)


def generate_analysis_report(X_test, Y_test, model, device):
    """
    生成门控分析报告
    """
    from scipy.stats import spearmanr

    print("\n" + "="*60)
    print("📈 门控机制分析报告")
    print("="*60)

    # 提取门控值
    visualizer = GateVisualizer(model, device)
    reset_gates, update_gates = visualizer.extract_gates(X_test[:200])

    # 计算统计量
    reset_mean = reset_gates.mean(axis=(1, 2))
    update_mean = update_gates.mean(axis=(1, 2))

    # 相关性分析（防止常数数组导致 NaN）
    y_slice = Y_test[:200]

    if len(np.unique(y_slice)) > 1 and len(np.unique(reset_mean)) > 1:
        reset_corr, reset_p = spearmanr(y_slice, reset_mean)
    else:
        reset_corr, reset_p = np.nan, np.nan
        print("   ⚠️ 重置门或RUL取值缺乏变化，相关性不可计算")

    if len(np.unique(y_slice)) > 1 and len(np.unique(update_mean)) > 1:
        update_corr, update_p = spearmanr(y_slice, update_mean)
    else:
        update_corr, update_p = np.nan, np.nan
        print("   ⚠️ 更新门或RUL取值缺乏变化，相关性不可计算")

    print(f"\n1. 门控-RUL相关性分析:")
    print(f"   重置门 vs RUL: 相关系数={reset_corr:.4f}, p值={reset_p:.4e}")
    print(f"   更新门 vs RUL: 相关系数={update_corr:.4f}, p值={update_p:.4e}")

    print(f"\n2. 门控激活值统计:")
    print(f"   重置门: 均值={reset_mean.mean():.4f}, 标准差={reset_mean.std():.4f}")
    print(f"   更新门: 均值={update_mean.mean():.4f}, 标准差={update_mean.std():.4f}")

    # 健康期vs故障期对比（用中位数动态分层，避免空集合）
    split = np.percentile(Y_test[:200], 50)
    healthy_mask = Y_test[:200] > split
    fault_mask = Y_test[:200] <= split
    print(f"   动态分层阈值(中位数): {split:.3f}，健康样本={healthy_mask.sum()}，故障样本={fault_mask.sum()}")

    reset_healthy = reset_mean[healthy_mask].mean() if healthy_mask.any() else np.nan
    reset_fault = reset_mean[fault_mask].mean() if fault_mask.any() else np.nan
    update_healthy = update_mean[healthy_mask].mean() if healthy_mask.any() else np.nan
    update_fault = update_mean[fault_mask].mean() if fault_mask.any() else np.nan

    print(f"\n3. 健康期vs故障期对比:")
    print(f"   重置门: 健康期={reset_healthy:.4f}, 故障期={reset_fault:.4f}, 差异={abs(reset_healthy-reset_fault):.4f}")
    print(f"   更新门: 健康期={update_healthy:.4f}, 故障期={update_fault:.4f}, 差异={abs(update_healthy-update_fault):.4f}")

    print(f"\n4. 物理一致性检验:")
    if update_corr < 0:
        print(f"   ✅ 更新门与RUL负相关，符合物理直觉（退化时遗忘健康状态）")
    else:
        print(f"   ⚠️  更新门与RUL正相关，不符合预期")

    if reset_fault > reset_healthy:
        print(f"   ✅ 故障期重置门更活跃，符合物理直觉（捕捉新故障模式）")
    else:
        print(f"   ⚠️  故障期重置门不够活跃，可能需要调整")

    print("="*60)


if __name__ == "__main__":
    main()
