"""
可视化单指标 vs 融合方法的对比
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def main():
    # 数据
    methods = ['RMS', 'Envelope\nPeak', 'BPFO\nAmplitude', '等权重\n融合', '手动平衡\n融合', '严格优化\n(BPFO≥0.15)', '严格优化\n(无约束)']

    # 评估指标
    monotonicity = [0.623, 0.566, 0.541, 0.541, 0.557, 0.557, 0.574]
    trendability = [0.932, 0.932, 0.874, 0.916, 0.924, 0.926, 0.935]
    robustness = [0.025, 0.469, 0.248, 0.318, 0.361, 0.368, 0.456]
    composite_score = [0.687, 0.703, 0.645, 0.642, 0.657, 0.661, 0.706]

    # 分类：单指标 vs 融合方法
    colors = ['#90CAF9', '#90CAF9', '#90CAF9', '#A5D6A7', '#A5D6A7', '#A5D6A7', '#66BB6A']

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: Monotonicity
    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, monotonicity, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Monotonicity', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.4, 0.7)
    ax1.set_title('单调性对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.566, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='最佳单指标')

    # 标注数值
    for bar, val in zip(bars1, monotonicity):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.legend(loc='upper left')

    # 子图2: Trendability
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, trendability, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Trendability', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.85, 0.95)
    ax2.set_title('趋势性对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.932, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='最佳单指标')

    for bar, val in zip(bars2, trendability):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax2.legend(loc='lower left')

    # 子图3: Robustness
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, robustness, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Robustness', fontsize=12, fontweight='bold')
    ax3.set_ylim(0.0, 0.5)
    ax3.set_title('鲁棒性对比', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0.469, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='最佳单指标')

    for bar, val in zip(bars3, robustness):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax3.legend(loc='upper left')

    # 子图4: Composite Score
    ax4 = axes[1, 1]
    bars4 = ax4.bar(methods, composite_score, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('综合评分', fontsize=12, fontweight='bold')
    ax4.set_ylim(0.6, 0.72)
    ax4.set_title('综合评分对比 (0.5×Mon + 0.4×Trend + 0.1×Rob)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0.703, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='最佳单指标')

    for bar, val in zip(bars4, composite_score):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.legend(loc='lower left')

    # 添加图例说明
    fig.text(0.5, 0.02, '蓝色: 单指标 | 浅绿色: 融合方法 | 深绿色: 最优融合',
             ha='center', fontsize=11, style='italic', color='#555')

    plt.suptitle('单指标 vs 融合方法性能对比 (Bearing1_1)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_path = "c:/Users/53031/Desktop/new-exp/results/exp1/single_vs_fusion_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存: {output_path}")
    plt.close()

    # 生成第二张图：雷达图对比
    fig2, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    categories = ['Monotonicity', 'Trendability', 'Robustness']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 选择关键方法进行对比
    selected_methods = {
        'RMS': ([0.623, 0.932, 0.025], '#1976D2'),
        'Envelope Peak': ([0.566, 0.932, 0.469], '#0288D1'),
        'BPFO Amplitude': ([0.541, 0.874, 0.248], '#0097A7'),
        '等权重融合': ([0.541, 0.916, 0.318], '#66BB6A'),
        '严格优化(无约束)': ([0.574, 0.935, 0.456], '#2E7D32')
    }

    for method_name, (values, color) in selected_methods.items():
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, 'o-', linewidth=2.5, label=method_name, color=color)
        ax.fill(angles, values_plot, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('单指标 vs 融合方法 - 雷达图对比', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    output_path2 = "c:/Users/53031/Desktop/new-exp/results/exp1/single_vs_fusion_radar.png"
    plt.tight_layout()
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"雷达图已保存: {output_path2}")
    plt.close()

    # 生成第三张图：表格形式
    fig3, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    # 原始数据（用于找最大值）
    data_values = {
        'mon': [0.623, 0.566, 0.541, 0.541, 0.557, 0.557, 0.574],
        'trend': [0.932, 0.932, 0.874, 0.916, 0.924, 0.926, 0.935],
        'rob': [0.025, 0.469, 0.248, 0.318, 0.361, 0.368, 0.456],
        'score': [0.687, 0.703, 0.645, 0.642, 0.657, 0.661, 0.706]
    }

    # 找每列最大值的索引
    max_mon_idx = data_values['mon'].index(max(data_values['mon']))
    max_trend_idx = data_values['trend'].index(max(data_values['trend']))
    max_rob_idx = data_values['rob'].index(max(data_values['rob']))
    max_score_idx = data_values['score'].index(max(data_values['score']))

    methods_list = ['RMS', 'Envelope Peak', 'BPFO Amplitude', '等权重融合', '手动平衡融合', '严格优化(BPFO≥0.15)', '严格优化(无约束)']
    types_list = ['单指标', '单指标', '单指标', '融合方法', '融合方法', '融合方法', '融合方法']

    table_data = [['方法', 'Monotonicity', 'Trendability', 'Robustness', '综合评分', '类型']]

    for i, method in enumerate(methods_list):
        row = [
            method,
            f"{data_values['mon'][i]:.3f}" + (' ★' if i == max_mon_idx else ''),
            f"{data_values['trend'][i]:.3f}" + (' ★' if i == max_trend_idx else ''),
            f"{data_values['rob'][i]:.3f}" + (' ★' if i == max_rob_idx else ''),
            f"{data_values['score'][i]:.3f}" + (' ★' if i == max_score_idx else ''),
            types_list[i]
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.22, 0.14, 0.14, 0.14, 0.14, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    # 设置表头样式
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 高亮单指标最优
    table[(2, 4)].set_facecolor('#BBDEFB')  # Envelope Peak 综合评分

    # 高亮融合方法最优
    table[(7, 0)].set_facecolor('#C8E6C9')
    table[(7, 1)].set_facecolor('#C8E6C9')
    table[(7, 2)].set_facecolor('#C8E6C9')
    table[(7, 3)].set_facecolor('#C8E6C9')
    table[(7, 4)].set_facecolor('#A5D6A7')
    table[(7, 5)].set_facecolor('#C8E6C9')

    plt.title('单指标 vs 融合方法详细对比表 (★ 表示该列最大值)',
              fontsize=14, fontweight='bold', pad=15)

    output_path3 = "c:/Users/53031/Desktop/new-exp/results/exp1/single_vs_fusion_table.png"
    plt.tight_layout()
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"对比表格已保存: {output_path3}")
    plt.close()


if __name__ == '__main__':
    main()
