"""
生成简洁的严格优化方法流程图（纯流程，公式清晰）
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams
import matplotlib.patches as mpatches

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def add_box(ax, x, y, w, h, text, fc="#E3F2FD", ec="#1565C0", fontsize=12, lw=2, alpha=0.8):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, multialignment='center')


def add_arrow(ax, x1, y1, x2, y2, label="", fontsize=11):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=30, lw=2.8, color='#424242'
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2 + 0.15
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=fontsize,
                color='#C62828', fontweight='bold')


def main():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 标题
    ax.text(0.5, 0.96, '严格优化的健康指标融合方法', fontsize=22, fontweight='bold',
            ha='center', va='top')

    # 步骤1: 输入
    y1 = 0.85
    ax.text(0.5, y1 + 0.03, '输入', fontsize=14, ha='center', fontweight='bold', color='#2E7D32')
    ax.text(0.5, y1 - 0.02, r'候选健康指标序列', fontsize=13, ha='center')
    ax.text(0.5, y1 - 0.06, r'$\mathbf{HI}_1(t), \; \mathbf{HI}_2(t), \; \mathbf{HI}_3(t)$',
            fontsize=15, ha='center', style='italic')
    ax.text(0.5, y1 - 0.10, '(RMS, Envelope Peak, BPFO Amplitude)', fontsize=11, ha='center', color='#555')

    # 箭头1
    add_arrow(ax, 0.5, y1 - 0.13, 0.5, 0.66, label='优化求解')

    # 步骤2: 优化问题
    y2 = 0.52
    ax.text(0.5, y2 + 0.12, '优化问题', fontsize=14, ha='center', fontweight='bold', color='#1565C0')

    # 分行显示优化公式
    ax.text(0.5, y2 + 0.06, r'$\mathbf{w}^* = \arg\max_{\mathbf{w}} \; \mathrm{Score}\left(\sum_{i=1}^{3} w_i \cdot \mathbf{HI}_i(t)\right)$',
            fontsize=16, ha='center')

    ax.text(0.5, y2 - 0.01, r'约束条件:', fontsize=12, ha='center', color='#555')
    ax.text(0.5, y2 - 0.06, r'$\sum_{i=1}^{3} w_i = 1, \quad w_i \geq 0$',
            fontsize=14, ha='center')

    # 箭头2
    add_arrow(ax, 0.5, y2 - 0.10, 0.5, 0.35, label='迭代计算')

    # 步骤3: 目标函数
    y3 = 0.18
    ax.text(0.5, y3 + 0.15, '目标函数（每次迭代）', fontsize=14, ha='center', fontweight='bold', color='#EF6C00')

    # 融合公式
    ax.text(0.5, y3 + 0.09, r'$\mathbf{HI}_{\mathrm{fused}}(t) = \sum_{i=1}^{3} w_i \cdot \mathbf{HI}_i(t)$',
            fontsize=15, ha='center')

    # 评估指标
    ax.text(0.5, y3 + 0.03, r'重新计算评估指标:', fontsize=12, ha='center', color='#555')
    ax.text(0.5, y3 - 0.02, r'$\mathrm{Mon}(\mathbf{HI}_{\mathrm{fused}}), \; \mathrm{Trend}(\mathbf{HI}_{\mathrm{fused}}), \; \mathrm{Rob}(\mathbf{HI}_{\mathrm{fused}})$',
            fontsize=13, ha='center')

    # 综合评分
    ax.text(0.5, y3 - 0.08, r'综合评分:', fontsize=12, ha='center', color='#555')
    ax.text(0.5, y3 - 0.13, r'$\mathrm{Score} = 0.5 \cdot \mathrm{Mon} + 0.4 \cdot \mathrm{Trend} + 0.1 \cdot \mathrm{Rob}$',
            fontsize=14, ha='center')

    # 箭头3
    add_arrow(ax, 0.5, y3 - 0.17, 0.5, 0.00, label='输出')

    # 步骤4: 输出
    y4 = -0.08
    ax.text(0.5, y4 + 0.06, '输出', fontsize=14, ha='center', fontweight='bold', color='#00838F')
    ax.text(0.5, y4 + 0.01, r'最优融合权重:', fontsize=12, ha='center', color='#555')
    ax.text(0.5, y4 - 0.04, r'$\mathbf{w}^* = [w_{\text{rms}}^*, \; w_{\text{env}}^*, \; w_{\text{bpfo}}^*]$',
            fontsize=15, ha='center')
    ax.text(0.5, y4 - 0.09, r'复合健康指标:', fontsize=12, ha='center', color='#555')
    ax.text(0.5, y4 - 0.14, r'$\mathbf{CHI}(t) = \sum_{i=1}^{3} w_i^* \cdot \mathbf{HI}_i(t)$',
            fontsize=15, ha='center')

    # 底部说明（浅色背景框）
    y_note = -0.26
    note_box = mpatches.FancyBboxPatch(
        (0.05, y_note - 0.02), 0.9, 0.10,
        boxstyle="round,pad=0.01",
        linewidth=2,
        edgecolor='#D32F2F',
        facecolor='#FFEBEE',
        alpha=0.7
    )
    ax.add_patch(note_box)


    ax.text(0.5, y_note - 0.01, r'严格优化: 先融合序列，再重新计算评估指标',
            fontsize=11, ha='center', color='#333')
    ax.text(0.5, y_note - 0.05, r'传统方法: 先评估单特征，再对分数做线性组合（假设 $\text{Mon}(\sum w_i \mathbf{HI}_i) \approx \sum w_i \text{Mon}(\mathbf{HI}_i)$，不严格）',
            fontsize=10, ha='center', color='#666')

    plt.tight_layout()
    output_path = "c:/Users/53031/Desktop/new-exp/results/exp1/strict_optimization_flowchart_clean.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"简洁流程图已保存: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
