"""
可视化严格优化的融合权重结果
每张图和表格单独输出
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from data_loader import XJTUDataLoader
from feature_extraction import FeatureExtractor
from hi_evaluation import HealthIndicatorEvaluator
from optimize_fusion_weights_strict import optimize_weights_scipy
from config import DATA_PATHS, get_fault_frequencies

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def normalize_hi(hi):
    return (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)


def build_fused_hi(feature_series_dict, weights):
    names = list(feature_series_dict.keys())
    fused = np.zeros_like(feature_series_dict[names[0]])
    for i, name in enumerate(names):
        fused += weights[i] * feature_series_dict[name]
    return fused


def evaluate_with_score(evaluator, hi):
    scores = evaluator.evaluate(hi)
    composite = 0.5 * scores['monotonicity'] + 0.4 * scores['trendability'] + 0.1 * scores['robustness']
    return scores, composite


def visualize_optimization_results(bearing_name='Bearing1_1', working_condition=1):
    loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])
    data_list, _ = loader.load_bearing_data_by_file(bearing_name)
    print(f"加载 {bearing_name}: {len(data_list)} 个文件")

    extractor = FeatureExtractor(sampling_rate=25600)
    fault_freqs = get_fault_frequencies(working_condition)

    rms_series, envelope_peak_series, bpfo_amplitude_series = [], [], []
    for signal in data_list:
        features = extractor.extract_all_features(signal, fault_freqs, highpass_cutoff=1000)
        rms_series.append(features['rms'])
        envelope_peak_series.append(features['envelope_peak'])
        bpfo_amplitude_series.append(features['BPFO_amplitude'])

    feature_series_dict = {
        'rms': np.array(rms_series),
        'envelope_peak': np.array(envelope_peak_series),
        'BPFO_amplitude': np.array(bpfo_amplitude_series)
    }

    evaluator = HealthIndicatorEvaluator()

    print("\n计算严格优化权重（无约束）...")
    weights_strict, _, _ = optimize_weights_scipy(feature_series_dict, evaluator)

    print("计算严格优化权重（BPFO>=0.15）...")
    weights_constrained, _, _ = optimize_weights_scipy(feature_series_dict, evaluator, min_weight=0.15)

    weights_manual = np.array([0.30, 0.50, 0.20])
    weights_equal = np.array([1 / 3, 1 / 3, 1 / 3])

    methods = ['严格优化（无约束）', '严格优化（BPFO≥0.15）', '手动平衡', '等权重']
    weights_map = {
        methods[0]: weights_strict,
        methods[1]: weights_constrained,
        methods[2]: weights_manual,
        methods[3]: weights_equal
    }

    hi_map = {k: build_fused_hi(feature_series_dict, v) for k, v in weights_map.items()}

    eval_map = {}
    score_map = {}
    for k, hi in hi_map.items():
        s, c = evaluate_with_score(evaluator, hi)
        eval_map[k] = s
        score_map[k] = c

    x = np.arange(len(next(iter(hi_map.values()))))

    output_root = DATA_PATHS['output_root']

    # 1) 融合HI曲线
    plt.figure(figsize=(10, 5))
    for name, color in zip(methods, ['b', 'r', 'g', 'm']):
        plt.plot(x, normalize_hi(hi_map[name]), color=color, linewidth=2, label=name)
    plt.xlabel('文件编号')
    plt.ylabel('归一化HI')
    plt.title(f'{bearing_name} - 融合HI曲线对比')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    p1 = f"{output_root}/strict_hi_curves.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()

    # 2) 单个特征曲线
    plt.figure(figsize=(10, 5))
    plt.plot(x, normalize_hi(feature_series_dict['rms']), 'c-', linewidth=1.8, label='rms', alpha=0.8)
    plt.plot(x, normalize_hi(feature_series_dict['envelope_peak']), color='orange', linewidth=1.8, label='envelope_peak', alpha=0.8)
    plt.plot(x, normalize_hi(feature_series_dict['BPFO_amplitude']), color='purple', linewidth=1.8, label='BPFO_amplitude', alpha=0.8)
    plt.xlabel('文件编号')
    plt.ylabel('归一化HI')
    plt.title(f'{bearing_name} - 单个候选特征曲线')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    p2 = f"{output_root}/strict_single_features.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()

    # 3) 权重柱状图
    plt.figure(figsize=(10, 5))
    feature_names = ['rms', 'envelope_peak', 'BPFO_amplitude']
    x_pos = np.arange(len(feature_names))
    width = 0.2
    plt.bar(x_pos - 1.5 * width, weights_strict, width, label=methods[0], color='b', alpha=0.75)
    plt.bar(x_pos - 0.5 * width, weights_constrained, width, label=methods[1], color='r', alpha=0.75)
    plt.bar(x_pos + 0.5 * width, weights_manual, width, label=methods[2], color='g', alpha=0.75)
    plt.bar(x_pos + 1.5 * width, weights_equal, width, label=methods[3], color='m', alpha=0.75)
    plt.xticks(x_pos, feature_names, rotation=10)
    plt.ylabel('权重')
    plt.ylim(0, 0.75)
    plt.title('融合权重对比')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    p3 = f"{output_root}/strict_weight_comparison.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=300, bbox_inches='tight')
    plt.close()

    # 4) 综合评分柱状图（从0.5开始）
    plt.figure(figsize=(10, 5))
    score_vals = [score_map[m] for m in methods]
    bars = plt.bar(methods, score_vals, color=['b', 'r', 'g', 'm'], alpha=0.75)
    plt.ylabel('综合评分')
    plt.title('综合评分对比（0.5×Mon + 0.4×Trend + 0.1×Rob）')
    plt.ylim(0.5, 0.72)
    plt.grid(True, alpha=0.3, axis='y')
    for bar, s in zip(bars, score_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, s + 0.002, f'{s:.4f}', ha='center', va='bottom', fontsize=10)
    p4 = f"{output_root}/strict_composite_scores.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=300, bbox_inches='tight')
    plt.close()

    # 5) 三项指标分组柱状图
    plt.figure(figsize=(10, 5))
    metric_names = ['monotonicity', 'trendability', 'robustness']
    metric_labels = ['Monotonicity', 'Trendability', 'Robustness']
    x_m = np.arange(len(metric_names))
    w = 0.2
    for i, (method, color) in enumerate(zip(methods, ['b', 'r', 'g', 'm'])):
        vals = [eval_map[method][m] for m in metric_names]
        plt.bar(x_m + (i - 1.5) * w, vals, w, label=method, color=color, alpha=0.75)
    plt.xticks(x_m, metric_labels)
    plt.ylabel('评估分数')
    plt.ylim(0.0, 1.0)
    plt.title('Mon/Trend/Rob 对比')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    p5 = f"{output_root}/strict_metric_comparison.png"
    plt.tight_layout()
    plt.savefig(p5, dpi=300, bbox_inches='tight')
    plt.close()

    # 6) 表格单图
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis('off')
    table_data = [[
        '方案', 'Mon', 'Trend', 'Rob', '综合评分', 'w_rms', 'w_env', 'w_bpfo'
    ]]
    for m in methods:
        table_data.append([
            m,
            f"{eval_map[m]['monotonicity']:.3f}",
            f"{eval_map[m]['trendability']:.3f}",
            f"{eval_map[m]['robustness']:.3f}",
            f"{score_map[m]:.4f}",
            f"{weights_map[m][0]:.3f}",
            f"{weights_map[m][1]:.3f}",
            f"{weights_map[m][2]:.3f}",
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.24, 0.08, 0.08, 0.08, 0.12, 0.10, 0.10, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for i in range(8):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    best_idx = int(np.argmax(score_vals)) + 1
    for i in range(8):
        table[(best_idx, i)].set_facecolor('#FFF176')

    plt.title('严格优化结果对比表', pad=8)
    p6 = f"{output_root}/strict_result_table.png"
    plt.tight_layout()
    plt.savefig(p6, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n图片已保存:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
    print(f"  {p4}")
    print(f"  {p5}")
    print(f"  {p6}")


if __name__ == '__main__':
    visualize_optimization_results(bearing_name='Bearing1_1', working_condition=1)
