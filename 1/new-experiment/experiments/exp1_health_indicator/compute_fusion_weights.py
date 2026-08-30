"""
计算选定特征的融合权重
支持多种权重计算方法，包括线性规划最优解
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from hi_evaluation import HealthIndicatorEvaluator


def compute_entropy_weight(evaluation_matrix: np.ndarray) -> np.ndarray:
    """
    熵权法计算权重

    Args:
        evaluation_matrix: 评估矩阵，shape=(n_features, n_metrics)
                          例如：[[mon1, trend1, rob1],
                                [mon2, trend2, rob2],
                                [mon3, trend3, rob3]]

    Returns:
        权重数组，shape=(n_features,)
    """
    n_features, n_metrics = evaluation_matrix.shape

    # Step 1: 归一化（按列）
    col_sum = evaluation_matrix.sum(axis=0)
    col_sum[col_sum == 0] = 1.0  # 避免除零
    normalized = evaluation_matrix / col_sum

    # Step 2: 计算信息熵
    k = 1.0 / np.log(n_metrics) if n_metrics > 1 else 1.0
    normalized_safe = np.where(normalized > 0, normalized, 1e-10)  # 避免log(0)
    entropy = -k * np.sum(normalized_safe * np.log(normalized_safe), axis=1)

    # Step 3: 计算权重
    diversity = 1 - entropy  # 信息效用值
    weights = diversity / diversity.sum()

    return weights


def compute_score_based_weight(scores: np.ndarray) -> np.ndarray:
    """
    基于综合评分的权重

    Args:
        scores: 综合评分数组

    Returns:
        权重数组
    """
    return scores / scores.sum()


def compute_single_metric_weight(metric_values: np.ndarray) -> np.ndarray:
    """
    基于单一评估指标的权重

    Args:
        metric_values: 单一指标值数组（如Trendability）

    Returns:
        权重数组
    """
    return metric_values / metric_values.sum()


def compute_combined_weight(evaluation_matrix: np.ndarray,
                           composite_scores: np.ndarray,
                           alpha: float = 0.7) -> np.ndarray:
    """
    组合权重：熵权法 + 综合评分

    Args:
        evaluation_matrix: 评估矩阵
        composite_scores: 综合评分
        alpha: 熵权法的权重（0-1），1-alpha为综合评分的权重

    Returns:
        组合权重
    """
    w_entropy = compute_entropy_weight(evaluation_matrix)
    w_score = compute_score_based_weight(composite_scores)

    # 组合
    combined_weights = alpha * w_entropy + (1 - alpha) * w_score

    # 归一化
    combined_weights = combined_weights / combined_weights.sum()

    return combined_weights


def compute_linear_programming_weight(evaluation_matrix: np.ndarray,
                                     composite_scores: np.ndarray,
                                     method: str = 'maximize_min') -> np.ndarray:
    """
    线性规划求最优权重

    Args:
        evaluation_matrix: 评估矩阵，shape=(n_features, n_metrics)
        composite_scores: 综合评分
        method: 优化目标
               'maximize_min' - 最大化最小评估分数（提升短板）
               'maximize_avg' - 最大化平均评估分数
               'maximize_weighted' - 最大化加权评估分数

    Returns:
        最优权重
    """
    n_features = len(composite_scores)

    if method == 'maximize_min':
        # 目标：最大化融合后的最小评估分数
        # max t
        # s.t. w1*mon1 + w2*mon2 + w3*mon3 >= t
        #      w1*trend1 + w2*trend2 + w3*trend3 >= t
        #      w1*rob1 + w2*rob2 + w3*rob3 >= t
        #      w1 + w2 + w3 = 1
        #      w1, w2, w3 >= 0

        # 转换为标准形式：min -t
        # 变量：[w1, w2, w3, t]
        c = np.zeros(n_features + 1)
        c[-1] = -1  # 最小化 -t，即最大化 t

        # 不等式约束：-A_ub @ x <= b_ub
        # w1*mon1 + w2*mon2 + w3*mon3 - t >= 0
        # 转换为：-w1*mon1 - w2*mon2 - w3*mon3 + t <= 0
        A_ub = []
        b_ub = []
        for i in range(evaluation_matrix.shape[1]):  # 对每个评估指标
            row = np.zeros(n_features + 1)
            row[:n_features] = -evaluation_matrix[:, i]
            row[-1] = 1
            A_ub.append(row)
            b_ub.append(0)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # 等式约束：A_eq @ x = b_eq
        # w1 + w2 + w3 = 1
        A_eq = np.zeros((1, n_features + 1))
        A_eq[0, :n_features] = 1
        b_eq = np.array([1])

        # 边界：w >= 0, t无约束
        bounds = [(0, None)] * n_features + [(None, None)]

        # 求解
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')

        if result.success:
            weights = result.x[:n_features]
            return weights / weights.sum()  # 归一化
        else:
            print(f"线性规划求解失败: {result.message}")
            return np.ones(n_features) / n_features

    elif method == 'maximize_avg':
        # 目标：最大化融合后的平均评估分数
        # max (w1*avg1 + w2*avg2 + w3*avg3)
        # 其中 avg_i = (mon_i + trend_i + rob_i) / 3

        avg_scores = evaluation_matrix.mean(axis=1)

        # 转换为最小化问题
        c = -avg_scores

        # 等式约束：w1 + w2 + w3 = 1
        A_eq = np.ones((1, n_features))
        b_eq = np.array([1])

        # 边界：w >= 0
        bounds = [(0, None)] * n_features

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            return result.x
        else:
            print(f"线性规划求解失败: {result.message}")
            return np.ones(n_features) / n_features

    elif method == 'maximize_weighted':
        # 目标：最大化加权综合评分
        # max (w1*score1 + w2*score2 + w3*score3)

        c = -composite_scores

        A_eq = np.ones((1, n_features))
        b_eq = np.array([1])

        bounds = [(0, None)] * n_features

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            return result.x
        else:
            print(f"线性规划求解失败: {result.message}")
            return np.ones(n_features) / n_features

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_nonlinear_optimization_weight(evaluation_matrix: np.ndarray,
                                         composite_scores: np.ndarray,
                                         w_mon: float = 0.5,
                                         w_trend: float = 0.4,
                                         w_rob: float = 0.1) -> np.ndarray:
    """
    非线性优化：最大化融合后的综合评估分数

    目标：max f(w) = Mon(融合HI) * w_mon + Trend(融合HI) * w_trend + Rob(融合HI) * w_rob

    Args:
        evaluation_matrix: 评估矩阵
        composite_scores: 综合评分
        w_mon: 单调性权重（默认0.5）
        w_trend: 趋势性权重（默认0.4）
        w_rob: 鲁棒性权重（默认0.1）

    Returns:
        最优权重
    """
    n_features = evaluation_matrix.shape[0]

    # 目标函数：融合后的综合评分
    def objective(w):
        # 融合后的评估分数
        fused_scores = evaluation_matrix.T @ w  # shape=(3,)
        # 综合评分
        composite = w_mon * fused_scores[0] + w_trend * fused_scores[1] + w_rob * fused_scores[2]
        return -composite  # 最小化负值 = 最大化

    # 约束：权重和为1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # 边界：0 <= w <= 1
    bounds = [(0, 1)] * n_features

    # 初始值：等权重
    w0 = np.ones(n_features) / n_features

    # 求解
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        print(f"非线性优化求解失败: {result.message}")
        return np.ones(n_features) / n_features


def analyze_selected_features(feature_names: list, df_evaluation: pd.DataFrame):
    """
    分析选定特征并计算多种权重

    Args:
        feature_names: 选定的特征名称列表
        df_evaluation: 特征评估结果DataFrame
    """
    print("=" * 80)
    print("选定特征的权重计算")
    print("=" * 80)

    # 提取选定特征的评估数据
    selected_data = df_evaluation[df_evaluation['feature_name'].isin(feature_names)]

    print(f"\n选定特征: {', '.join(feature_names)}")
    print("\n评估分数:")
    print("-" * 80)
    print(f"{'特征名称':<20} {'单调性':<12} {'趋势性':<12} {'鲁棒性':<12} {'综合评分':<12}")
    print("-" * 80)

    evaluation_matrix = []
    composite_scores = []
    trendability_values = []

    for _, row in selected_data.iterrows():
        feature_name = row['feature_name']
        mon = row['monotonicity']
        trend = row['trendability']
        rob = row['robustness']
        comp = row['composite_score']

        print(f"{feature_name:<20} {mon:<12.4f} {trend:<12.4f} {rob:<12.4f} {comp:<12.4f}")

        evaluation_matrix.append([mon, trend, rob])
        composite_scores.append(comp)
        trendability_values.append(trend)

    evaluation_matrix = np.array(evaluation_matrix)
    composite_scores = np.array(composite_scores)
    trendability_values = np.array(trendability_values)

    # 计算各种权重
    print("\n\n权重计算结果:")
    print("=" * 80)

    # 方法1: 熵权法
    w_entropy = compute_entropy_weight(evaluation_matrix)
    print("\n方法1: 熵权法（基于评估矩阵）")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {w_entropy[i]:.4f} ({w_entropy[i]*100:.2f}%)")

    # 方法2: 综合评分加权
    w_score = compute_score_based_weight(composite_scores)
    print("\n方法2: 综合评分加权")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {w_score[i]:.4f} ({w_score[i]*100:.2f}%)")

    # 方法3: 趋势性加权
    w_trend = compute_single_metric_weight(trendability_values)
    print("\n方法3: 趋势性加权（只考虑Trendability）")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {w_trend[i]:.4f} ({w_trend[i]*100:.2f}%)")

    # 方法4: 等权重
    w_equal = np.ones(len(feature_names)) / len(feature_names)
    print("\n方法4: 等权重")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {w_equal[i]:.4f} ({w_equal[i]*100:.2f}%)")

    # 方法5: 组合权重（推荐）
    w_combined = compute_combined_weight(evaluation_matrix, composite_scores, alpha=0.7)
    print("\n方法5: 组合权重 - 70%熵权法 + 30%综合评分")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {w_combined[i]:.4f} ({w_combined[i]*100:.2f}%)")

    # 方法6: 线性规划 - 最大化最小评估分数
    w_lp_min = compute_linear_programming_weight(evaluation_matrix, composite_scores, method='maximize_min')
    print("\n方法6: 线性规划 - 最大化最小评估分数（提升短板）")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {w_lp_min[i]:.4f} ({w_lp_min[i]*100:.2f}%)")

    # 方法7: 线性规划 - 最大化平均评估分数
    w_lp_avg = compute_linear_programming_weight(evaluation_matrix, composite_scores, method='maximize_avg')
    print("\n方法7: 线性规划 - 最大化平均评估分数")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {w_lp_avg[i]:.4f} ({w_lp_avg[i]*100:.2f}%)")

    # 方法8: 线性规划 - 最大化加权综合评分
    w_lp_weighted = compute_linear_programming_weight(evaluation_matrix, composite_scores, method='maximize_weighted')
    print("\n方法8: 线性规划 - 最大化加权综合评分")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {w_lp_weighted[i]:.4f} ({w_lp_weighted[i]*100:.2f}%)")

    # 方法9: 非线性优化 - 最大化融合后的综合评估分数
    w_nonlinear = compute_nonlinear_optimization_weight(evaluation_matrix, composite_scores,
                                                        w_mon=0.5, w_trend=0.4, w_rob=0.1)
    print("\n方法9: 非线性优化 - 最大化融合后的综合评估分数 ★★ 推荐")
    print("       (权重: Mon=0.5, Trend=0.4, Rob=0.1)")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {w_nonlinear[i]:.4f} ({w_nonlinear[i]*100:.2f}%)")

    # 计算每种方法融合后的评估分数
    print("\n\n各方法融合后的评估分数对比:")
    print("=" * 80)
    print(f"{'方法':<30} {'单调性':<12} {'趋势性':<12} {'鲁棒性':<12} {'综合评分':<12}")
    print("-" * 80)

    methods = {
        '熵权法': w_entropy,
        '综合评分加权': w_score,
        '趋势性加权': w_trend,
        '等权重': w_equal,
        '组合权重': w_combined,
        'LP-最大化最小分数': w_lp_min,
        'LP-最大化平均分数': w_lp_avg,
        'LP-最大化加权分数': w_lp_weighted,
        '非线性优化': w_nonlinear
    }

    best_composite = 0
    best_method = None

    for method_name, weights in methods.items():
        fused_scores = evaluation_matrix.T @ weights
        # 使用新的权重计算综合评分
        fused_composite = 0.5 * fused_scores[0] + 0.4 * fused_scores[1] + 0.1 * fused_scores[2]

        marker = ""
        if fused_composite > best_composite:
            best_composite = fused_composite
            best_method = method_name
            marker = " ★"

        print(f"{method_name:<30} {fused_scores[0]:<12.4f} {fused_scores[1]:<12.4f} {fused_scores[2]:<12.4f} {fused_composite:<12.4f}{marker}")

    print("\n" + "=" * 80)
    print(f"最优方法: {best_method}（综合评分: {best_composite:.4f}）")
    print("=" * 80)

    return {
        'entropy': w_entropy,
        'score': w_score,
        'trendability': w_trend,
        'equal': w_equal,
        'combined': w_combined,
        'lp_min': w_lp_min,
        'lp_avg': w_lp_avg,
        'lp_weighted': w_lp_weighted,
        'nonlinear': w_nonlinear
    }


if __name__ == '__main__':
    # 读取评估结果
    df = pd.read_csv('../../results/exp1/Bearing1_1_feature_evaluation.csv')

    # 选定的特征
    selected_features = ['rms', 'envelope_peak', 'BPFO_amplitude']

    # 分析并计算权重
    weights = analyze_selected_features(selected_features, df)

    print("\n\n建议:")
    print("- 如果追求客观性，使用熵权法")
    print("- 如果追求简单性，使用综合评分加权")
    print("- 如果追求平衡性，使用组合权重")
    print("- 如果追求最优性，使用非线性优化 ★★ 强烈推荐")

