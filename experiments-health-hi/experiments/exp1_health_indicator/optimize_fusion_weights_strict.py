"""
严格的融合权重优化
基于原始特征序列，而不是评估矩阵
"""

import numpy as np
from scipy.optimize import minimize
from hi_evaluation import HealthIndicatorEvaluator
from typing import Dict, Tuple


def objective_function(w: np.ndarray,
                      feature_series_dict: Dict[str, np.ndarray],
                      evaluator: HealthIndicatorEvaluator,
                      w_mon: float = 0.5,
                      w_trend: float = 0.4,
                      w_rob: float = 0.1) -> float:
    """
    目标函数：给定融合权重，计算融合后HI的综合评分

    关键：这里是对融合后的HI序列重新计算Mon/Trend/Rob，
         而不是对单个特征的评分做线性组合！

    Args:
        w: 融合权重 [w1, w2, w3]
        feature_series_dict: 原始特征序列字典
        evaluator: 评估器实例
        w_mon: 单调性权重（默认0.5）
        w_trend: 趋势性权重（默认0.4）
        w_rob: 鲁棒性权重（默认0.1）

    Returns:
        综合评分（越高越好）
    """
    # Step 1: 构造融合HI序列
    hi_fused = np.zeros_like(list(feature_series_dict.values())[0])

    feature_names = list(feature_series_dict.keys())
    for i, name in enumerate(feature_names):
        hi_fused += w[i] * feature_series_dict[name]

    # Step 2: 对融合后的HI，重新计算评估指标
    scores = evaluator.evaluate(hi_fused)

    # Step 3: 计算综合评分
    composite_score = (w_mon * scores['monotonicity'] +
                      w_trend * scores['trendability'] +
                      w_rob * scores['robustness'])

    return composite_score


def optimize_weights_scipy(feature_series_dict: Dict[str, np.ndarray],
                          evaluator: HealthIndicatorEvaluator,
                          w_mon: float = 0.5,
                          w_trend: float = 0.4,
                          w_rob: float = 0.1,
                          min_weight: float = 0.0) -> Tuple[np.ndarray, float, Dict]:
    """
    使用scipy优化融合权重（严格版本）

    Args:
        feature_series_dict: 原始特征序列字典
        evaluator: 评估器实例
        w_mon: 单调性权重
        w_trend: 趋势性权重
        w_rob: 鲁棒性权重
        min_weight: 每个特征的最小权重（可选约束）

    Returns:
        最优权重, 最优综合评分, 融合后HI的详细评估分数
    """
    n_features = len(feature_series_dict)

    # 目标函数（最小化负的综合评分）
    def objective(w):
        return -objective_function(w, feature_series_dict, evaluator,
                                  w_mon, w_trend, w_rob)

    # 约束：权重和为1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # 边界：min_weight <= w <= 1
    bounds = [(min_weight, 1)] * n_features

    # 初始值：等权重
    w0 = np.ones(n_features) / n_features

    # 优化
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000})

    if result.success:
        optimal_weights = result.x
        optimal_score = -result.fun

        # 计算融合后HI的详细评估分数
        hi_fused = np.zeros_like(list(feature_series_dict.values())[0])
        feature_names = list(feature_series_dict.keys())
        for i, name in enumerate(feature_names):
            hi_fused += optimal_weights[i] * feature_series_dict[name]

        detailed_scores = evaluator.evaluate(hi_fused)

        return optimal_weights, optimal_score, detailed_scores
    else:
        print(f"优化失败: {result.message}")
        return None, None, None


def grid_search_weights(feature_series_dict: Dict[str, np.ndarray],
                       evaluator: HealthIndicatorEvaluator,
                       w_mon: float = 0.5,
                       w_trend: float = 0.4,
                       w_rob: float = 0.1,
                       n_points: int = 20) -> Tuple[np.ndarray, float, Dict]:
    """
    网格搜索最优融合权重（用于验证scipy结果）

    Args:
        feature_series_dict: 原始特征序列字典
        evaluator: 评估器实例
        w_mon: 单调性权重
        w_trend: 趋势性权重
        w_rob: 鲁棒性权重
        n_points: 每个维度的网格点数

    Returns:
        最优权重, 最优综合评分, 融合后HI的详细评估分数
    """
    n_features = len(feature_series_dict)

    if n_features != 3:
        raise ValueError("网格搜索目前只支持3个特征")

    best_score = 0
    best_weights = None

    # 生成网格点
    for w1 in np.linspace(0, 1, n_points):
        for w2 in np.linspace(0, 1-w1, n_points):
            w3 = 1 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue

            w = np.array([w1, w2, w3])
            score = objective_function(w, feature_series_dict, evaluator,
                                      w_mon, w_trend, w_rob)

            if score > best_score:
                best_score = score
                best_weights = w

    # 计算融合后HI的详细评估分数
    hi_fused = np.zeros_like(list(feature_series_dict.values())[0])
    feature_names = list(feature_series_dict.keys())
    for i, name in enumerate(feature_names):
        hi_fused += best_weights[i] * feature_series_dict[name]

    detailed_scores = evaluator.evaluate(hi_fused)

    return best_weights, best_score, detailed_scores


if __name__ == '__main__':
    # 测试代码
    print("测试严格融合权重优化...")
    print("=" * 80)

    from data_loader import XJTUDataLoader
    from feature_extraction import FeatureExtractor
    from config import DATA_PATHS, get_fault_frequencies

    # 加载Bearing1_1数据
    bearing_name = 'Bearing1_1'
    working_condition = 1

    loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])
    data_list, file_names = loader.load_bearing_data_by_file(bearing_name)
    print(f"加载 {bearing_name}: {len(data_list)} 个文件")

    # 提取三个候选特征的时间序列
    extractor = FeatureExtractor(sampling_rate=25600)
    fault_freqs = get_fault_frequencies(working_condition)

    rms_series = []
    envelope_peak_series = []
    bpfo_amplitude_series = []

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

    print(f"提取特征序列长度: {len(rms_series)}")

    # 初始化评估器
    evaluator = HealthIndicatorEvaluator()

    # 方法1: scipy优化
    print("\n" + "=" * 80)
    print("方法1: scipy优化（严格版本）")
    print("-" * 80)
    weights_scipy, score_scipy, scores_scipy = optimize_weights_scipy(
        feature_series_dict, evaluator)

    if weights_scipy is not None:
        print(f"\n最优权重:")
        for name, w in zip(feature_series_dict.keys(), weights_scipy):
            print(f"  {name:<20} {w:.4f} ({w*100:.2f}%)")

        print(f"\n融合后HI的评估分数:")
        print(f"  Monotonicity:  {scores_scipy['monotonicity']:.4f}")
        print(f"  Trendability:  {scores_scipy['trendability']:.4f}")
        print(f"  Robustness:    {scores_scipy['robustness']:.4f}")
        print(f"  综合评分:      {score_scipy:.4f}")

    # 方法2: 网格搜索（验证）
    print("\n" + "=" * 80)
    print("方法2: 网格搜索（验证scipy结果）")
    print("-" * 80)
    weights_grid, score_grid, scores_grid = grid_search_weights(
        feature_series_dict, evaluator, n_points=15)

    print(f"\n最优权重:")
    for name, w in zip(feature_series_dict.keys(), weights_grid):
        print(f"  {name:<20} {w:.4f} ({w*100:.2f}%)")

    print(f"\n融合后HI的评估分数:")
    print(f"  Monotonicity:  {scores_grid['monotonicity']:.4f}")
    print(f"  Trendability:  {scores_grid['trendability']:.4f}")
    print(f"  Robustness:    {scores_grid['robustness']:.4f}")
    print(f"  综合评分:      {score_grid:.4f}")

    # 对比单个特征的评估分数
    print("\n" + "=" * 80)
    print("对比：单个特征的评估分数")
    print("-" * 80)

    for name, series in feature_series_dict.items():
        scores = evaluator.evaluate(series)
        composite = 0.5 * scores['monotonicity'] + 0.4 * scores['trendability'] + 0.1 * scores['robustness']
        print(f"\n{name}:")
        print(f"  Monotonicity:  {scores['monotonicity']:.4f}")
        print(f"  Trendability:  {scores['trendability']:.4f}")
        print(f"  Robustness:    {scores['robustness']:.4f}")
        print(f"  综合评分:      {composite:.4f}")

    print("\n" + "=" * 80)
    print("完成！")

