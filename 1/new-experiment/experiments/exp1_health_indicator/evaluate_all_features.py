"""
计算Bearing1_1的所有候选健康指标的评估分数
用于人工筛选最优指标
"""

import numpy as np
import pandas as pd
from data_loader import XJTUDataLoader
from feature_extraction import FeatureExtractor
from hi_evaluation import HealthIndicatorEvaluator
from config import DATA_PATHS, get_fault_frequencies


def compute_all_features_for_bearing(bearing_name: str, working_condition: int):
    """
    计算指定轴承的所有特征序列

    Args:
        bearing_name: 轴承名称
        working_condition: 工况编号

    Returns:
        特征字典，每个特征是一个时间序列
    """
    # 加载数据
    loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])
    data_list, file_names = loader.load_bearing_data_by_file(bearing_name)
    print(f"加载 {bearing_name}: {len(data_list)} 个文件")

    # 获取故障频率
    fault_freqs = get_fault_frequencies(working_condition)

    # 初始化特征提取器
    extractor = FeatureExtractor(sampling_rate=25600)

    # 存储所有特征的时间序列
    feature_series = {}

    # 对每个文件提取特征
    for i, signal in enumerate(data_list):
        # 提取所有特征
        features = extractor.extract_all_features(signal, fault_freqs, highpass_cutoff=1000)

        # 存储到时间序列
        for feature_name, feature_value in features.items():
            if feature_name not in feature_series:
                feature_series[feature_name] = []
            feature_series[feature_name].append(feature_value)

    # 转换为numpy数组
    for feature_name in feature_series:
        feature_series[feature_name] = np.array(feature_series[feature_name])

    print(f"提取特征数量: {len(feature_series)}")
    return feature_series


def evaluate_all_features(feature_series: dict):
    """
    评估所有特征

    Args:
        feature_series: 特征字典

    Returns:
        评估结果DataFrame
    """
    evaluator = HealthIndicatorEvaluator()

    results = []

    for feature_name, series in feature_series.items():
        # 评估
        scores = evaluator.evaluate(series)

        # 计算综合评分（权重: Mon=0.5, Trend=0.4, Rob=0.1）
        composite_score = (0.5 * scores['monotonicity'] +
                          0.4 * scores['trendability'] +
                          0.1 * scores['robustness'])

        results.append({
            'feature_name': feature_name,
            'monotonicity': scores['monotonicity'],
            'trendability': scores['trendability'],
            'robustness': scores['robustness'],
            'composite_score': composite_score
        })

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 按综合评分排序
    df = df.sort_values('composite_score', ascending=False)

    return df


def print_evaluation_results(df: pd.DataFrame):
    """
    打印评估结果，标注最大值

    Args:
        df: 评估结果DataFrame
    """
    print("\n" + "=" * 100)
    print("Bearing1_1 所有候选健康指标评估结果")
    print("=" * 100)

    # 找到每列的最大值
    max_mon = df['monotonicity'].max()
    max_trend = df['trendability'].max()
    max_rob = df['robustness'].max()
    max_comp = df['composite_score'].max()

    print(f"\n{'特征名称':<25} {'单调性':<12} {'趋势性':<12} {'鲁棒性':<12} {'综合评分':<12}")
    print("-" * 100)

    for idx, row in df.iterrows():
        feature_name = row['feature_name']
        mon = row['monotonicity']
        trend = row['trendability']
        rob = row['robustness']
        comp = row['composite_score']

        # 标注最大值
        mon_str = f"{mon:.4f}" + (" ★" if mon == max_mon else "  ")
        trend_str = f"{trend:.4f}" + (" ★" if trend == max_trend else "  ")
        rob_str = f"{rob:.4f}" + (" ★" if rob == max_rob else "  ")
        comp_str = f"{comp:.4f}" + (" ★" if comp == max_comp else "  ")

        print(f"{feature_name:<25} {mon_str:<12} {trend_str:<12} {rob_str:<12} {comp_str:<12}")

    print("\n" + "=" * 100)
    print("说明: ★ 表示该列的最大值")
    print("=" * 100)


def save_results(df: pd.DataFrame, output_path: str):
    """
    保存结果到CSV

    Args:
        df: 评估结果DataFrame
        output_path: 输出路径
    """
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    # 计算Bearing1_1的所有特征
    bearing_name = 'Bearing1_1'
    working_condition = 1

    print("步骤1: 提取所有特征...")
    feature_series = compute_all_features_for_bearing(bearing_name, working_condition)

    print("\n步骤2: 评估所有特征...")
    df_results = evaluate_all_features(feature_series)

    print("\n步骤3: 显示评估结果...")
    print_evaluation_results(df_results)

    # 保存结果
    output_path = f"{DATA_PATHS['output_root']}/{bearing_name}_feature_evaluation.csv"
    save_results(df_results, output_path)

    print("\n完成！请根据评估结果人工筛选性能好的指标。")
