"""
批量评估多个轴承的所有特征
用于对比不同轴承的特征表现
"""

import numpy as np
import pandas as pd
from data_loader import XJTUDataLoader
from feature_extraction import FeatureExtractor
from hi_evaluation import HealthIndicatorEvaluator
from config import DATA_PATHS, get_fault_frequencies


# 轴承配置
BEARING_CONFIG = {
    'Bearing1_1': {'condition': 1},
    'Bearing1_2': {'condition': 1},
    'Bearing1_3': {'condition': 1},
    'Bearing1_4': {'condition': 1},
    'Bearing1_5': {'condition': 1},
    'Bearing2_1': {'condition': 2},
    'Bearing2_2': {'condition': 2},
    'Bearing2_3': {'condition': 2},
}


def compute_all_features_for_bearing(bearing_name: str, working_condition: int):
    """计算指定轴承的所有特征序列"""
    loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])
    data_list, file_names = loader.load_bearing_data_by_file(bearing_name)

    fault_freqs = get_fault_frequencies(working_condition)
    extractor = FeatureExtractor(sampling_rate=25600)

    feature_series = {}

    for signal in data_list:
        features = extractor.extract_all_features(signal, fault_freqs, highpass_cutoff=1000)
        for feature_name, feature_value in features.items():
            if feature_name not in feature_series:
                feature_series[feature_name] = []
            feature_series[feature_name].append(feature_value)

    for feature_name in feature_series:
        feature_series[feature_name] = np.array(feature_series[feature_name])

    return feature_series, len(data_list)


def evaluate_all_features(feature_series: dict):
    """评估所有特征"""
    evaluator = HealthIndicatorEvaluator()
    results = []

    for feature_name, series in feature_series.items():
        scores = evaluator.evaluate(series)
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

    df = pd.DataFrame(results)
    df = df.sort_values('composite_score', ascending=False)
    return df


def batch_evaluate_bearings(bearing_names: list):
    """批量评估多个轴承"""
    all_results = []

    for bearing_name in bearing_names:
        print(f"\n处理 {bearing_name}...")
        config = BEARING_CONFIG[bearing_name]

        try:
            feature_series, num_files = compute_all_features_for_bearing(
                bearing_name, config['condition'])
            print(f"  文件数: {num_files}")

            df_eval = evaluate_all_features(feature_series)
            df_eval['bearing_name'] = bearing_name
            df_eval['num_files'] = num_files

            all_results.append(df_eval)

        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 合并所有结果
    df_all = pd.concat(all_results, ignore_index=True)
    return df_all


def analyze_feature_consistency(df_all: pd.DataFrame):
    """分析特征在不同轴承上的一致性"""
    print("\n" + "=" * 100)
    print("特征在不同轴承上的表现统计")
    print("=" * 100)

    # 按特征分组，计算平均值和标准差
    feature_stats = df_all.groupby('feature_name').agg({
        'monotonicity': ['mean', 'std'],
        'trendability': ['mean', 'std'],
        'robustness': ['mean', 'std'],
        'composite_score': ['mean', 'std']
    }).round(4)

    # 按平均综合评分排序
    feature_stats = feature_stats.sort_values(('composite_score', 'mean'), ascending=False)

    print(f"\n{'特征名称':<25} {'综合评分(均值)':<15} {'综合评分(标准差)':<15} {'单调性(均值)':<15} {'趋势性(均值)':<15} {'鲁棒性(均值)':<15}")
    print("-" * 100)

    for feature_name in feature_stats.index:
        comp_mean = feature_stats.loc[feature_name, ('composite_score', 'mean')]
        comp_std = feature_stats.loc[feature_name, ('composite_score', 'std')]
        mon_mean = feature_stats.loc[feature_name, ('monotonicity', 'mean')]
        trend_mean = feature_stats.loc[feature_name, ('trendability', 'mean')]
        rob_mean = feature_stats.loc[feature_name, ('robustness', 'mean')]

        print(f"{feature_name:<25} {comp_mean:<15.4f} {comp_std:<15.4f} {mon_mean:<15.4f} {trend_mean:<15.4f} {rob_mean:<15.4f}")

    return feature_stats


def print_top_features_per_bearing(df_all: pd.DataFrame, top_n: int = 5):
    """打印每个轴承的Top N特征"""
    print("\n" + "=" * 100)
    print(f"每个轴承的Top {top_n}特征")
    print("=" * 100)

    for bearing_name in df_all['bearing_name'].unique():
        df_bearing = df_all[df_all['bearing_name'] == bearing_name]
        df_top = df_bearing.nlargest(top_n, 'composite_score')

        print(f"\n{bearing_name}:")
        print("-" * 100)
        for idx, row in df_top.iterrows():
            print(f"  {row['feature_name']:<25} 综合评分: {row['composite_score']:.4f}  "
                  f"(Mon={row['monotonicity']:.3f}, Trend={row['trendability']:.3f}, Rob={row['robustness']:.3f})")


if __name__ == '__main__':
    # 选择要评估的轴承（先评估工况1和工况2的部分轴承）
    bearings_to_evaluate = [
        'Bearing1_1',  # 已经评估过
        'Bearing1_2',  # 工况1，另一个轴承
        'Bearing2_1',  # 工况2，寿命最长的轴承
        'Bearing2_2',  # 工况2，另一个轴承
    ]

    print("=" * 100)
    print("批量评估多个轴承的特征")
    print("=" * 100)

    # 批量评估
    df_all = batch_evaluate_bearings(bearings_to_evaluate)

    # 保存详细结果
    output_path = f"{DATA_PATHS['output_root']}/all_bearings_feature_evaluation.csv"
    df_all.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {output_path}")

    # 分析特征一致性
    feature_stats = analyze_feature_consistency(df_all)

    # 保存统计结果
    stats_path = f"{DATA_PATHS['output_root']}/feature_statistics.csv"
    feature_stats.to_csv(stats_path, encoding='utf-8-sig')
    print(f"\n统计结果已保存: {stats_path}")

    # 打印每个轴承的Top 5特征
    print_top_features_per_bearing(df_all, top_n=5)

    print("\n" + "=" * 100)
    print("完成！")
    print("=" * 100)
