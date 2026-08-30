"""
实验1最终版：基于筛选特征的健康指标融合
选定特征：rms, envelope_peak, BPFO_amplitude
融合方法：组合权重（70%熵权法 + 30%综合评分）
"""

import numpy as np
import pandas as pd
from data_loader import XJTUDataLoader
from feature_extraction import FeatureExtractor
from hi_evaluation import HealthIndicatorEvaluator
from config import DATA_PATHS, get_fault_frequencies


class SelectedFeatureFusion:
    """基于筛选特征的健康指标融合器"""

    def __init__(self, selected_features: list, working_condition: int):
        """
        初始化

        Args:
            selected_features: 选定的特征名称列表
            working_condition: 工况编号
        """
        self.selected_features = selected_features
        self.working_condition = working_condition
        self.extractor = FeatureExtractor(sampling_rate=25600)
        self.evaluator = HealthIndicatorEvaluator()
        self.fault_freqs = get_fault_frequencies(working_condition)

    def extract_features_for_bearing(self, bearing_name: str):
        """
        提取指定轴承的选定特征序列

        Args:
            bearing_name: 轴承名称

        Returns:
            特征字典，每个特征是一个时间序列
        """
        loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])
        data_list, file_names = loader.load_bearing_data_by_file(bearing_name)

        feature_series = {name: [] for name in self.selected_features}

        for signal in data_list:
            features = self.extractor.extract_all_features(
                signal, self.fault_freqs, highpass_cutoff=1000)

            for feature_name in self.selected_features:
                feature_series[feature_name].append(features[feature_name])

        # 转换为numpy数组
        for feature_name in feature_series:
            feature_series[feature_name] = np.array(feature_series[feature_name])

        return feature_series, file_names

    def compute_fusion_weights(self, feature_series: dict, alpha: float = 0.7):
        """
        计算融合权重（组合权重法）

        Args:
            feature_series: 特征字典
            alpha: 熵权法的权重（默认0.7）

        Returns:
            权重字典
        """
        # 评估每个特征
        evaluation_matrix = []
        composite_scores = []

        for feature_name in self.selected_features:
            scores = self.evaluator.evaluate(feature_series[feature_name])
            evaluation_matrix.append([
                scores['monotonicity'],
                scores['trendability'],
                scores['robustness']
            ])
            # 综合评分（Mon=0.5, Trend=0.4, Rob=0.1）
            comp_score = (0.5 * scores['monotonicity'] +
                         0.4 * scores['trendability'] +
                         0.1 * scores['robustness'])
            composite_scores.append(comp_score)

        evaluation_matrix = np.array(evaluation_matrix)
        composite_scores = np.array(composite_scores)

        # 方法1: 熵权法
        w_entropy = self._compute_entropy_weight(evaluation_matrix)

        # 方法2: 综合评分加权
        w_score = composite_scores / composite_scores.sum()

        # 组合权重
        combined_weights = alpha * w_entropy + (1 - alpha) * w_score
        combined_weights = combined_weights / combined_weights.sum()

        return dict(zip(self.selected_features, combined_weights))

    def _compute_entropy_weight(self, evaluation_matrix: np.ndarray) -> np.ndarray:
        """熵权法计算权重"""
        n_features, n_metrics = evaluation_matrix.shape

        # 归一化
        col_sum = evaluation_matrix.sum(axis=0)
        col_sum[col_sum == 0] = 1.0
        normalized = evaluation_matrix / col_sum

        # 计算信息熵
        k = 1.0 / np.log(n_metrics) if n_metrics > 1 else 1.0
        normalized_safe = np.where(normalized > 0, normalized, 1e-10)
        entropy = -k * np.sum(normalized_safe * np.log(normalized_safe), axis=1)

        # 计算权重
        diversity = 1 - entropy
        weights = diversity / diversity.sum()

        return weights

    def fuse_features(self, feature_series: dict, weights: dict):
        """
        融合特征为复合健康指标

        Args:
            feature_series: 特征字典
            weights: 权重字典

        Returns:
            复合健康指标序列
        """
        # 归一化每个特征到[0, 1]
        normalized_features = {}
        for feature_name, series in feature_series.items():
            normalized = (series - series.min()) / (series.max() - series.min() + 1e-10)
            normalized_features[feature_name] = normalized

        # 加权融合
        composite_hi = np.zeros(len(next(iter(normalized_features.values()))))
        for feature_name, weight in weights.items():
            composite_hi += weight * normalized_features[feature_name]

        return composite_hi

    def process_bearing(self, bearing_name: str):
        """
        处理单个轴承：提取特征 → 计算权重 → 融合

        Args:
            bearing_name: 轴承名称

        Returns:
            结果字典
        """
        print(f"\n处理 {bearing_name}...")

        # 提取特征
        feature_series, file_names = self.extract_features_for_bearing(bearing_name)
        print(f"  提取特征: {len(file_names)}个文件")

        # 计算权重
        weights = self.compute_fusion_weights(feature_series)
        print(f"  融合权重:")
        for name, weight in weights.items():
            print(f"    {name}: {weight:.4f} ({weight*100:.2f}%)")

        # 融合
        composite_hi = self.fuse_features(feature_series, weights)

        # 评估复合HI
        composite_scores = self.evaluator.evaluate(composite_hi)
        composite_score = (0.5 * composite_scores['monotonicity'] +
                          0.4 * composite_scores['trendability'] +
                          0.1 * composite_scores['robustness'])

        print(f"  复合HI评估:")
        print(f"    单调性: {composite_scores['monotonicity']:.4f}")
        print(f"    趋势性: {composite_scores['trendability']:.4f}")
        print(f"    鲁棒性: {composite_scores['robustness']:.4f}")
        print(f"    综合评分: {composite_score:.4f}")

        return {
            'bearing_name': bearing_name,
            'feature_series': feature_series,
            'weights': weights,
            'composite_hi': composite_hi,
            'file_names': file_names,
            'evaluation': composite_scores,
            'composite_score': composite_score
        }


if __name__ == '__main__':
    # 选定的特征
    selected_features = ['rms', 'envelope_peak', 'BPFO_amplitude']

    print("=" * 80)
    print("实验1：基于筛选特征的健康指标融合")
    print("=" * 80)
    print(f"\n选定特征: {', '.join(selected_features)}")
    print("融合方法: 组合权重（70%熵权法 + 30%综合评分）")
    print("评估权重: Mon=0.5, Trend=0.4, Rob=0.1")

    # 测试Bearing1_1
    fusion = SelectedFeatureFusion(selected_features, working_condition=1)
    result = fusion.process_bearing('Bearing1_1')

    # 保存结果
    output_dir = DATA_PATHS['output_root']
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 保存复合HI数据
    np.savez(f"{output_dir}/Bearing1_1_composite_HI_final.npz",
             composite_hi=result['composite_hi'],
             rms=result['feature_series']['rms'],
             envelope_peak=result['feature_series']['envelope_peak'],
             BPFO_amplitude=result['feature_series']['BPFO_amplitude'],
             weights=np.array(list(result['weights'].values())),
             feature_names=selected_features,
             file_names=result['file_names'])

    print(f"\n结果已保存: {output_dir}/Bearing1_1_composite_HI_final.npz")

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
