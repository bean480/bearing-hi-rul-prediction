"""
健康指标融合模块
使用熵权法（Entropy Weight Method）进行多指标融合
"""

import numpy as np
from typing import Dict, List
from hi_evaluation import HealthIndicatorEvaluator


class EntropyWeightFusion:
    """熵权法融合器"""

    def __init__(self):
        self.evaluator = HealthIndicatorEvaluator()

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        归一化矩阵（按列归一化）

        Args:
            matrix: 输入矩阵，shape=(n_indicators, n_metrics)

        Returns:
            归一化后的矩阵
        """
        # 避免除零
        col_sum = matrix.sum(axis=0)
        col_sum[col_sum == 0] = 1.0

        normalized = matrix / col_sum
        return normalized

    @staticmethod
    def _compute_entropy(matrix: np.ndarray) -> np.ndarray:
        """
        计算信息熵

        Args:
            matrix: 归一化后的矩阵，shape=(n_indicators, n_metrics)

        Returns:
            每个指标的信息熵，shape=(n_indicators,)
        """
        n_indicators, n_metrics = matrix.shape

        # 计算信息熵
        # H_i = -k * Σ(p_ij * ln(p_ij))
        # 其中 k = 1 / ln(n_metrics)

        k = 1.0 / np.log(n_metrics) if n_metrics > 1 else 1.0

        # 避免log(0)
        matrix_safe = np.where(matrix > 0, matrix, 1e-10)

        # 计算每行的熵
        entropy = -k * np.sum(matrix_safe * np.log(matrix_safe), axis=1)

        return entropy

    def compute_weights(self, hi_list: List[np.ndarray]) -> np.ndarray:
        """
        使用熵权法计算权重

        Args:
            hi_list: 健康指标列表，每个元素是一个健康指标序列

        Returns:
            权重数组，shape=(n_indicators,)，和为1
        """
        n_indicators = len(hi_list)

        # Step 1: 评估每个健康指标
        scores_matrix = []
        for hi in hi_list:
            scores = self.evaluator.evaluate(hi)
            scores_matrix.append([
                scores['monotonicity'],
                scores['trendability'],
                scores['robustness']
            ])

        scores_matrix = np.array(scores_matrix)  # shape=(n_indicators, 3)

        # Step 2: 归一化评估矩阵
        normalized_matrix = self._normalize_matrix(scores_matrix)

        # Step 3: 计算信息熵
        entropy = self._compute_entropy(normalized_matrix)

        # Step 4: 计算权重
        # w_i = (1 - H_i) / Σ(1 - H_i)
        diversity = 1 - entropy  # 信息效用值
        weights = diversity / diversity.sum()

        return weights

    def fuse(self, hi_list: List[np.ndarray], weights: np.ndarray = None) -> np.ndarray:
        """
        融合多个健康指标

        Args:
            hi_list: 健康指标列表
            weights: 权重数组（可选），如果为None则自动计算

        Returns:
            融合后的复合健康指标
        """
        if weights is None:
            weights = self.compute_weights(hi_list)

        # 归一化每个健康指标到[0, 1]
        hi_normalized = []
        for hi in hi_list:
            hi_norm = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
            hi_normalized.append(hi_norm)

        hi_normalized = np.array(hi_normalized)  # shape=(n_indicators, n_samples)

        # 加权融合
        hi_composite = np.dot(weights, hi_normalized)

        return hi_composite

    def fuse_with_details(self, hi_dict: Dict[str, np.ndarray]) -> Dict:
        """
        融合健康指标并返回详细信息

        Args:
            hi_dict: 健康指标字典，如 {'hi1': array, 'hi2': array}

        Returns:
            包含融合结果和详细信息的字典
        """
        hi_names = list(hi_dict.keys())
        hi_list = [hi_dict[name] for name in hi_names]

        # 评估每个健康指标
        scores_dict = {}
        for name, hi in zip(hi_names, hi_list):
            scores_dict[name] = self.evaluator.evaluate(hi)

        # 计算权重
        weights = self.compute_weights(hi_list)

        # 融合
        hi_composite = self.fuse(hi_list, weights)

        # 评估复合健康指标
        composite_scores = self.evaluator.evaluate(hi_composite)

        return {
            'hi_composite': hi_composite,
            'weights': dict(zip(hi_names, weights)),
            'scores': scores_dict,
            'composite_scores': composite_scores
        }


if __name__ == '__main__':
    # 测试代码
    print("测试熵权法融合模块...")
    print("=" * 50)

    from data_loader import XJTUDataLoader
    from hi_computation import HealthIndicatorComputer
    from config import DATA_PATHS

    # 加载数据
    loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])
    data_list, _ = loader.load_bearing_data_by_file('Bearing1_1')

    # 计算健康指标
    hi_computer = HealthIndicatorComputer(working_condition=1)
    hi_dict = hi_computer.compute_hi_for_bearing(data_list, fault_type='BPFO')

    # 融合
    fusion = EntropyWeightFusion()
    result = fusion.fuse_with_details(hi_dict)

    print("\n评估分数:")
    print("-" * 50)
    for name, scores in result['scores'].items():
        print(f"\n{name}:")
        print(f"  Monotonicity: {scores['monotonicity']:.4f}")
        print(f"  Trendability: {scores['trendability']:.4f}")
        print(f"  Robustness:   {scores['robustness']:.4f}")

    print("\n" + "-" * 50)
    print("\n熵权法计算的权重:")
    for name, weight in result['weights'].items():
        print(f"  {name}: {weight:.4f}")

    print("\n" + "-" * 50)
    print("\n复合健康指标评估:")
    print(f"  Monotonicity: {result['composite_scores']['monotonicity']:.4f}")
    print(f"  Trendability: {result['composite_scores']['trendability']:.4f}")
    print(f"  Robustness:   {result['composite_scores']['robustness']:.4f}")

    print("\n" + "=" * 50)
    print("测试完成！")
