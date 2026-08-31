"""
健康指标评估模块
实现三个评估指标：Monotonicity（单调性）、Trendability（趋势性）、Robustness（鲁棒性）
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict


class HealthIndicatorEvaluator:
    """健康指标评估器"""

    @staticmethod
    def compute_monotonicity(hi: np.ndarray) -> float:
        """
        计算单调性（Monotonicity）
        衡量健康指标是否单调递增

        Args:
            hi: 健康指标序列

        Returns:
            单调性分数 [0, 1]，越接近1表示单调性越好
        """
        n = len(hi)
        if n < 2:
            return 0.0

        # 计算相邻点的差分
        diff = np.diff(hi)

        # 统计递增的点数（diff > 0）
        num_increasing = np.sum(diff > 0)

        # 单调性 = 递增点数 / 总点数
        monotonicity = num_increasing / (n - 1)

        return monotonicity

    @staticmethod
    def compute_trendability(hi: np.ndarray) -> float:
        """
        计算趋势性（Trendability）
        衡量健康指标与理想单调曲线的相关性

        Args:
            hi: 健康指标序列

        Returns:
            趋势性分数 [0, 1]，越接近1表示趋势性越好
        """
        n = len(hi)
        if n < 2:
            return 0.0

        # 构建理想单调递增序列（线性）
        ideal_trend = np.linspace(0, 1, n)

        # 归一化健康指标到[0, 1]
        hi_norm = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)

        # 计算Pearson相关系数
        try:
            corr, _ = pearsonr(hi_norm, ideal_trend)
            # 相关系数范围[-1, 1]，转换到[0, 1]
            trendability = (corr + 1) / 2
        except:
            trendability = 0.0

        return trendability

    @staticmethod
    def compute_robustness(hi: np.ndarray) -> float:
        """
        计算鲁棒性（Robustness）
        衡量健康指标的稳定性（抗噪声能力）

        Args:
            hi: 健康指标序列

        Returns:
            鲁棒性分数 [0, 1]，越接近1表示鲁棒性越好
        """
        n = len(hi)
        if n < 3:
            return 0.0

        # 归一化健康指标到[0, 1]
        hi_norm = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)

        # 计算一阶差分（变化率）
        diff = np.abs(np.diff(hi_norm))

        # 计算变异系数（标准差/均值）
        # 变异系数越小，说明变化越平稳，鲁棒性越好
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        if mean_diff < 1e-10:
            return 1.0

        cv = std_diff / mean_diff  # 变异系数

        # 将变异系数转换为鲁棒性分数[0, 1]
        # 使用指数衰减函数：robustness = exp(-cv)
        robustness = np.exp(-cv)

        return robustness

    def evaluate(self, hi: np.ndarray) -> Dict[str, float]:
        """
        综合评估健康指标

        Args:
            hi: 健康指标序列

        Returns:
            包含三个评估指标的字典
        """
        monotonicity = self.compute_monotonicity(hi)
        trendability = self.compute_trendability(hi)
        robustness = self.compute_robustness(hi)

        return {
            'monotonicity': monotonicity,
            'trendability': trendability,
            'robustness': robustness
        }


if __name__ == '__main__':
    # 测试代码
    print("测试健康指标评估模块...")
    print("=" * 50)

    evaluator = HealthIndicatorEvaluator()

    # 测试1：理想单调递增序列
    print("\n测试1：理想单调递增序列")
    hi_ideal = np.linspace(0, 1, 100)
    scores = evaluator.evaluate(hi_ideal)
    print(f"  Monotonicity: {scores['monotonicity']:.4f}")
    print(f"  Trendability: {scores['trendability']:.4f}")
    print(f"  Robustness:   {scores['robustness']:.4f}")

    # 测试2：带噪声的递增序列
    print("\n测试2：带噪声的递增序列")
    hi_noisy = np.linspace(0, 1, 100) + np.random.normal(0, 0.05, 100)
    scores = evaluator.evaluate(hi_noisy)
    print(f"  Monotonicity: {scores['monotonicity']:.4f}")
    print(f"  Trendability: {scores['trendability']:.4f}")
    print(f"  Robustness:   {scores['robustness']:.4f}")

    # 测试3：实际数据（从hi_computation导入）
    print("\n测试3：实际Bearing1_1数据")
    from data_loader import XJTUDataLoader
    from hi_computation import HealthIndicatorComputer
    from config import DATA_PATHS

    loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])
    data_list, _ = loader.load_bearing_data_by_file('Bearing1_1')

    hi_computer = HealthIndicatorComputer(working_condition=1)
    hi_dict = hi_computer.compute_hi_for_bearing(data_list, fault_type='BPFO')

    print("\n  HI1 (RMS):")
    scores_hi1 = evaluator.evaluate(hi_dict['hi1'])
    print(f"    Monotonicity: {scores_hi1['monotonicity']:.4f}")
    print(f"    Trendability: {scores_hi1['trendability']:.4f}")
    print(f"    Robustness:   {scores_hi1['robustness']:.4f}")

    print("\n  HI2 (包络谱BPFO):")
    scores_hi2 = evaluator.evaluate(hi_dict['hi2'])
    print(f"    Monotonicity: {scores_hi2['monotonicity']:.4f}")
    print(f"    Trendability: {scores_hi2['trendability']:.4f}")
    print(f"    Robustness:   {scores_hi2['robustness']:.4f}")

    print("\n" + "=" * 50)
    print("测试完成！")
