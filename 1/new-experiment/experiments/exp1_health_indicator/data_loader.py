"""
实验1：健康指标构建 - 数据加载模块
负责加载XJTU-SY数据集的振动信号
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


class XJTUDataLoader:
    """XJTU-SY数据集加载器"""

    def __init__(self, data_root: str):
        """
        初始化数据加载器

        Args:
            data_root: XJTU-SY数据集根目录
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_root}")

    def load_bearing_data(self,
                         bearing_name: str,
                         channel: str = 'horizontal') -> Tuple[np.ndarray, List[str]]:
        """
        加载单个轴承的全寿命数据

        Args:
            bearing_name: 轴承名称，如 'Bearing1_1'
            channel: 振动通道，'horizontal' 或 'vertical'

        Returns:
            data: 振动信号数组，shape = (n_samples,)
            file_list: CSV文件名列表（按时间顺序）
        """
        bearing_dir = self.data_root / bearing_name

        if not bearing_dir.exists():
            raise FileNotFoundError(f"轴承目录不存在: {bearing_dir}")

        # 获取所有CSV文件并按数字排序
        csv_files = sorted(bearing_dir.glob('*.csv'),
                          key=lambda x: int(x.stem))

        if len(csv_files) == 0:
            raise FileNotFoundError(f"未找到CSV文件: {bearing_dir}")

        print(f"加载轴承 {bearing_name}，共 {len(csv_files)} 个文件...")

        # 确定列索引（第0列=水平，第1列=垂直）
        col_idx = 0 if channel == 'horizontal' else 1

        # 逐个加载CSV文件
        data_segments = []
        file_names = []

        for csv_file in csv_files:
            try:
                # 读取CSV（有表头，跳过第一行）
                df = pd.read_csv(csv_file, header=0)

                # 提取指定通道的数据
                segment = df.iloc[:, col_idx].values
                data_segments.append(segment)
                file_names.append(csv_file.name)

            except Exception as e:
                print(f"警告: 读取文件失败 {csv_file.name}: {e}")
                continue

        # 拼接所有数据段
        full_data = np.concatenate(data_segments)

        print(f"加载完成: {len(file_names)} 个文件, 总采样点数: {len(full_data)}")

        return full_data, file_names

    def load_bearing_data_windowed(self,
                                   bearing_name: str,
                                   channel: str = 'horizontal',
                                   window_size: int = 1024,
                                   hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据并分割为滑动窗口

        Args:
            bearing_name: 轴承名称
            channel: 振动通道
            window_size: 窗口大小
            hop_length: 跳跃长度

        Returns:
            windows: 窗口数据，shape = (n_windows, window_size)
            time_indices: 每个窗口的中心时间索引
        """
        # 加载完整数据
        full_data, _ = self.load_bearing_data(bearing_name, channel)

        # 滑动窗口分割
        n_windows = (len(full_data) - window_size) // hop_length + 1
        windows = np.zeros((n_windows, window_size))
        time_indices = np.zeros(n_windows, dtype=int)

        for i in range(n_windows):
            start_idx = i * hop_length
            end_idx = start_idx + window_size
            windows[i] = full_data[start_idx:end_idx]
            time_indices[i] = start_idx + window_size // 2  # 窗口中心

        print(f"窗口分割完成: {n_windows} 个窗口")

        return windows, time_indices

    def load_bearing_data_by_file(self,
                                  bearing_name: str,
                                  channel: str = 'horizontal') -> Tuple[List[np.ndarray], List[str]]:
        """
        按文件加载轴承数据（不拼接）

        Args:
            bearing_name: 轴承名称
            channel: 振动通道

        Returns:
            data_list: 每个文件的数据列表
            file_names: 文件名列表
        """
        bearing_dir = self.data_root / bearing_name

        if not bearing_dir.exists():
            raise FileNotFoundError(f"轴承目录不存在: {bearing_dir}")

        # 获取所有CSV文件并按数字排序
        csv_files = sorted(bearing_dir.glob('*.csv'),
                          key=lambda x: int(x.stem))

        if len(csv_files) == 0:
            raise FileNotFoundError(f"未找到CSV文件: {bearing_dir}")

        print(f"加载轴承 {bearing_name}，共 {len(csv_files)} 个文件...")

        # 确定列索引
        col_idx = 0 if channel == 'horizontal' else 1

        # 逐个加载CSV文件（不拼接）
        data_list = []
        file_names = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, header=0)
                segment = df.iloc[:, col_idx].values
                data_list.append(segment)
                file_names.append(csv_file.name)
            except Exception as e:
                print(f"警告: 读取文件失败 {csv_file.name}: {e}")
                continue

        print(f"加载完成: {len(file_names)} 个文件")

        return data_list, file_names

    @staticmethod
    def get_available_bearings(data_root: str) -> List[str]:
        """
        获取数据集中所有可用的轴承名称

        Args:
            data_root: 数据集根目录

        Returns:
            轴承名称列表
        """
        data_path = Path(data_root)
        bearings = [d.name for d in data_path.iterdir()
                   if d.is_dir() and d.name.startswith('Bearing')]
        return sorted(bearings)


if __name__ == '__main__':
    # 测试数据加载
    from config import DATA_PATHS, FEATURE_CONFIG

    print("=" * 60)
    print("测试XJTU-SY数据加载器")
    print("=" * 60)

    # 初始化加载器
    loader = XJTUDataLoader(DATA_PATHS['xjtu_root'])

    # 列出所有可用轴承
    bearings = loader.get_available_bearings(DATA_PATHS['xjtu_root'])
    print(f"\n可用轴承: {bearings}")

    # 测试加载第一个轴承
    if len(bearings) > 0:
        test_bearing = bearings[0]
        print(f"\n测试加载: {test_bearing}")

        try:
            # 加载完整数据
            data, files = loader.load_bearing_data(test_bearing, channel='horizontal')
            print(f"\n数据统计:")
            print(f"  均值: {np.mean(data):.6f}")
            print(f"  标准差: {np.std(data):.6f}")
            print(f"  最大值: {np.max(data):.6f}")
            print(f"  最小值: {np.min(data):.6f}")

            # 测试窗口分割
            print(f"\n测试窗口分割...")
            windows, time_idx = loader.load_bearing_data_windowed(
                test_bearing,
                window_size=FEATURE_CONFIG['window_size'],
                hop_length=FEATURE_CONFIG['hop_length']
            )
            print(f"  窗口数组形状: {windows.shape}")
            print(f"  时间索引形状: {time_idx.shape}")

        except FileNotFoundError as e:
            print(f"\n错误: {e}")
            print("请确保数据已放入正确的目录")
    else:
        print("\n未找到任何轴承数据，请检查数据路径")
