"""
实验1 深度HI — 数据加载模块
逐文件独立切分 + 逐轴承Z-score标准化 + 元数据标注
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset

from config import (
    DATA_ROOT, SAMPLING_RATE, POINTS_PER_FILE,
    WINDOW_SIZE, HOP_LENGTH,
    BEARINGS_BY_CONDITION, ALL_BEARINGS,
    get_condition,
)


class BearingDataset:
    """
    单个轴承的数据集。

    包含:
      - windows: [T, WINDOW_SIZE] 滑动窗口
      - time_positions: [T] 归一化时间位置 (0~1)
      - file_boundaries: [T] 文件边界标记
      - bearing_name: str
      - condition: int

    缓存: 首次加载后保存 .npz，再次加载 < 0.1s
    """

    def __init__(self, bearing_name: str, data_root: str = DATA_ROOT,
                 cache_dir: str = None):
        self.bearing_name = bearing_name
        self.condition = get_condition(bearing_name)
        self.bearing_dir = Path(data_root) / bearing_name

        if not self.bearing_dir.exists():
            raise FileNotFoundError(f"轴承目录不存在: {self.bearing_dir}")

        if cache_dir is None:
            cache_dir = Path(data_root) / '.cache'
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / f'{bearing_name}.npz'

        # 尝试读缓存
        if self.cache_file.exists():
            self._load_cache()
        else:
            self._load_and_slice()
            self._save_cache()

        self.T = len(self.windows)

    def _load_cache(self):
        data = np.load(self.cache_file)
        self.windows = data['windows']
        self.time_positions = data['time_positions']
        self.file_boundaries = data['file_boundaries']

    def _save_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_file,
            windows=self.windows,
            time_positions=self.time_positions,
            file_boundaries=self.file_boundaries,
        )

    def _load_and_slice(self):
        """
        逐文件加载 + 逐文件独立切窗 + 记录文件边界。
        结果直接写入 self.windows / self.time_positions / self.file_boundaries。
        """
        csv_files = sorted(
            self.bearing_dir.glob('*.csv'),
            key=lambda f: int(f.stem)
        )
        n_files = len(csv_files)

        # Step 1: 计算全局 μ, σ（在拼接后的全数据上，仅用于标准化）
        all_signals = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, header=0)
            signal = df.iloc[:, 0].values.astype(np.float32)  # 水平振动
            all_signals.append(signal)

        global_signal = np.concatenate(all_signals)
        mu = global_signal.mean()
        sigma = global_signal.std()

        # Step 2: 逐文件独立切窗
        windows_list = []
        time_list = []
        boundary_list = []

        for file_idx, csv_file in enumerate(csv_files):
            signal = all_signals[file_idx]
            signal = (signal - mu) / sigma  # Z-score 标准化

            # 文件内滑动窗口
            n_windows_in_file = (POINTS_PER_FILE - WINDOW_SIZE) // HOP_LENGTH + 1

            for w in range(n_windows_in_file):
                start = w * HOP_LENGTH
                window = signal[start:start + WINDOW_SIZE]
                windows_list.append(window)

                # 时间位置：文件内插值
                file_time_start = file_idx / n_files
                file_time_end = (file_idx + 1) / n_files
                t = file_time_start + (w / n_windows_in_file) * (file_time_end - file_time_start)
                time_list.append(t)

                # 文件边界：该文件最后一个窗口
                boundary_list.append(w == n_windows_in_file - 1)

        self.windows = np.stack(windows_list).astype(np.float32)
        self.time_positions = np.array(time_list, dtype=np.float32)
        self.file_boundaries = np.array(boundary_list, dtype=bool)

    def to_torch(self, device='cpu'):
        """转为PyTorch tensor"""
        return {
            'windows': torch.from_numpy(self.windows).unsqueeze(1).to(device),
            'time_positions': torch.from_numpy(self.time_positions).to(device),
            'file_boundaries': torch.from_numpy(self.file_boundaries).to(device),
        }

    def __repr__(self):
        return f"BearingDataset({self.bearing_name}, cond={self.condition}, T={self.T})"


def load_all_bearings(data_root: str = DATA_ROOT) -> Dict[str, BearingDataset]:
    """加载所有15个轴承的数据"""
    datasets = {}
    for name in ALL_BEARINGS:
        print(f"加载 {name}...", end=' ')
        datasets[name] = BearingDataset(name, data_root)
        print(f"  {datasets[name].T} 个窗口")
    return datasets


def get_leave_one_out_split(condition: int, test_idx: int
                            ) -> Tuple[List[str], str]:
    """
    同工况留一法划分。

    Args:
        condition: 工况编号 1/2/3
        test_idx: 测试轴承在该工况内的索引 0~4

    Returns:
        train_names: 训练轴承名列表 (4个)
        test_name: 测试轴承名 (1个)
    """
    bearings = BEARINGS_BY_CONDITION[condition]
    test_name = bearings[test_idx]
    train_names = [b for b in bearings if b != test_name]
    return train_names, test_name


def get_fixed_ablation_split():
    """
    消融实验固定划分：工况1, Bearing1_1测试, Bearing1_2~1_5训练。
    """
    train_names = ['Bearing1_2', 'Bearing1_3', 'Bearing1_4', 'Bearing1_5']
    test_name = 'Bearing1_1'
    return train_names, test_name


if __name__ == '__main__':
    print("=" * 60)
    print("测试数据加载模块")
    print("=" * 60)

    # 测试单个轴承加载
    ds = BearingDataset('Bearing1_1')
    print(f"\n{ds}")
    print(f"  windows shape:     {ds.windows.shape}")
    print(f"  time_positions:    {ds.time_positions[:10]} ... {ds.time_positions[-5:]}")
    print(f"  file_boundaries:   {ds.file_boundaries.sum()} / {ds.T}")

    # 测试留一法划分
    train, test = get_leave_one_out_split(condition=1, test_idx=0)
    print(f"\n工况1 Fold 0: 训练={train}, 测试={test}")

    # 测试全部加载
    print("\n加载全部轴承...")
    all_ds = load_all_bearings()
    total_windows = sum(ds.T for ds in all_ds.values())
    print(f"\n总计: {total_windows} 个窗口")
