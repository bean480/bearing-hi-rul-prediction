import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.fft import fft

class PHM2012DataBuilder:
    def __init__(self, config):
        self.train_dirs = config['data']['train_dirs']
        self.test_dirs = config['data']['test_dirs']
        self.processed_dir = config['data']['processed_dir']
        self.fs = config['data']['sampling_rate']
        self.window_size = config['data']['window_size']
        self.healthy_samples = config['data']['healthy_samples']
        
        os.makedirs(self.processed_dir, exist_ok=True)
        # 用于保存训练集的统计量，防止数据泄露
        self.train_mean = None
        self.train_std = None

    def extract_features(self, file_path):
        """提取单个文件的特征 (时域 + 频域能量比)"""
        try:
            df = pd.read_csv(file_path, header=None)
            signal = df.iloc[:, 4].values  # 第5列为水平加速度
        except Exception:
            # 兼容有些解压后是 .txt 格式或者分隔符为空格的情况
            try:
                df = pd.read_csv(file_path, header=None, sep=r'\s+')
                signal = df.iloc[:, 4].values
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return None

        # 时域特征
        rms = np.sqrt(np.mean(signal**2))
        kurt = kurtosis(signal)
        p2p = np.max(signal) - np.min(signal)
        var = np.var(signal)

        # 频域特征 (简单三分频带能量比)
        N = len(signal)
        yf = np.abs(fft(signal))[:N//2]
        total_energy = np.sum(yf) + 1e-8
        band1 = np.sum(yf[:N//6]) / total_energy
        band2 = np.sum(yf[N//6:N//3]) / total_energy
        band3 = np.sum(yf[N//3:]) / total_energy

        return [rms, kurt, p2p, var, band1, band2, band3]

    def process_single_bearing(self, dir_path):
        """处理单个轴承文件夹，返回其原始特征矩阵"""
        print(f"正在读取轴承数据: {dir_path} ...")
        # 兼容 .csv 和 .txt 后缀
        files = sorted(glob.glob(os.path.join(dir_path, "*.*")))
        # 过滤出符合 acc_ 命名规则的文件
        files = [f for f in files if "acc_" in os.path.basename(f)]
        
        if not files:
            raise ValueError(f"在 {dir_path} 中未找到振动数据文件！")
            
        feature_list = []
        for idx, f in enumerate(files):
            feats = self.extract_features(f)
            if feats is not None:
                feature_list.append(feats)
                
        return np.array(feature_list)

    def find_fpt_3sigma(self, rms_array):
        """基于 3-sigma 原则寻找退化起点 FPT"""
        healthy_rms = rms_array[:self.healthy_samples]
        mu = np.mean(healthy_rms)
        sigma = np.std(healthy_rms)
        threshold = mu + 3 * sigma
        
        for i in range(self.healthy_samples, len(rms_array) - 3):
            if (rms_array[i] > threshold and 
                rms_array[i+1] > threshold and 
                rms_array[i+2] > threshold):
                return i
        return int(len(rms_array) * 0.7)
    
    def find_fpt_macro_amplitude(self, rms_array):
        """
        基于宏观退化幅值的 FPT 检测 (工业界实战/顶刊常用方法)
        解决早期微小波动导致的标签误判问题。
        """
        import numpy as np
        
        # 1. 确定健康期基准
        healthy_mean = np.mean(rms_array[:self.healthy_samples])
        
        # 2. 提取寿命末期的最大故障幅值 (取最后 50 个点的均值，防止单一离群点干扰)
        failure_amplitude = np.mean(rms_array[-50:])
        
        # 3. 设定宏观退化阈值：比如，当信号增长达到最大故障跨度的 5% 时
        # 这个 0.05 (5%) 就是过滤早期潜伏期噪声的黄金参数
        threshold = healthy_mean + 0.05 * (failure_amplitude - healthy_mean)
        
        # 4. 从后向前找，或者加入持续性判断，找到真实物理拐点
        sustain_steps = 15 # 必须连续 15 步超过宏观阈值
        consecutive_count = 0
        
        for i in range(self.healthy_samples, len(rms_array)):
            if rms_array[i] > threshold:
                consecutive_count += 1
                if consecutive_count >= sustain_steps:
                    return i - sustain_steps + 1
            else:
                consecutive_count = 0
                
        return int(len(rms_array) * 0.7)
        
        for i in range(self.healthy_samples, len(smoothed_rms)):
            if smoothed_rms[i] > threshold:
                consecutive_count += 1
                if consecutive_count >= sustain_steps:
                    # 确认为真实退化，返回最初越界的那个时间点
                    return i - sustain_steps + 1 
            else:
                consecutive_count = 0 # 若信号回落到健康区间，则重新计数
                
        # 兜底返回值
        return int(len(rms_array) * 0.7)

    def generate_labels(self, total_length, fpt_idx):
        """生成分段线性 RUL 标签 (归一化到 [0, 1])"""
        labels = np.zeros(total_length)
        max_rul = total_length - fpt_idx
        labels[:fpt_idx] = max_rul
        labels[fpt_idx:] = np.arange(max_rul, 0, -1)
        return labels / max_rul

    def create_sliding_window(self, features, labels):
        """构建时序滑窗 (Batch, Seq_len, Features)"""
        X, Y = [], []
        T = len(features)
        for i in range(T - self.window_size):
            X.append(features[i : i + self.window_size, :])
            Y.append(labels[i + self.window_size]) 
        return np.array(X), np.array(Y)

    def build_dataset(self, dirs, is_train=True):
        """核心处理管道"""
        all_X, all_Y = [], []
        raw_features_dict = {}

        # 1. 提取所有指定轴承的原始特征
        for d in dirs:
            raw_features_dict[d] = self.process_single_bearing(d)

        # 2. 如果是训练集，计算全局均值和标准差
        if is_train:
            all_train_features = np.vstack(list(raw_features_dict.values()))
            self.train_mean = np.mean(all_train_features, axis=0)
            self.train_std = np.std(all_train_features, axis=0) + 1e-8

        # 3. 归一化、找 FPT、打标签、切窗
        for d, features in raw_features_dict.items():
            # 使用训练集的分布进行 Z-score 归一化
            norm_features = (features - self.train_mean) / self.train_std

            # --- 【V1.2 新增】: EMA 指数移动平均平滑，滤除高频物理噪声 ---
            df_norm = pd.DataFrame(norm_features)
            # span=10 表示利用过去 10 个时间步的均值进行平滑，值越大越平滑
            norm_features = df_norm.ewm(span=10, adjust=False).mean().values
            
            # 使用第一列 (RMS) 寻找 FPT
            # 更换为宏观退化幅值方法，避免早期噪声导致提前判定
            fpt_idx = self.find_fpt_macro_amplitude(norm_features[:, 0])
            print(f"[{os.path.basename(d)}] FPT 宏观幅值检测于时间步: {fpt_idx}")
            
            labels = self.generate_labels(len(norm_features), fpt_idx)
            X, Y = self.create_sliding_window(norm_features, labels)
            all_X.append(X)
            all_Y.append(Y)

        # 将多个轴承的张量在 Batch 维度拼接
        final_X = np.concatenate(all_X, axis=0)
        final_Y = np.concatenate(all_Y, axis=0)
        return final_X, final_Y

    def process(self):
        print("=== 开始处理训练集 ===")
        X_train, Y_train = self.build_dataset(self.train_dirs, is_train=True)
        
        print("\n=== 开始处理测试集 ===")
        X_test, Y_test = self.build_dataset(self.test_dirs, is_train=False)
        
        # 保存为 Numpy 数组
        np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.processed_dir, "Y_train.npy"), Y_train)
        np.save(os.path.join(self.processed_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.processed_dir, "Y_test.npy"), Y_test)
        
        print("\n✅ 所有数据处理完成！")
        print(f"训练集张量形态: X {X_train.shape}, Y {Y_train.shape}")
        print(f"测试集张量形态: X {X_test.shape}, Y {Y_test.shape}")