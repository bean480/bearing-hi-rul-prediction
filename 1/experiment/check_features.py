import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体，防止图表中的中文显示为方块 (Windows 常用雅黑)
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_features():
    print("正在加载已处理的数据...")
    # 加载训练集张量
    X = np.load("./data/processed/X_train.npy")
    Y = np.load("./data/processed/Y_train.npy")
    
    print(f"成功加载！X shape: {X.shape}, Y shape: {Y.shape}")
    
    # 为了看清单个轴承的完整生命周期，我们取前 2500 个样本 
    # (根据之前 PHM2012 Bearing1_1 的长度，这通常能覆盖第一个轴承的整个周期)
    plot_len = min(2500, len(X))
    
    # 我们的张量形状是 (Batch, Seq_len, Features)
    # 我们提取每个滑动窗口的最后一个时间步（-1）来绘制连续的趋势曲线
    rms_trend = X[:plot_len, -1, 0]           # 索引 0: RMS (均方根，代表整体振动能量)
    kurtosis_trend = X[:plot_len, -1, 1]      # 索引 1: 峰度 (对早期冲击敏感)
    high_freq_trend = X[:plot_len, -1, 6]     # 索引 6: 高频带能量比 (物理频域特征)
    rul_labels = Y[:plot_len]                 # 对应的 RUL 标签
    
    # 创建一个包含 4 个子图的画布
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # 1. 绘制 RUL 标签 (检查分段线性和 FPT 拐点)
    axs[0].plot(rul_labels, color='red', linewidth=2)
    axs[0].set_title("目标标签: RUL (剩余使用寿命) - 检查 FPT 拐点是否合理")
    axs[0].set_ylabel("RUL (归一化)")
    axs[0].grid(True)
    
    # 2. 绘制 RMS 特征
    axs[1].plot(rms_trend, color='blue', alpha=0.8)
    axs[1].set_title("时域特征: RMS (均方根) 归一化后")
    axs[1].set_ylabel("Z-score")
    axs[1].grid(True)
    
    # 3. 绘制 峰度 特征
    axs[2].plot(kurtosis_trend, color='green', alpha=0.8)
    axs[2].set_title("时域特征: Kurtosis (峰度) - 观察是否在早期有毛刺突变")
    axs[2].set_ylabel("Z-score")
    axs[2].grid(True)
    
    # 4. 绘制 频域物理特征
    axs[3].plot(high_freq_trend, color='purple', alpha=0.8)
    axs[3].set_title("频域特征: 高频带能量比 - 观察高频冲击能量的演化")
    axs[3].set_ylabel("Z-score")
    axs[3].set_xlabel("时间步 (Time Steps / Files)")
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_features()