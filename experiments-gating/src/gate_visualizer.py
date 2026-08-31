import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class GateVisualizer:
    """
    门控机制可视化工具

    功能：
    1. 门控激活值热力图
    2. 门控演化曲线
    3. 门控-RUL关系分析
    4. 健康期vs故障期门控对比
    """

    def __init__(self, model, device, save_dir='./results/gates'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 设置中文字体（避免中文乱码）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def extract_gates(self, X_data, batch_size=128):
        """
        提取所有样本的门控激活值（批处理加速）

        Args:
            X_data: (N, seq_len, features) numpy array
            batch_size: 批处理大小
        Returns:
            reset_gates: (N, seq_len, hidden_dim)
            update_gates: (N, seq_len, hidden_dim)
        """
        self.model.eval()
        reset_gates_list = []
        update_gates_list = []

        with torch.no_grad():
            # 批处理
            for i in range(0, len(X_data), batch_size):
                batch_end = min(i + batch_size, len(X_data))
                batch_x = torch.FloatTensor(X_data[i:batch_end]).to(self.device)

                _, gates = self.model(batch_x, return_gates=True)

                reset_gates_list.append(gates['reset_gate'].cpu().numpy())
                update_gates_list.append(gates['update_gate'].cpu().numpy())

                # 打印进度
                if (i // batch_size) % 10 == 0:
                    print(f"  处理进度: {batch_end}/{len(X_data)} ({100*batch_end/len(X_data):.1f}%)")

        return np.concatenate(reset_gates_list, axis=0), np.concatenate(update_gates_list, axis=0)

    def plot_gate_heatmap(self, X_data, sample_idx=0, save_name='gate_heatmap.png'):
        """
        绘制单个样本的门控激活值热力图

        Args:
            X_data: (N, seq_len, features)
            sample_idx: 样本索引
            save_name: 保存文件名
        """
        self.model.eval()

        with torch.no_grad():
            x = torch.FloatTensor([X_data[sample_idx]]).to(self.device)
            _, gates = self.model(x, return_gates=True)

            reset_gate = gates['reset_gate'].cpu().numpy()[0]  # (seq_len, hidden_dim)
            update_gate = gates['update_gate'].cpu().numpy()[0]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # 重置门热力图
        im1 = axes[0].imshow(reset_gate.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        axes[0].set_xlabel('Time Step', fontsize=12)
        axes[0].set_ylabel('Hidden Unit', fontsize=12)
        axes[0].set_title('Reset Gate Activation', fontsize=14, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Activation Value', fontsize=10)

        # 更新门热力图
        im2 = axes[1].imshow(update_gate.T, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
        axes[1].set_xlabel('Time Step', fontsize=12)
        axes[1].set_ylabel('Hidden Unit', fontsize=12)
        axes[1].set_title('Update Gate Activation', fontsize=14, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Activation Value', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 热力图已保存: {save_path}")
        plt.close()

    def plot_gate_evolution(self, X_data, Y_data, n_samples=500, save_name='gate_evolution.png'):
        """
        绘制门控值随RUL的演化曲线

        Args:
            X_data: (N, seq_len, features)
            Y_data: (N,) RUL标签
            n_samples: 采样数量
            save_name: 保存文件名
        """
        # 随机采样
        if len(X_data) > n_samples:
            indices = np.random.choice(len(X_data), n_samples, replace=False)
            X_sample = X_data[indices]
            Y_sample = Y_data[indices]
        else:
            X_sample = X_data
            Y_sample = Y_data

        print(f"正在提取 {len(X_sample)} 个样本的门控值...")
        reset_gates, update_gates = self.extract_gates(X_sample)

        # 计算每个样本的平均门控值
        reset_mean = reset_gates.mean(axis=(1, 2))  # (N,)
        update_mean = update_gates.mean(axis=(1, 2))

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # 重置门 vs RUL（反转X轴，从1→0显示）
        axes[0].scatter(Y_sample, reset_mean, alpha=0.5, s=20, c='coral')
        axes[0].set_xlabel('RUL (1=Healthy → 0=Failure)', fontsize=12)
        axes[0].set_ylabel('Average Reset Gate Activation', fontsize=12)
        axes[0].set_title('Reset Gate vs RUL', fontsize=14, fontweight='bold')
        axes[0].invert_xaxis()  # 反转X轴
        axes[0].grid(True, alpha=0.3)

        # 添加趋势线
        z = np.polyfit(Y_sample, reset_mean, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(Y_sample.min(), Y_sample.max(), 100)
        axes[0].plot(x_trend, p(x_trend), 'r--', linewidth=2, label='Trend')
        axes[0].legend()

        # 更新门 vs RUL（反转X轴，从1→0显示）
        axes[1].scatter(Y_sample, update_mean, alpha=0.5, s=20, c='steelblue')
        axes[1].set_xlabel('RUL (1=Healthy → 0=Failure)', fontsize=12)
        axes[1].set_ylabel('Average Update Gate Activation', fontsize=12)
        axes[1].set_title('Update Gate vs RUL', fontsize=14, fontweight='bold')
        axes[1].invert_xaxis()  # 反转X轴
        axes[1].grid(True, alpha=0.3)

        # 添加趋势线
        z = np.polyfit(Y_sample, update_mean, 2)
        p = np.poly1d(z)
        axes[1].plot(x_trend, p(x_trend), 'b--', linewidth=2, label='Trend')
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 演化曲线已保存: {save_path}")
        plt.close()

    def plot_gate_comparison(self, X_data, Y_data, threshold=None, save_name='gate_comparison.png'):
        """
        对比健康期和故障期的门控模式

        Args:
            X_data: (N, seq_len, features)
            Y_data: (N,) RUL标签
            threshold: RUL阈值，None时自动使用中位数
            save_name: 保存文件名
        """
        if threshold is None:
            threshold = np.percentile(Y_data, 50)

        # 分离健康期和故障期样本
        healthy_mask = Y_data > threshold
        fault_mask = Y_data <= threshold

        X_healthy = X_data[healthy_mask]
        X_fault = X_data[fault_mask]

        print(f"使用阈值: {threshold:.3f}")
        print(f"健康期样本: {len(X_healthy)}, 故障期样本: {len(X_fault)}")

        # 提取门控值
        print("提取健康期门控值...")
        reset_healthy, update_healthy = self.extract_gates(X_healthy[:100])  # 限制数量
        print("提取故障期门控值...")
        reset_fault, update_fault = self.extract_gates(X_fault[:100])

        # 计算平均值
        reset_healthy_mean = reset_healthy.mean(axis=0)  # (seq_len, hidden_dim)
        update_healthy_mean = update_healthy.mean(axis=0)
        reset_fault_mean = reset_fault.mean(axis=0)
        update_fault_mean = update_fault.mean(axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 健康期 - 重置门
        im1 = axes[0, 0].imshow(reset_healthy_mean.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Hidden Unit')
        axes[0, 0].set_title('Healthy: Reset Gate', fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])

        # 健康期 - 更新门
        im2 = axes[0, 1].imshow(update_healthy_mean.T, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Hidden Unit')
        axes[0, 1].set_title('Healthy: Update Gate', fontweight='bold')
        plt.colorbar(im2, ax=axes[0, 1])

        # 故障期 - 重置门
        im3 = axes[1, 0].imshow(reset_fault_mean.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Hidden Unit')
        axes[1, 0].set_title('Fault: Reset Gate', fontweight='bold')
        plt.colorbar(im3, ax=axes[1, 0])

        # 故障期 - 更新门
        im4 = axes[1, 1].imshow(update_fault_mean.T, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Hidden Unit')
        axes[1, 1].set_title('Fault: Update Gate', fontweight='bold')
        plt.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 对比图已保存: {save_path}")
        plt.close()

    def plot_gate_statistics(self, X_data, Y_data, save_name='gate_statistics.png'):
        """
        绘制门控值的统计分布

        Args:
            X_data: (N, seq_len, features)
            Y_data: (N,) RUL标签
            save_name: 保存文件名
        """
        print("提取门控统计信息...")

        # 分层采样：确保包含各个RUL阶段
        healthy_idx = np.where(Y_data > 0.8)[0]
        mid_idx = np.where((Y_data >= 0.2) & (Y_data <= 0.8))[0]
        fault_idx = np.where(Y_data < 0.2)[0]

        # 每个阶段采样
        n_per_stage = 67  # 每阶段67个，共约200个
        sampled_idx = []
        if len(healthy_idx) > 0:
            sampled_idx.extend(np.random.choice(healthy_idx, min(n_per_stage, len(healthy_idx)), replace=False))
        if len(mid_idx) > 0:
            sampled_idx.extend(np.random.choice(mid_idx, min(n_per_stage, len(mid_idx)), replace=False))
        if len(fault_idx) > 0:
            sampled_idx.extend(np.random.choice(fault_idx, min(n_per_stage, len(fault_idx)), replace=False))

        sampled_idx = np.array(sampled_idx)
        print(f"分层采样: 健康期{min(n_per_stage, len(healthy_idx))}, 中间期{min(n_per_stage, len(mid_idx))}, 故障期{min(n_per_stage, len(fault_idx))}")

        reset_gates, update_gates = self.extract_gates(X_data[sampled_idx])

        # 计算统计量
        reset_mean = reset_gates.mean(axis=(1, 2))
        update_mean = update_gates.mean(axis=(1, 2))
        reset_std = reset_gates.std(axis=(1, 2))
        update_std = update_gates.std(axis=(1, 2))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 重置门均值分布
        axes[0, 0].hist(reset_mean, bins=30, color='coral', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Mean Activation')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reset Gate: Mean Distribution', fontweight='bold')
        axes[0, 0].axvline(reset_mean.mean(), color='red', linestyle='--',
                          label=f'Overall Mean: {reset_mean.mean():.3f}')
        axes[0, 0].legend()

        # 更新门均值分布
        axes[0, 1].hist(update_mean, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Mean Activation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Update Gate: Mean Distribution', fontweight='bold')
        axes[0, 1].axvline(update_mean.mean(), color='blue', linestyle='--',
                          label=f'Overall Mean: {update_mean.mean():.3f}')
        axes[0, 1].legend()

        # 重置门标准差分布
        axes[1, 0].hist(reset_std, bins=30, color='coral', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Std Deviation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reset Gate: Std Distribution', fontweight='bold')

        # 更新门标准差分布
        axes[1, 1].hist(update_std, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Std Deviation')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Update Gate: Std Distribution', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 统计图已保存: {save_path}")
        plt.close()

    def analyze_gate_patterns(self, X_data, Y_data, n_clusters=3, save_name='gate_patterns.png'):
        """
        门控激活模式聚类分析

        Args:
            X_data: (N, seq_len, features)
            Y_data: (N,) RUL标签
            n_clusters: 聚类数量（默认3：健康/退化/故障）
            save_name: 保存文件名
        """
        from sklearn.cluster import KMeans

        print(f"\n{'='*60}")
        print(f"门控激活模式聚类分析 (K={n_clusters})")
        print(f"{'='*60}")

        # 提取门控值
        reset_gates, update_gates = self.extract_gates(X_data)

        # 将门控值展平为特征向量 (n_samples, seq_len * hidden_dim * 2)
        gate_features = np.concatenate([
            reset_gates.reshape(len(reset_gates), -1),
            update_gates.reshape(len(update_gates), -1)
        ], axis=1)

        print(f"门控特征维度: {gate_features.shape}")

        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(gate_features)

        # 分析每个聚类的RUL分布
        print(f"\n聚类结果分析:")
        cluster_info = []
        for i in range(n_clusters):
            cluster_rul = Y_data[clusters == i]
            info = {
                'cluster': i,
                'count': len(cluster_rul),
                'mean_rul': cluster_rul.mean(),
                'std_rul': cluster_rul.std(),
                'min_rul': cluster_rul.min(),
                'max_rul': cluster_rul.max()
            }
            cluster_info.append(info)
            print(f"  聚类 {i}: 样本数={info['count']:4d}, "
                  f"平均RUL={info['mean_rul']:.3f}±{info['std_rul']:.3f}, "
                  f"范围=[{info['min_rul']:.3f}, {info['max_rul']:.3f}]")

        # 按平均RUL排序，判断聚类是否对应退化阶段
        cluster_info_sorted = sorted(cluster_info, key=lambda x: x['mean_rul'], reverse=True)

        print(f"\n物理意义验证:")
        stage_names = ['健康期', '退化期', '故障期']
        for idx, info in enumerate(cluster_info_sorted[:n_clusters]):
            if idx < len(stage_names):
                print(f"  {stage_names[idx]} ← 聚类{info['cluster']} (平均RUL={info['mean_rul']:.3f})")

        # 可视化：聚类结果 vs RUL
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 聚类散点图 (RUL vs Sample Index)
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # 绿/橙/红
        for i in range(n_clusters):
            mask = clusters == i
            axes[0, 0].scatter(Y_data[mask], np.arange(len(Y_data))[mask],
                             c=colors[i % len(colors)], label=f'Cluster {i}',
                             alpha=0.6, s=20)
        axes[0, 0].set_xlabel('RUL (1=Healthy → 0=Failure)', fontsize=11)
        axes[0, 0].set_ylabel('Sample Index', fontsize=11)
        axes[0, 0].set_title('Gate Pattern Clustering vs RUL', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_xaxis()

        # 子图2: 每个聚类的RUL分布（箱线图）
        rul_by_cluster = [Y_data[clusters == i] for i in range(n_clusters)]
        bp = axes[0, 1].boxplot(rul_by_cluster, labels=[f'C{i}' for i in range(n_clusters)],
                                patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0, 1].set_xlabel('Cluster', fontsize=11)
        axes[0, 1].set_ylabel('RUL', fontsize=11)
        axes[0, 1].set_title('RUL Distribution per Cluster', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 子图3: 每个聚类的门控均值对比
        reset_mean_by_cluster = [reset_gates[clusters == i].mean() for i in range(n_clusters)]
        update_mean_by_cluster = [update_gates[clusters == i].mean() for i in range(n_clusters)]

        x_pos = np.arange(n_clusters)
        width = 0.35
        axes[1, 0].bar(x_pos - width/2, reset_mean_by_cluster, width,
                      label='Reset Gate', color='coral', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, update_mean_by_cluster, width,
                      label='Update Gate', color='steelblue', alpha=0.7)
        axes[1, 0].set_xlabel('Cluster', fontsize=11)
        axes[1, 0].set_ylabel('Mean Gate Activation', fontsize=11)
        axes[1, 0].set_title('Gate Activation by Cluster', fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f'C{i}' for i in range(n_clusters)])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0.35, 0.60])  # 聚焦门控变化范围

        # 子图4: 聚类样本数量
        cluster_counts = [len(Y_data[clusters == i]) for i in range(n_clusters)]
        axes[1, 1].bar(range(n_clusters), cluster_counts, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Cluster', fontsize=11)
        axes[1, 1].set_ylabel('Sample Count', fontsize=11)
        axes[1, 1].set_title('Cluster Size Distribution', fontweight='bold')
        axes[1, 1].set_xticks(range(n_clusters))
        axes[1, 1].set_xticklabels([f'C{i}' for i in range(n_clusters)])
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 聚类分析图已保存: {save_path}")
        plt.close()

        return clusters, cluster_info

    def correlate_gates_with_features(self, X_data, Y_data, feature_names=None):
        """
        分析门控值与原始振动特征的相关性

        Args:
            X_data: (N, seq_len, features)
            Y_data: (N,) RUL标签
            feature_names: 特征名称列表（可选）
        """
        from scipy.stats import spearmanr

        print(f"\n{'='*60}")
        print(f"门控值与振动特征的相关性分析")
        print(f"{'='*60}")

        # 提取门控值
        reset_gates, update_gates = self.extract_gates(X_data)
        reset_mean = reset_gates.mean(axis=(1, 2))  # (n_samples,)
        update_mean = update_gates.mean(axis=(1, 2))

        # 使用最后一个时间步的特征（最新状态）
        latest_features = X_data[:, -1, :]  # (n_samples, n_features)
        n_features = latest_features.shape[1]

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]

        print(f"\n相关性分析结果 (Spearman):")
        print(f"{'特征':<15} {'Reset Gate':<20} {'Update Gate':<20}")
        print(f"{'-'*60}")

        correlations = []
        for feat_idx in range(n_features):
            feat = latest_features[:, feat_idx]

            # 计算Spearman相关系数
            r_reset, p_reset = spearmanr(feat, reset_mean)
            r_update, p_update = spearmanr(feat, update_mean)

            correlations.append({
                'feature': feature_names[feat_idx],
                'reset_r': r_reset,
                'reset_p': p_reset,
                'update_r': r_update,
                'update_p': p_update
            })

            # 标记显著性
            reset_sig = '***' if p_reset < 0.001 else ('**' if p_reset < 0.01 else ('*' if p_reset < 0.05 else ''))
            update_sig = '***' if p_update < 0.001 else ('**' if p_update < 0.01 else ('*' if p_update < 0.05 else ''))

            print(f"{feature_names[feat_idx]:<15} "
                  f"r={r_reset:>6.3f} {reset_sig:<3} "
                  f"r={r_update:>6.3f} {update_sig:<3}")

        print(f"\n注: * p<0.05, ** p<0.01, *** p<0.001")

        return correlations

    def plot_single_bearing_trajectory(self, bearing_data, bearing_rul,
                                      save_name='single_bearing_trajectory.png'):
        """
        绘制单个轴承完整生命周期的门控演化

        Args:
            bearing_data: (seq_len, features) 单个轴承的时序数据
            bearing_rul: (seq_len,) 对应的RUL值
            save_name: 保存文件名
        """
        print(f"\n{'='*60}")
        print(f"单个轴承生命周期轨迹可视化")
        print(f"{'='*60}")

        # 如果输入是2D，需要扩展为3D
        if bearing_data.ndim == 2:
            # 假设这是完整的时序数据，需要构造滑动窗口
            print(f"输入数据形状: {bearing_data.shape}")
            print(f"提示: 需要提供滑动窗口格式的数据 (n_windows, seq_len, features)")
            return

        # 提取门控值
        reset_gates, update_gates = self.extract_gates(bearing_data)
        reset_mean = reset_gates.mean(axis=(1, 2))  # (n_windows,)
        update_mean = update_gates.mean(axis=(1, 2))

        print(f"轴承数据: {len(bearing_data)} 个时间窗口")
        print(f"RUL范围: {bearing_rul.min():.3f} → {bearing_rul.max():.3f}")

        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        time_steps = np.arange(len(bearing_rul))

        # 子图1: RUL曲线
        axes[0].plot(time_steps, bearing_rul, 'k-', linewidth=2, label='RUL')
        axes[0].fill_between(time_steps, bearing_rul, alpha=0.3, color='gray')
        axes[0].set_ylabel('RUL', fontsize=12, fontweight='bold')
        axes[0].set_title('Single Bearing Lifecycle Trajectory', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')
        axes[0].set_ylim([-0.05, 1.05])

        # 子图2: 重置门演化
        axes[1].plot(time_steps, reset_mean, 'coral', linewidth=2, label='Reset Gate')
        axes[1].fill_between(time_steps, reset_mean, alpha=0.3, color='coral')
        axes[1].set_ylabel('Reset Gate', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')
        axes[1].set_ylim([0.35, 0.60])  # 聚焦变化范围

        # 子图3: 更新门演化
        axes[2].plot(time_steps, update_mean, 'steelblue', linewidth=2, label='Update Gate')
        axes[2].fill_between(time_steps, update_mean, alpha=0.3, color='steelblue')
        axes[2].set_ylabel('Update Gate', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time Window Index', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')
        axes[2].set_ylim([0.48, 0.56])  # 聚焦变化范围

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 单轴承轨迹图已保存: {save_path}")
        plt.close()
