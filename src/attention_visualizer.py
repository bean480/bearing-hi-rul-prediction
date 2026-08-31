import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AttentionVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_weights = None
        
        # 加载最优权重
        model_path = "./results/checkpoints/best_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # 开启评估模式，关闭 Dropout
        
        # 【核心技巧】：注册 Hook，窃取注意力层的输出
        # 注意：这里的 'attention' 需要根据你 models.py 里的实际层名字来改
        # 如果你的注意力层叫 'cbam' 或 'attn'，请在下面替换
        self._register_hook()

    def _register_hook(self):
        """自动寻找模型中的注意力层并挂上钩子"""
        target_layer = None
        # 遍历模型寻找包含 'attention' 或 'cbam' 的层
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'cbam' in name.lower():
                target_layer = module
                print(f"🔗 成功将 Hook 挂载到注意力层: {name}")
                break
                
        if target_layer is None:
            print("⚠️ 警告: 未自动找到名称包含 'attention' 或 'cbam' 的层。")
            print("请检查 src/models.py 中的网络结构命名。")
            return

        # 定义钩子函数
        def hook_fn(module, input, output):
            # 不同的注意力层输出格式可能不同，这里尝试捕获
            if isinstance(output, tuple):
                self.attention_weights = output[0].detach().cpu().numpy()
            else:
                self.attention_weights = output.detach().cpu().numpy()
                
        target_layer.register_forward_hook(hook_fn)

    def visualize_lifespan_attention(self, x_test):
        """可视化整个轴承生命周期内的注意力变化"""
        if getattr(self, 'attention_weights', False) is False:
             return
             
        print("🔍 正在提取全生命周期特征...")
        x_tensor = torch.FloatTensor(x_test).to(self.device)
        
        with torch.no_grad():
            _ = self.model(x_tensor) # 前向传播一次，触发 Hook 记录权重
            
        weights = self.attention_weights
        
        # 如果是 3D 张量 (Batch, Seq, Features)，我们对其进行降维以方便画图
        if len(weights.shape) == 3:
            # 取特征维度的平均值，观察序列上的注意力变化
            weights_2d = np.mean(weights, axis=-1) 
        elif len(weights.shape) == 4: # CNN 的 spatial attention
            weights_2d = np.mean(weights, axis=(2,3))
        else:
            weights_2d = weights

        # 采样：为了图表美观，我们不要画所有点，均匀抽取 100 个时间步
        total_steps = weights_2d.shape[0]
        sample_indices = np.linspace(0, total_steps - 1, 100, dtype=int)
        sampled_weights = weights_2d[sample_indices]
        
        # === 【关键优化】：按列/按时间步进行 Min-Max 归一化 ===
        # 目的：消除绝对数值过大的影响，只看每个特征在不同时间相对它自己的活跃程度
        # 为了防止除以 0，加一个极小的数 1e-8
        min_vals = np.min(sampled_weights, axis=0, keepdims=True)
        max_vals = np.max(sampled_weights, axis=0, keepdims=True)
        sampled_weights_norm = (sampled_weights - min_vals) / (max_vals - min_vals + 1e-8)
        
        # 将归一化后的数据传给 sns.heatmap
        # 后面的 sns.heatmap 记得用 sampled_weights_norm.T 画图

        # === 开始绘制热力图 ===
        plt.figure(figsize=(14, 6))
        
        # 转置矩阵，使得 X 轴为时间步，Y 轴为注意力维度
        sns.heatmap(sampled_weights_norm.T, cmap='viridis', 
                    cbar_kws={'label': 'Attention Weight'},
                    xticklabels=10, yticklabels=False)
        
        plt.title('Evolution of Attention Weights over Bearing Lifespan', fontsize=16)
        plt.xlabel('Degradation Timeline (0% -> 100% Failure)', fontsize=12)
        plt.ylabel('Attention Units / Features', fontsize=12)
        
        # 添加一条垂直线，模拟退化起点 (FPT) 的位置 (假设在 70% 处)
        plt.axvline(x=70, color='red', linestyle='--', linewidth=2, label='Estimated Failure Point (FPT)')
        plt.legend(loc='upper left')
        
        save_dir = "./results/figures"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "attention_heatmap.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 注意力热力图已保存至: {save_path}")
        plt.show()