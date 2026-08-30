import torch
import torch.nn as nn



class PINNLoss(nn.Module):
    def __init__(self, lambda_mono=0.1, lambda_exp=0.05, penalty_ratio=1.5, health_weight=2.0, lambda_gate=0.01):
        super(PINNLoss, self).__init__()
        self.lambda_mono = lambda_mono
        self.lambda_exp = lambda_exp
        self.penalty_ratio = penalty_ratio
        # 【新增】健康期保真权重，用于对抗全局跷跷板效应
        self.health_weight = health_weight
        # 【新增】门控正则化权重
        self.lambda_gate = lambda_gate

    def forward(self, y_pred, y_true, gates=None):
        diff = y_pred - y_true
        
        # ==========================================
        # 1. 状态感知均方误差 (Condition-Aware MSE)
        # ==========================================
        weights = torch.ones_like(diff)
        
        # 掩码：判断当前处于哪个阶段
        is_healthy = y_true >= 0.99
        is_degrading = y_true < 0.99
        is_overestimating = diff > 0
        
        # 策略 A：如果是健康期，加大拟合权重，强迫锚定 1.0，但无不对称惩罚
        weights[is_healthy] = self.health_weight
        
        # 策略 B：如果是退化期，且高估了寿命（危险），施加 3 倍重罚
        weights[is_degrading & is_overestimating] = self.penalty_ratio
        
        mse_loss = torch.mean(weights * (diff ** 2))

        # ==========================================
        # 2. 状态感知单调性约束 (Condition-Aware Mono)
        # ==========================================
        diff_pred = y_pred[1:] - y_pred[:-1]
        
        # 【核心改进】：只在退化期施加单调性约束！
        # 健康期的自然波动（如 0.98 -> 1.01）不应被惩罚
        degrad_mask = (y_true[:-1] < 0.99).float() 
        
        # 只计算退化期内“反弹（>0）”的惩罚
        mono_loss = torch.mean(torch.relu(diff_pred) * degrad_mask)

        # ==========================================
        # 3. 指数加速退化约束 (保持不变)
        # ==========================================
        time_steps = torch.arange(y_pred.size(0), dtype=torch.float32, device=y_pred.device)
        normalized_time = time_steps / time_steps.max()
        target_curve = torch.exp(-3.0 * normalized_time)
        exp_loss = torch.mean((y_pred.squeeze() - target_curve)**2)

        total_loss = mse_loss + self.lambda_mono * mono_loss + self.lambda_exp * exp_loss

        # ==========================================
        # 4. 门控正则化约束
        # ==========================================
        gate_loss = 0.0
        if gates is not None:
            # gates: {'reset_gate': (batch, seq_len, hidden), 'update_gate': (batch, seq_len, hidden)}
            reset_gate = gates['reset_gate']  # (batch, seq_len, hidden)
            update_gate = gates['update_gate']

            # 计算每个样本的平均门控值
            reset_mean = reset_gate.mean(dim=[1, 2])  # (batch,)
            update_mean = update_gate.mean(dim=[1, 2])

            # 约束1：门控方差不能太小（避免常数）
            reset_var = reset_gate.var(dim=[1, 2]).mean()
            update_var = update_gate.var(dim=[1, 2]).mean()
            variance_penalty = torch.relu(0.01 - reset_var) + torch.relu(0.01 - update_var)

            # 约束2：更新门应该与RUL负相关（故障期更高）
            # 使用Pearson相关系数的负数作为损失
            update_rul_corr = self._pearson_corr(update_mean, y_true)
            correlation_penalty = torch.relu(update_rul_corr + 0.3)  # 期望相关系数 < -0.3

            gate_loss = variance_penalty + correlation_penalty

        total_loss = total_loss + self.lambda_gate * gate_loss
        return total_loss, mse_loss, mono_loss, exp_loss

    def _pearson_corr(self, x, y):
        """计算Pearson相关系数（数值稳定版本）"""
        # 检查输入是否有效
        if x.numel() < 2 or y.numel() < 2:
            return torch.tensor(0.0, device=x.device)

        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean
        y_centered = y - y_mean

        # 计算标准差
        x_std = torch.sqrt((x_centered**2).sum() + 1e-8)
        y_std = torch.sqrt((y_centered**2).sum() + 1e-8)

        # 如果标准差太小（说明数据几乎是常数），返回0
        if x_std < 1e-6 or y_std < 1e-6:
            return torch.tensor(0.0, device=x.device)

        # 计算相关系数
        corr = (x_centered * y_centered).sum() / (x_std * y_std)

        # 裁剪到[-1, 1]范围，防止数值误差
        corr = torch.clamp(corr, -1.0, 1.0)

        return corr