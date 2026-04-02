import torch
import torch.nn as nn

class PINNLoss(nn.Module):
    def __init__(self, lambda_mono=0.1, lambda_exp=0.05, penalty_ratio=1.5, health_weight=5.0):
        super(PINNLoss, self).__init__()
        self.lambda_mono = lambda_mono
        self.lambda_exp = lambda_exp
        self.penalty_ratio = penalty_ratio 
        # 【新增】健康期保真权重，用于对抗全局跷跷板效应
        self.health_weight = health_weight 

    def forward(self, y_pred, y_true):
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
        return total_loss, mse_loss, mono_loss, exp_loss