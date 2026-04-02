import torch
import torch.nn as nn

class PINNLoss(nn.Module):
    def __init__(self, lambda_mono=0.1, lambda_exp=0.05, penalty_ratio=3.0):
        super(PINNLoss, self).__init__()
        self.lambda_mono = lambda_mono
        self.lambda_exp = lambda_exp
        # 高估寿命(危险)的惩罚是低估的 3 倍
        self.penalty_ratio = penalty_ratio 

    def forward(self, y_pred, y_true):
        # 1. 非对称均方误差 (Asymmetric MSE)
        diff = y_pred - y_true
        weight_matrix = torch.where(diff > 0, 
                                    torch.tensor(self.penalty_ratio, device=diff.device), 
                                    torch.tensor(1.0, device=diff.device))
        mse_loss = torch.mean(weight_matrix * (diff ** 2))

        # 2. 单调性物理约束
        diff_pred = y_pred[1:] - y_pred[:-1]
        mono_loss = torch.mean(torch.relu(diff_pred))

        # 3. 指数加速退化约束
        time_steps = torch.arange(y_pred.size(0), dtype=torch.float32, device=y_pred.device)
        normalized_time = time_steps / time_steps.max()
        target_curve = torch.exp(-3.0 * normalized_time)
        exp_loss = torch.mean((y_pred.squeeze() - target_curve)**2)

        # 组合总损失
        total_loss = mse_loss + self.lambda_mono * mono_loss + self.lambda_exp * exp_loss
        
        # 【修复点】：返回 4 个值，满足 trainer.py 的解包和日志打印需求
        return total_loss, mse_loss, mono_loss, exp_loss