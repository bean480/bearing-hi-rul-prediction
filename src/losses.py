import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNLoss(nn.Module):
    def __init__(self, lambda_mono=0.1, lambda_exp=0.05):
        super(PINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_mono = lambda_mono  # 单调性约束权重
        self.lambda_exp = lambda_exp    # 指数退化约束权重

    def forward(self, pred, target):
        """
        pred: 模型预测值 (Batch,)
        target: 真实 RUL 标签 (Batch,)
        """
        # 1. 基础任务损失 (MSE)
        loss_mse = self.mse(pred, target)

        # 2. 物理约束一：单调性损失 (Monotonicity Loss)
        # 轴承寿命在物理上不可逆，预测值 y_t 应该小于 y_{t-1}
        # 我们计算 Batch 内相邻样本的差值（假设 Batch 内是按时间排序的切片）
        # 如果 y_t - y_{t-1} > 0，说明寿命增加了，产生惩罚
        diff = pred[1:] - pred[:-1]
        loss_mono = torch.mean(F.relu(diff)) # 只惩罚大于 0 的部分

        # 3. 物理约束二：指数退化一致性 (Exponential Trend Loss)
        # 轴承后期退化通常符合指数规律：y = a * exp(-bt)
        # 我们通过惩罚预测值与对数线性趋势的偏差来逼近它
        # 这里简化处理：要求预测值的二阶导数（加速度）大于0，即曲线是下凸的
        if len(pred) > 2:
            second_diff = pred[2:] - 2*pred[1:-1] + pred[:-2]
            loss_exp = torch.mean(F.relu(-second_diff)) # 惩罚上凸（减速退化）的情况
        else:
            loss_exp = 0.0

        # 总损失函数
        total_loss = loss_mse + self.lambda_mono * loss_mono + self.lambda_exp * loss_exp
        
        return total_loss, loss_mse, loss_mono, loss_exp