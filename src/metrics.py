import numpy as np

class RULMetrics:
    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """
        计算均方根误差 (RMSE)
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def calculate_score(y_true, y_pred):
        """
        计算经典的非对称惩罚分数 (Asymmetric Scoring Function)
        注意：这里的 a1 和 a2 根据归一化 [0,1] 的尺度进行了微调。
        如果是真实的循环次数/小时数，通常 a1=13, a2=10。
        由于我们的 RUL 范围是 0 到 1，我们使用 a1=0.13, a2=0.10。
        """
        d = y_pred - y_true
        score = 0.0
        
        # 惩罚系数 (归一化尺度)
        a1 = 0.13 # 提前预测的宽容惩罚因子
        a2 = 0.10 # 滞后预测的严厉惩罚因子
        
        for error in d:
            if error < 0:
                # 提前预测 (Early Prediction)
                score += (np.exp(-error / a1) - 1)
            else:
                # 滞后预测 (Late Prediction) - 惩罚极重！
                score += (np.exp(error / a2) - 1)
                
        return score