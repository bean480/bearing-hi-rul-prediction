import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.metrics import RULMetrics  # 【新增这一行】

class RULEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        model_path = "./results/checkpoints/best_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到权重文件: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"✅ Loaded weights: {model_path}")

    def predict_with_uncertainty(self, x_test, n_iter=50):
        self.model.train() 
        x_tensor = torch.FloatTensor(x_test).to(self.device)
        
        predictions = []
        print(f"Running {n_iter} MC-Dropout iterations...")
        
        with torch.no_grad():
            for _ in range(n_iter):
                output = self.model(x_tensor)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred

    def visualize_result(self, true_rul, mean_pred, std_pred, save_name="prediction_result.png"):
        save_dir = "./results/figures"
        os.makedirs(save_dir, exist_ok=True)

        # ====== 【核心新增：计算定量指标】 ======
        rmse_val = RULMetrics.calculate_rmse(true_rul, mean_pred)
        score_val = RULMetrics.calculate_score(true_rul, mean_pred)
        
        print("\n" + "="*40)
        print("🏆 定量评估结果 (Quantitative Results)")
        print("="*40)
        print(f"🔹 RMSE (均方根误差): {rmse_val:.4f}")
        print(f"🔹 Score (非对称惩罚分数): {score_val:.2f}")
        print("="*40 + "\n")
        # =======================================

        plt.figure(figsize=(12, 6))
        
        plt.plot(true_rul, label='True RUL', color='black', linestyle='--', linewidth=2)
        plt.plot(mean_pred, label=f'Pred RUL (RMSE: {rmse_val:.4f})', color='blue', linewidth=1.5)
        
        plt.fill_between(
            range(len(mean_pred)),
            mean_pred - 1.96 * std_pred,
            mean_pred + 1.96 * std_pred,
            color='blue', alpha=0.2, label='95% Confidence Interval'
        )
        
        # 把 RMSE 写进标题里，出图更专业！
        plt.title(f"Bearing RUL Prediction | RMSE: {rmse_val:.4f} | Score: {score_val:.2f}", fontsize=14)
        plt.xlabel("Time Steps (Files)", fontsize=12)
        plt.ylabel("Normalized RUL", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        plt.show()