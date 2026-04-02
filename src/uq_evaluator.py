import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class RULEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        model_path = "./results/checkpoints/best_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到权重文件: {model_path}，请先执行训练。")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"✅ Loaded weights: {model_path}")

    def predict_with_uncertainty(self, x_test, n_iter=50):
        self.model.train() # 激活 Dropout
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
        # --- 核心修复：确保文件夹存在 ---
        save_dir = "./results/figures"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"📁 Created directory: {save_dir}")

        plt.figure(figsize=(12, 6))
        
        # 使用英文标签，避开字体问题
        plt.plot(true_rul, label='True RUL', color='black', linestyle='--', linewidth=2)
        plt.plot(mean_pred, label='Pred RUL (Mean)', color='blue', linewidth=1.5)
        
        plt.fill_between(
            range(len(mean_pred)),
            mean_pred - 1.96 * std_pred,
            mean_pred + 1.96 * std_pred,
            color='blue', alpha=0.2, label='95% Confidence Interval (UQ)'
        )
        
        plt.title("Bearing RUL Prediction with Uncertainty Quantification", fontsize=14)
        plt.xlabel("Time Steps (Files)", fontsize=12)
        plt.ylabel("Normalized RUL", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        print(f"📊 Plot saved at: {save_path}")
        plt.show() # 这行会弹窗显示图片