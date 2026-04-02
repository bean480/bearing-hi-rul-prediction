import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class RULTrainer:
    def __init__(self, model, criterion, config):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 1. 优化器配置 (AdamW 是工业界处理时序问题的首选)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['train']['lr'], 
            weight_decay=config['train']['weight_decay']
        )
        
        # 2. 学习率调度器 (余弦退火，防止模型在后期震荡)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['train']['epochs']
        )
        
        self.best_loss = float('inf')
        self.save_path = os.path.join(config['train']['save_dir'], 'best_model.pth')
        os.makedirs(config['train']['save_dir'], exist_ok=True)

    def train(self, train_loader, val_loader):
        epochs = self.config['train']['epochs']
        print(f"开始训练，使用设备: {self.device}")
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            mse_list, mono_list, exp_list = [], [], []
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                
                # 计算复合损失
                total_loss, l_mse, l_mono, l_exp = self.criterion(output, batch_y)
                
                total_loss.backward()
                self.optimizer.step()
                
                train_losses.append(total_loss.item())
                mse_list.append(l_mse.item())
                mono_list.append(l_mono.item() if isinstance(l_mono, torch.Tensor) else l_mono)
                exp_list.append(l_exp.item() if isinstance(l_exp, torch.Tensor) else l_exp)
            
            self.scheduler.step()
            
            # 验证环节
            val_loss = self.validate(val_loader)
            
            # 打印当前 Epoch 的状态
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"| Loss: {np.mean(train_losses):.4f} "
                  f"(MSE:{np.mean(mse_list):.4f}, Mono:{np.mean(mono_list):.4f}) "
                  f"| Val Loss: {val_loss:.4f}")
            
            # 保存最优模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"⭐ 已保存最优模型至 {self.save_path}")

    def validate(self, val_loader):
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                # 验证时主要看回归误差 (MSE)
                loss_mse = torch.nn.functional.mse_loss(output, batch_y)
                val_losses.append(loss_mse.item())
        return np.mean(val_losses)