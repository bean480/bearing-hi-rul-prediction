import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class RULTrainer:
    def __init__(self, model, criterion, config, use_multi_gpu=True):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.use_multi_gpu = use_multi_gpu

        # 多卡训练配置
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.model.to(self.device)

            if use_multi_gpu and torch.cuda.device_count() > 1:
                print(f"🚀 检测到 {torch.cuda.device_count()} 张GPU，启用DataParallel多卡训练")
                self.model = torch.nn.DataParallel(self.model)
            else:
                print(f"📍 使用单卡训练: {self.device}")
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)
            print("⚠️ 未检测到GPU，使用CPU训练")
        
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

        # 早停机制
        self.patience = config['train'].get('patience', 10)
        self.early_stop_counter = 0

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

                # 获取模型输出和门控值
                output, gates = self.model(batch_x, return_gates=True)

                # 检查输出是否有NaN
                if torch.isnan(output).any():
                    print(f"⚠️ 警告: 模型输出包含NaN，跳过此batch")
                    continue

                # 计算复合损失（传入门控值）
                total_loss, l_mse, l_mono, l_exp = self.criterion(output, batch_y, gates)

                # 检查损失是否有NaN
                if torch.isnan(total_loss):
                    print(f"⚠️ 警告: 损失为NaN，跳过此batch")
                    continue

                total_loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
                self.early_stop_counter = 0
                # 如果使用DataParallel，需要保存model.module的state_dict
                model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                torch.save(model_to_save.state_dict(), self.save_path)
                print(f"⭐ 已保存最优模型至 {self.save_path}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"🛑 早停触发：验证损失连续{self.patience}轮未改善")
                    break

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