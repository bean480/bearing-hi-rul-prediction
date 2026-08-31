import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 单卡训练：只使用第一张卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import yaml
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from src.data_builder import PHM2012DataBuilder
from src.models import BearingRULModel
from src.losses import PINNLoss
from src.trainer import RULTrainer
from src.uq_evaluator import RULEvaluator

def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子已设置: {seed}")

def load_data():
    X_train = np.load("./data/processed/X_train.npy")
    Y_train = np.load("./data/processed/Y_train.npy")
    X_test = np.load("./data/processed/X_test.npy")
    Y_test = np.load("./data/processed/Y_test.npy")
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))
    return train_ds, test_ds

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="轴承 RUL 预测实验平台")
    parser.add_argument('--mode', type=str, default='eval',
                        choices=['data', 'train', 'eval'],
                        help='执行模式: data(预处理), train(训练), eval(评估可视化)')
    args = parser.parse_args()

    # 2. 加载全局配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 3. 设置随机种子（确保可复现性）
    seed = config['train'].get('seed', 42)
    set_seed(seed)

    # 4. 根据模式执行不同逻辑
    if args.mode == 'data':
        print("🛠️ 模式：数据预处理...")
        builder = PHM2012DataBuilder(config)
        builder.process()

    elif args.mode == 'train':
        print("🚀 模式：模型训练...")

        # 验证多卡配置
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"✅ 检测到 {num_gpus} 张GPU:")
            for i in range(num_gpus):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            num_gpus = 1
            print("⚠️ 未检测到GPU")

        train_ds, test_ds = load_data()

        # 单卡训练，使用原始batch_size
        effective_batch_size = config['train']['batch_size']
        print(f"📦 Batch size: {effective_batch_size}")

        train_loader = DataLoader(train_ds, batch_size=effective_batch_size,
                                 shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(test_ds, batch_size=effective_batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)

        model = BearingRULModel(input_features=7, seq_len=30)
        criterion = PINNLoss(lambda_mono=0.1, lambda_exp=0.05)
        trainer = RULTrainer(model, criterion, config, use_multi_gpu=True)
        trainer.train(train_loader, val_loader)

    elif args.mode == 'eval':
        print("📊 模式：不确定性评估与可视化...")
        # 仅加载测试集
        X_test = np.load("./data/processed/X_test.npy")
        Y_test = np.load("./data/processed/Y_test.npy")
        
        model = BearingRULModel(input_features=7, seq_len=30)
        evaluator = RULEvaluator(model, config)
        
        # 执行带有 MC Dropout 的预测
        # mean_pred, std_pred = evaluator.predict_with_uncertainty(X_test, n_iter=50)
        mean_pred, std_pred = evaluator.predict_with_uncertainty(X_test, n_iter=50)
        evaluator.visualize_result(Y_test, mean_pred, std_pred)
        # === 【新增：注意力热力图可视化】 ===
        print("\n" + "="*40)
        print("🧠 启动可解释性分析 (Explainability Analysis)...")
        from src.attention_visualizer import AttentionVisualizer
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        visualizer = AttentionVisualizer(model, device)
        # 传入测试集数据绘制热力图
        visualizer.visualize_lifespan_attention(X_test)

if __name__ == "__main__":
    main()