"""
检查预处理数据中是否有NaN或Inf
"""
import numpy as np

print("检查训练集...")
X_train = np.load("./data/processed/X_train.npy")
Y_train = np.load("./data/processed/Y_train.npy")

print(f"X_train形状: {X_train.shape}")
print(f"X_train包含NaN: {np.isnan(X_train).any()}")
print(f"X_train包含Inf: {np.isinf(X_train).any()}")
print(f"X_train范围: [{X_train.min():.4f}, {X_train.max():.4f}]")

print(f"\nY_train形状: {Y_train.shape}")
print(f"Y_train包含NaN: {np.isnan(Y_train).any()}")
print(f"Y_train包含Inf: {np.isinf(Y_train).any()}")
print(f"Y_train范围: [{Y_train.min():.4f}, {Y_train.max():.4f}]")

print("\n检查测试集...")
X_test = np.load("./data/processed/X_test.npy")
Y_test = np.load("./data/processed/Y_test.npy")

print(f"X_test形状: {X_test.shape}")
print(f"X_test包含NaN: {np.isnan(X_test).any()}")
print(f"X_test包含Inf: {np.isinf(X_test).any()}")
print(f"X_test范围: [{X_test.min():.4f}, {X_test.max():.4f}]")

print(f"\nY_test形状: {Y_test.shape}")
print(f"Y_test包含NaN: {np.isnan(Y_test).any()}")
print(f"Y_test包含Inf: {np.isinf(Y_test).any()}")
print(f"Y_test范围: [{Y_test.min():.4f}, {Y_test.max():.4f}]")

# 如果有NaN，找出位置
if np.isnan(X_train).any():
    nan_indices = np.where(np.isnan(X_train))
    print(f"\n❌ X_train中有{len(nan_indices[0])}个NaN值")
    print(f"   位置示例: 样本{nan_indices[0][:5]}, 时间步{nan_indices[1][:5]}, 特征{nan_indices[2][:5]}")

if np.isinf(X_train).any():
    inf_indices = np.where(np.isinf(X_train))
    print(f"\n❌ X_train中有{len(inf_indices[0])}个Inf值")
    print(f"   位置示例: 样本{inf_indices[0][:5]}, 时间步{inf_indices[1][:5]}, 特征{inf_indices[2][:5]}")

if np.isnan(X_test).any():
    nan_indices = np.where(np.isnan(X_test))
    print(f"\n❌ X_test中有{len(nan_indices[0])}个NaN值")
    print(f"   位置示例: 样本{nan_indices[0][:5]}, 时间步{nan_indices[1][:5]}, 特征{nan_indices[2][:5]}")

if np.isinf(X_test).any():
    inf_indices = np.where(np.isinf(X_test))
    print(f"\n❌ X_test中有{len(inf_indices[0])}个Inf值")
    print(f"   位置示例: 样本{inf_indices[0][:5]}, 时间步{inf_indices[1][:5]}, 特征{inf_indices[2][:5]}")
