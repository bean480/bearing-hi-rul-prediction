# 轴承健康指标构建与剩余寿命预测研究

基于"机理-数据-模型协同驱动"框架的滚动轴承剩余使用寿命（RUL）预测研究仓库。

## 目录结构

```
├── README.md                          # 本文件
├── config.yaml                        # 根目录 RUL 预测实验配置
├── main.py                            # 根目录 RUL 预测主程序（PHM2012 轴承数据）
├── check_features.py                  # 特征检查脚本
├── src/                               # 根目录 RUL 预测核心代码
│   ├── data_builder.py                # PHM2012 数据加载与预处理
│   ├── models.py                      # 轴承 RUL 预测模型（BearingRULModel）
│   ├── losses.py                      # 损失函数（PINN、非对称损失）
│   ├── trainer.py                     # 训练器
│   ├── metrics.py                     # 评估指标（RMSE、Asymmetric Score）
│   ├── uq_evaluator.py                # 不确定性量化评估
│   └── attention_visualizer.py        # 注意力可视化
│
├── experiments-gating/                # 【实验】BiGRU 门控机制可解释性分析
│   ├── main.py                        # 门控实验主程序
│   ├── visualize_gates.py             # 门控激活值可视化
│   ├── diagnose_fpt.py                # 首预测点（FPT）诊断
│   └── src/                           # 含 custom_gru.py、gate_visualizer.py
│
├── experiments-health-hi/             # 【实验】深度健康指标（HI）构建（当前主线）
│   ├── docs/                          # 实验方案、大纲、文献汇总
│   ├── experiments/                   # 各子实验
│   │   ├── exp_vmd_test/              # VMD 分解与故障频带重构
│   │   ├── exp_feature_pool/          # 特征池构建
│   │   ├── exp_feature_screening/     # 特征筛选（单调性/趋势性/物理性）
│   │   ├── exp1_health_indicator/     # 健康指标计算与融合
│   │   └── exp1_deep_hi/              # 深度学习健康指标构建
│   └── figures/                       # 结果图
│
├── model-comparison/                  # 【实验】多模型对比（CNN/GRU/LSTM/Transformer 等）
│   ├── api_test.py                    # API 测试脚本
│   ├── test.py                        # 测试脚本
│   ├── CNN预测模型/                    # 各模型训练 notebook（.ipynb）
│   ├── BiGRU预测模型/
│   ├── ...                            # 其余模型目录
│   └── 相关可视化/                     # 模型性能对比图
│
└── papers/                            # 论文与参考文献
```

## 实验阶段

- **阶段 1（已完成）**：PHM2012 轴承 RUL 预测基础实验 + BiGRU 门控机制可解释性分析 + 多模型对比

## 数据说明

- 原始数据、预处理缓存、`.npz`/`.npy`/`.csv` 数据文件均不纳入版本管理
- 模型权重（`.pth`/`.pt`）、`.pkl`、图表（`.png`）不纳入版本管理
- 详细忽略规则见 `.gitignore`
