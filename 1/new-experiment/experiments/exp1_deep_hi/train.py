"""
实验1 深度HI — 训练模块
逐轴承训练 + warmup动态权重 + 留一法交叉验证 + 消融实验
"""

import os
import json
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from typing import Dict, List

from config import (
    EPOCHS, LR_WARMUP, LR_MAX, LR_MIN, WARMUP_EPOCHS, PHYSICS_START,
    LAMBDA_AUG, LAMBDA_SEG_TEMP, LAMBDA_VALUE_MONO, LAMBDA_TREND, LAMBDA_ANCHOR,
    OUTPUT_ROOT, BEARINGS_BY_CONDITION,
    AUG_NOISE_STD, AUG_SCALE_MIN, AUG_SCALE_MAX,
    TRAIN_WINDOWS_PER_BEARING,
)
from model import DeepHIModel
from losses import compute_total_loss


# ============================================================
# 数据增强
# ============================================================

def augment_view(x):
    """对一批窗口做数据增强 (噪声+缩放, 不改变退化程度)."""
    noise = torch.randn_like(x) * AUG_NOISE_STD
    x_aug = x + noise
    scale = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(AUG_SCALE_MIN, AUG_SCALE_MAX)
    return x_aug * scale


# ============================================================
# 动态权重 + 学习率
# ============================================================

def get_training_params(epoch, ablation=None):
    """
    根据epoch和消融配置返回 (lambdas_dict, lr).

    ablation:
      None     → 完整方法 D' (值感知单调性)
      'A'      → 纯CAE
      'B'      → +增强对比
      'C'      → +分段时序对比 (≈Wang)
      'D'      → +全局单调性+趋势
      'D_prime'→ +值感知单调性+趋势 (完整)
    """

    # == 学习率 ==
    if epoch < WARMUP_EPOCHS:
        lr = LR_WARMUP + (LR_MAX - LR_WARMUP) * (epoch / WARMUP_EPOCHS)
    else:
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        lr = LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1 + np.cos(np.pi * progress))

    # == 损失权重 ==
    if ablation == 'A':
        # 纯CAE: 所有对比+物理损失关掉
        return {'aug': 0.0, 'seg': 0.0, 'mono': 0.0, 'trend': 0.0}, lr

    elif ablation == 'B':
        # CAE + 增强对比
        progress = min(1.0, epoch / WARMUP_EPOCHS)
        return {'aug': LAMBDA_AUG * progress, 'seg': 0.0, 'mono': 0.0, 'trend': 0.0}, lr

    elif ablation == 'C':
        # + 分段时序对比
        if epoch < WARMUP_EPOCHS:
            p = epoch / WARMUP_EPOCHS
            return {'aug': LAMBDA_AUG * p, 'seg': 0.0, 'mono': 0.0, 'trend': 0.0, 'anchor': 0.0}, lr
        elif epoch < PHYSICS_START:
            p = (epoch - WARMUP_EPOCHS) / (PHYSICS_START - WARMUP_EPOCHS)
            return {'aug': LAMBDA_AUG, 'seg': LAMBDA_SEG_TEMP * p, 'mono': 0.0, 'trend': 0.0, 'anchor': 0.0}, lr
        else:
            return {'aug': LAMBDA_AUG, 'seg': LAMBDA_SEG_TEMP, 'mono': 0.0, 'trend': 0.0, 'anchor': 0.0}, lr

    else:
        # ablation is None or 'D' or 'D_prime' — 完整方法
        if epoch < WARMUP_EPOCHS:
            p = epoch / WARMUP_EPOCHS
            return {'aug': LAMBDA_AUG * p, 'seg': 0.0, 'mono': 0.0, 'trend': 0.0, 'anchor': LAMBDA_ANCHOR * p}, lr
        elif epoch < PHYSICS_START:
            p = (epoch - WARMUP_EPOCHS) / (PHYSICS_START - WARMUP_EPOCHS)
            return {'aug': LAMBDA_AUG, 'seg': LAMBDA_SEG_TEMP * p, 'mono': 0.0, 'trend': 0.0, 'anchor': LAMBDA_ANCHOR}, lr
        else:
            return {'aug': LAMBDA_AUG, 'seg': LAMBDA_SEG_TEMP,
                    'mono': LAMBDA_VALUE_MONO, 'trend': LAMBDA_TREND, 'anchor': LAMBDA_ANCHOR}, lr


# ============================================================
# 训练一个模型
# ============================================================

def train_model(
    train_datasets: Dict,
    epochs: int = EPOCHS,
    ablation: str = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
):
    """
    逐轴承训练深度HI模型。

    Args:
        train_datasets: {bearing_name: BearingDataset}
        epochs:         训练轮数
        ablation:       消融配置 None/'A'/'B'/'C'/'D'
        device:         训练设备
        verbose:        是否显示进度条

    Returns:
        model, history (list of loss dicts)
    """
    model = DeepHIModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"模型参数量: {total_params:,}")

    optimizer = Adam(model.parameters(), lr=LR_WARMUP)
    train_names = list(train_datasets.keys())
    history = []

    use_value_aware = (ablation != 'D')  # D用全局单调性, D'用值感知

    pbar = tqdm(range(epochs), desc='Training', disable=not verbose)
    for epoch in pbar:
        # 动态参数
        lmbdas, lr = get_training_params(epoch, ablation)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        epoch_losses = {'recon': 0, 'aug': 0, 'seg_temp': 0, 'mono': 0, 'trend': 0, 'anchor': 0, 'total': 0}
        n_bearings = 0

        np.random.shuffle(train_names)

        for name in train_names:
            ds = train_datasets[name]
            data = ds.to_torch(device)
            x_full = data['windows']
            tp_full = data['time_positions']

            T = x_full.shape[0]
            if T < 10:
                continue

            # == 子采样（每epoch每轴承随机采固定数量窗口）==
            if T > TRAIN_WINDOWS_PER_BEARING:
                idx = torch.randperm(T)[:TRAIN_WINDOWS_PER_BEARING].sort()[0]
                x = x_full[idx]
                t_pos = tp_full[idx]
            else:
                x = x_full
                t_pos = tp_full

            optimizer.zero_grad()

            # == 增强视图 ==
            v1 = augment_view(x)  # [N, 1, L]
            v2 = augment_view(x)  # [N, 1, L]

            # == 前向传播 ==
            z1, x_hat1, HI = model(v1, t_pos)

            # 第二个视图: 只用encoder+decoder做重构, HI用第一视图
            z2 = model.encoder(v2)
            z2_smooth = model.temporal_conv(z2)
            x_hat2 = model.decoder(z2_smooth)

            # == 计算损失 ==
            losses = compute_total_loss(
                x=x, x_hat1=x_hat1, x_hat2=x_hat2,
                z1=z1, z2=z2, HI=HI, time_positions=t_pos,
                lambda_aug=lmbdas['aug'],
                lambda_seg=lmbdas['seg'],
                lambda_mono=lmbdas['mono'],
                lambda_trend=lmbdas['trend'],
                lambda_anchor=lmbdas.get('anchor', 0.0),
                use_value_aware=use_value_aware,
            )

            L_total = losses['total']
            if isinstance(L_total, torch.Tensor):
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k in epoch_losses:
                v = losses[k]
                epoch_losses[k] += v.item() if isinstance(v, torch.Tensor) else float(v)
            n_bearings += 1

        if n_bearings > 0:
            for k in epoch_losses:
                epoch_losses[k] /= n_bearings
        history.append(dict(epoch_losses))

        # 更新进度条
        prefix = f"[{ablation}]" if ablation else "[D']"
        pbar.set_postfix({
            f'{prefix}total': f"{epoch_losses['total']:.4f}",
            'recon': f"{epoch_losses['recon']:.4f}",
            'mono': f"{epoch_losses['mono']:.4f}",
            'n': n_bearings,
        })

    model.history = history
    return model


# ============================================================
# 单工况留一法
# ============================================================

def run_condition_cv(
    condition: int,
    all_datasets: Dict,
    ablation: str = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_models: bool = True,
):
    """
    单个工况内的5折留一法交叉验证。

    Returns:
        dict: {test_name: {'HI': array, 'history': list, ...}}
    """
    bearings = BEARINGS_BY_CONDITION[condition]
    results = {}

    for test_idx, test_name in enumerate(bearings):
        print(f"\n{'='*50}")
        print(f"工况{condition} Fold {test_idx+1}/{len(bearings)}")
        print(f"  训练: {[b for b in bearings if b != test_name]}")
        print(f"  测试: {test_name}")
        print(f"{'='*50}")

        train_names = [b for b in bearings if b != test_name]
        train_ds = {n: all_datasets[n] for n in train_names}

        model = train_model(train_ds, ablation=ablation, device=device)

        # 提取HI
        model.eval()
        test_ds = all_datasets[test_name]
        data = test_ds.to_torch(device)
        with torch.no_grad():
            HI = model.extract_hi(data['windows'], data['time_positions']).cpu().numpy()

        results[test_name] = {
            'HI': HI,
            'time_positions': test_ds.time_positions,
            'history': model.history,
            'bearing_name': test_name,
            'condition': condition,
        }

        if save_models:
            save_dir = f"{OUTPUT_ROOT}/checkpoints/cond{condition}"
            if ablation:
                save_dir += f"_{ablation}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/fold_{test_name}.pt")

        # 保存 HI 数据
        hi_dir = f"{OUTPUT_ROOT}/hi_data/cond{condition}"
        if ablation:
            hi_dir += f"_{ablation}"
        os.makedirs(hi_dir, exist_ok=True)
        np.savez(f"{hi_dir}/{test_name}_HI.npz",
                 HI=HI, time_positions=test_ds.time_positions)

    return results


# ============================================================
# 消融实验
# ============================================================

def run_ablation(condition: int, all_datasets: Dict, device='cpu'):
    """
    消融实验: 对比 A, B, C, D, D' 五种配置。
    固定工况1, Bearing1_1测试 → 快速验证各模块贡献。
    """
    bearings = BEARINGS_BY_CONDITION[condition]
    test_name = 'Bearing1_1'
    train_names = [b for b in bearings if b != test_name]
    train_ds = {n: all_datasets[n] for n in train_names}
    test_ds = all_datasets[test_name]

    configs = {
        'A': '纯CAE',
        'B': '+增强对比',
        'C': '+分段时序对比',
        'D': '+全局单调性+趋势',
        "D'": '+值感知单调性+趋势',
    }

    all_results = {}
    for ab, desc in configs.items():
        print(f"\n{'#'*60}")
        print(f"# 消融 {ab}: {desc}")
        print(f"{'#'*60}")

        model = train_model(train_ds, ablation=ab if ab != "D'" else None, device=device)

        # 提取HI
        model.eval()
        data = test_ds.to_torch(device)
        with torch.no_grad():
            HI = model.extract_hi(data['windows'], data['time_positions']).cpu().numpy()

        all_results[ab] = {
            'HI': HI,
            'time_positions': test_ds.time_positions,
            'history': model.history,
            'config': ab,
            'description': desc,
        }

    # 保存结果
    save_dir = f"{OUTPUT_ROOT}/ablation"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(all_results, f"{save_dir}/all_results.pt")

    # 保存HI数据
    for ab, res in all_results.items():
        np.savez(f"{save_dir}/HI_{ab}.npz",
                 HI=res['HI'], time_positions=res['time_positions'])

    print(f"\n消融结果已保存到 {save_dir}/")
    return all_results
