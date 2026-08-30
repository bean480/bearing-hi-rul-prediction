"""
实验1 深度HI — 损失函数模块 (优化版)
采样子+矢量化, 适配 T=5000~8000 的大窗口数
"""

import torch
import torch.nn.functional as F

from config import TAU, DELTA_POS, SEG_WINDOWS_MIN, MONO_EPS

# 子采样上限 — 避免 O(T^2) 爆炸
MAX_WINDOWS_AUG = 1500      # 增强对比最大窗口数
MAX_WINDOWS_SEG = 2000      # 时序对比最大窗口数
MAX_POS_PAIRS    = 3000     # 时序对比最大正样本对


# ============================================================
# 工具函数
# ============================================================

def _subsample(z1, z2, HI_ref, time_positions, max_n):
    """随机子采样, 保证所有tensor同步"""
    T = z1.shape[0]
    if T <= max_n:
        return z1, z2, HI_ref, time_positions, torch.arange(T)

    idx = torch.randperm(T)[:max_n].sort()[0]
    return z1[idx], z2[idx], HI_ref[idx], time_positions[idx], idx


# ============================================================
# 1. 重构损失
# ============================================================

def reconstruction_loss(x, x_hat1, x_hat2):
    loss = F.mse_loss(x_hat1, x) + F.mse_loss(x_hat2, x)
    return loss / 2


# ============================================================
# 2. 增强视图对比损失 (矢量化)
# ============================================================

def aug_contrastive_loss(z1, z2, tau=TAU):
    """
    z1, z2: [T, D] — 调用前应已子采样到 MAX_WINDOWS_AUG
    """
    T = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)           # [2T, D]
    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / tau                     # [2T, 2T]

    pos_mask = torch.zeros(2*T, 2*T, device=z.device)
    for i in range(T):
        pos_mask[i, i+T] = 1
        pos_mask[i+T, i] = 1

    self_mask = torch.eye(2*T, device=z.device)

    pos_sim = (sim * pos_mask).sum(dim=1)
    all_sim = (sim.exp() * (1 - self_mask)).sum(dim=1)
    loss = (-pos_sim + torch.log(all_sim + 1e-8)).mean()
    return loss


# ============================================================
# 3. 分段时序对比损失 (矢量化 + 子采样)
# ============================================================

def segment_temporal_contrast_loss(z, time_positions, tau=TAU,
                                     delta_pos=DELTA_POS,
                                     max_windows=MAX_WINDOWS_SEG,
                                     max_pairs=MAX_POS_PAIRS):
    """
    自适应分段 + 段内邻近窗口正样本。

    优化: 子采样窗口 + 矢量化相似度矩阵
    """
    T, D = z.shape

    # 子采样
    if T > max_windows:
        idx = torch.randperm(T)[:max_windows].sort()[0]
        z = z[idx]
        time_positions = time_positions[idx]
        T = len(z)

    K = max(3, min(10, T // SEG_WINDOWS_MIN))
    if T < K * 2:
        return torch.tensor(0.0, device=z.device)

    z = F.normalize(z, dim=1)                      # [T, D]
    segment_ids = (time_positions * K).long().clamp(0, K - 1)

    # 全局相似度矩阵: cos(z_i, z_j) / tau
    sim = (z @ z.T) / tau                           # [T, T]

    # 索引距离矩阵 (正比于时间距离，因为子采样保留了排序)
    idx_diff = (torch.arange(T, device=z.device).unsqueeze(0) -
                torch.arange(T, device=z.device).unsqueeze(1)).abs().float()

    loss = 0.0
    count = 0

    for seg in range(K):
        mask = (segment_ids == seg)
        indices = mask.nonzero(as_tuple=True)[0]
        n = len(indices)
        if n < 2:
            continue

        # 段内子矩阵
        seg_sim = sim[indices][:, indices]          # [n, n]
        seg_idx_diff = idx_diff[indices][:, indices]  # [n, n]

        # 正样本掩码: 索引距离 <= delta_pos，不包含自身
        pos = (seg_idx_diff > 0) & (seg_idx_diff <= delta_pos)

        # 如果正样本对太多，随机子采样
        pos_pairs = pos.nonzero()
        if len(pos_pairs) > max_pairs:
            keep = torch.randperm(len(pos_pairs))[:max_pairs]
            pos_pairs = pos_pairs[keep]

        if len(pos_pairs) == 0:
            continue

        # 提取所有正样本对的元素
        anchors = pos_pairs[:, 0]
        positives = pos_pairs[:, 1]

        # 正样本相似度
        pos_sim_vals = seg_sim[anchors, positives]

        # 负样本: 对每个锚点，取所有非自身+非正样本的窗口
        # 矢量化: exp(sim) 求和 - exp(pos_sim)
        exp_sim_full = seg_sim.exp()

        # 对每个锚点，求和所有非自身的exp(sim)
        exp_all = exp_sim_full.sum(dim=1).unsqueeze(1)                # [n, 1]
        exp_self = exp_sim_full[torch.arange(n), torch.arange(n)].unsqueeze(1)  # [n, 1]
        # 近似: exp_all - exp_self (去除了自身，正样本对微量偏差可忽略)
        exp_neg = exp_all - exp_self

        # 取每个锚点对应的负样本和
        anchor_exp_neg = exp_neg[anchors].squeeze()   # [num_pairs]

        loss += (-pos_sim_vals + torch.log(anchor_exp_neg + 1e-8)).sum()
        count += len(pos_pairs)

    if count == 0:
        return torch.tensor(0.0, device=z.device)

    return loss / count


# ============================================================
# 4. 值感知单调性损失
# ============================================================

def value_aware_monotonicity_loss(HI, eps=MONO_EPS):
    """HI越高, 下降惩罚越重 (O(T), 不需要优化)"""
    if len(HI) < 2:
        return torch.tensor(0.0, device=HI.device)
    diff = HI[1:] - HI[:-1]
    hinge = F.relu(-diff + eps)
    weights = HI[1:] ** 1.5              # 比 HI^2 温和，避免早期约束过弱
    return (weights * hinge).mean()


def global_monotonicity_loss(HI, eps=MONO_EPS):
    """全局单调性 (消融D对比项)"""
    if len(HI) < 2:
        return torch.tensor(0.0, device=HI.device)
    return F.relu(HI[:-1] - HI[1:] + eps).mean()


# ============================================================
# 5. 趋势一致性损失
# ============================================================

def trend_consistency_loss(HI, time_positions):
    """L = 1 - Pearson(HI, t)"""
    if len(HI) < 2:
        return torch.tensor(0.0, device=HI.device)
    HI_c, t_c = HI - HI.mean(), time_positions - time_positions.mean()
    cov = (HI_c * t_c).sum()
    rho = cov / (HI_c.norm() * t_c.norm() + 1e-8)
    return 1.0 - rho.clamp(-1, 1)


# ============================================================
# 总损失组装
# ============================================================

# ============================================================
# 6. 锚点损失
# ============================================================

def anchor_loss(HI, k_early=None, k_late=None):
    """
    锚点约束 (自监督): 早期HI≈0, 末期HI≈1
    利用全寿命实验设计自带的"从健康到失效"的先验

    L = mean(HI[:k])² + (1 - mean(HI[-k:]))²

    不需要人工标注, 物理依据:
      轴承加速退化实验: 第1个CSV = 刚安装, 最后1个CSV = 失效停止
    """
    T = len(HI)
    if k_early is None:
        k_early = max(1, int(T * 0.05))   # 前5%窗口 = 健康
    if k_late is None:
        k_late  = max(1, int(T * 0.05))   # 后5%窗口 = 失效

    L_early = HI[:k_early].mean() ** 2
    L_late  = (1.0 - HI[-k_late:].mean()) ** 2
    return L_early + L_late


# ============================================================
# 总损失组装
# ============================================================

def compute_total_loss(
    x, x_hat1, x_hat2, z1, z2, HI, time_positions,
    lambda_aug, lambda_seg, lambda_mono, lambda_trend, lambda_anchor=0.0,
    use_value_aware=True,
):
    """
    Args:
        x, x_hat1, x_hat2: [T, 1, L]
        z1, z2:            [T, D]
        HI:                [T]
        time_positions:    [T]
        lambda_anchor:     锚点损失权重
    Returns:
        dict of losses
    """
    # 子采样 (用于对比损失)
    z1_sub, z2_sub, hi_sub, tp_sub, _ = _subsample(
        z1, z2, HI, time_positions, MAX_WINDOWS_AUG
    )

    L_recon = reconstruction_loss(x, x_hat1, x_hat2)

    L_aug = aug_contrastive_loss(z1_sub, z2_sub) if lambda_aug > 0 else 0.0

    L_seg = segment_temporal_contrast_loss(z1_sub, tp_sub) if lambda_seg > 0 else 0.0

    if lambda_mono > 0:
        L_mono = (value_aware_monotonicity_loss if use_value_aware
                  else global_monotonicity_loss)(HI)
    else:
        L_mono = 0.0

    L_trend = trend_consistency_loss(HI, time_positions) if lambda_trend > 0 else 0.0

    L_anchor = anchor_loss(HI) if lambda_anchor > 0 else 0.0

    L_total = (
        L_recon
        + lambda_aug * L_aug
        + lambda_seg * L_seg
        + lambda_mono * L_mono
        + lambda_trend * L_trend
        + lambda_anchor * L_anchor
    )

    def _t(v):
        if isinstance(v, torch.Tensor): return v
        return torch.tensor(float(v), device=x.device)

    return {
        'total': L_total,
        'recon': _t(L_recon),
        'aug': _t(L_aug),
        'seg_temp': _t(L_seg),
        'mono': _t(L_mono),
        'trend': _t(L_trend),
        'anchor': _t(L_anchor),
    }
