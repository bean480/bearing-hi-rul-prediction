"""VMD (Variational Mode Decomposition) 实现
参考: Dragomiretskiy & Zosso (2014), IEEE TSP
"""

import numpy as np
from scipy.fft import fft, ifft

def vmd(signal, alpha=2000, tau=0, K=5, DC=False, init=1, tol=1e-7, max_iter=500):
    """
    Args:
        signal: 1D 输入信号 [N]
        alpha:  带宽约束参数 (推荐 2000)
        tau:    噪声容忍度 (0 = 无噪声容忍)
        K:      分解模态数
        DC:     是否包含直流分量
        init:   初始化方式 (1 = 均匀分布 omega)
        tol:    收敛容差
        max_iter: 最大迭代次数
    Returns:
        u:     [K, N] 各模态时域信号
        u_hat: [K, N//2+1] 各模态频域表示
        omega: [K] 各模态中心频率 (归一化, 0~0.5)
    """
    N = len(signal)
    N2 = N // 2 + 1
    
    # 频域
    f_hat = fft(signal)[:N2]  # [N2], 只取正半轴
    
    # 初始化: omega 均匀分布在 [0, 0.5]
    omega = np.linspace(0, 0.5 - 0.5/K, K)
    
    # 初始化: 各模态频域表示
    u_hat = np.zeros((K, N2), dtype=complex)
    for k in range(K):
        # 在 omega_k 附近初始化窄带分量
        center_bin = int(omega[k] * (N2 - 1))
        half_width = max(5, N2 // (4 * K))
        lo = max(0, center_bin - half_width)
        hi = min(N2, center_bin + half_width)
        u_hat[k, lo:hi] = f_hat[lo:hi] / K
    
    # 拉格朗日乘子
    lambda_hat = np.zeros(N2, dtype=complex)
    
    # 频率轴
    freqs = np.arange(N2) / (N2 - 1) * 0.5  # [0, 0.5]
    
    # 迭代
    for n in range(max_iter):
        u_hat_old = u_hat.copy()
        
        # 逐模态更新
        for k in range(K):
            # 当前残差
            sum_others = f_hat.copy()
            for j in range(K):
                if j != k:
                    sum_others -= u_hat[j]
            sum_others += lambda_hat / 2
            
            # Wiener 滤波: u_hat_k = sum_others / (1 + 2*alpha*(omega - omega_k)^2)
            filter_denom = 1 + 2 * alpha * (freqs - omega[k])**2
            u_hat[k] = sum_others / filter_denom
            
            # 更新中心频率 omega_k
            power = np.abs(u_hat[k])**2
            omega[k] = np.sum(freqs * power) / (np.sum(power) + 1e-15)
        
        # 更新拉格朗日乘子
        if tau > 0:
            sum_u = np.sum(u_hat, axis=0)
            lambda_hat = lambda_hat + tau * (f_hat - sum_u)
        
        # 收敛检查
        diff = np.sum(np.abs(u_hat - u_hat_old)**2, axis=1)
        rel_change = np.sum(diff) / (np.sum(np.abs(u_hat)**2) + 1e-15)
        
        if rel_change < tol and n > 10:
            break
    
    # 回到时域
    u = np.zeros((K, N))
    for k in range(K):
        u_full = np.zeros(N, dtype=complex)
        u_full[:N2] = u_hat[k]
        # 共轭对称填充负半轴
        if N % 2 == 0:
            u_full[N2:] = np.conj(u_hat[k, -2:0:-1])
        else:
            u_full[N2:] = np.conj(u_hat[k, -1:0:-1])
        u[k] = np.real(ifft(u_full))
    
    omega_hz = omega * (len(signal) / (2 * (N2 - 1)))  # 转换为归一化频率
    
    return u, u_hat, omega, omega_hz
