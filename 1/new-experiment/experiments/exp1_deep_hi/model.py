"""
实验1 深度HI — 模型模块
1D-CNN编码器 + 时序平滑 + 转置卷积解码器 + 时间感知HI投影头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    WINDOW_SIZE, LATENT_DIM, PROJECTOR_HIDDEN,
    ENCODER_CHANNELS, KERNEL_SIZES, STRIDES,
    TEMPORAL_KERNEL_SIZE, TEMPORAL_LAYERS, TIME_EMBED_DIM,
)


class Encoder(nn.Module):
    """1D-CNN编码器: 振动信号 → 退化表征"""

    def __init__(self):
        super().__init__()

        layers = []
        in_ch = 1  # 单通道振动
        for out_ch, k, s in zip(ENCODER_CHANNELS, KERNEL_SIZES, STRIDES):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=0))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.GELU())
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        # 计算卷积输出维度
        conv_out_len = self._compute_output_len(WINDOW_SIZE)

        # 最终投影到latent_dim
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_ch * conv_out_len, LATENT_DIM),
            nn.GELU(),
        )

    def _compute_output_len(self, L):
        for _, k, s in zip(ENCODER_CHANNELS, KERNEL_SIZES, STRIDES):
            L = (L - k) // s + 1
        return L

    def forward(self, x):
        """
        Args:
            x: [B, 1, WINDOW_SIZE]  振动窗口
        Returns:
            z: [B, LATENT_DIM]       退化表征
        """
        h = self.conv(x)
        z = self.project(h)
        return z


class Decoder(nn.Module):
    """转置卷积解码器: 退化表征 → 重构信号

    编码器输出特征图维度: [128, 5] (Flatten前)
    逐层还原: 5→8→23→121→1024
    """

    def __init__(self):
        super().__init__()
        self.enc_out_channels = ENCODER_CHANNELS[-1]  # 128
        self.enc_out_len = 5  # 见 config.py 维度验证

        # latent → 特征图
        self.expand = nn.Sequential(
            nn.Linear(LATENT_DIM, self.enc_out_channels * self.enc_out_len),
            nn.GELU(),
        )

        # 转置卷积层 (反向堆叠)
        # 维度: 5→8→23→121→1024 (含 output_padding 修正)
        self.deconv = nn.Sequential(
            # 5→8: (5-1)*1+4=8 ✓
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            # 8→23: (8-1)*2+8=22, +output_padding=1 → 23 ✓
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),

            # 23→121: (23-1)*4+32=120, +output_padding=1 → 121 ✓
            nn.ConvTranspose1d(32, 16, kernel_size=32, stride=4, output_padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),

            # 121→1024: (121-1)*8+64=1024 ✓
            nn.ConvTranspose1d(16, 1, kernel_size=64, stride=8),
        )

    def forward(self, z):
        """
        Args:
            z: [B, LATENT_DIM]
        Returns:
            x_hat: [B, 1, WINDOW_SIZE]
        """
        h = self.expand(z)
        h = h.view(-1, self.enc_out_channels, self.enc_out_len)
        h = self.deconv(h)
        return h  # 已经精确为 [B, 1, 1024]


class TemporalConv1D(nn.Module):
    """时序卷积平滑模块: 在时间维度上做轻量1D卷积, 让每个窗口的表示融合邻窗信息.

    设计原则:
      - casual=False: 允许看到前后邻窗 (离线HI构建场景)
      - 每层后接 BN + GELU
      - padding=same 保持序列长度不变
      - groups=1 (非depthwise), 让跨通道信息也得到融合

    Input:  [T, D]  编码器输出的特征序列
    Output: [T, D]  时序平滑后的特征序列
    """

    def __init__(self, dim=LATENT_DIM, kernel_size=TEMPORAL_KERNEL_SIZE,
                 num_layers=TEMPORAL_LAYERS):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = dim
            padding = kernel_size // 2  # same padding
            layers.append(nn.Conv1d(in_ch, dim, kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.GELU())
        self.conv = nn.Sequential(*layers)

    def forward(self, z):
        """
        Args:
            z: [T, D]  时间有序的特征序列
        Returns:
            z_smooth: [T, D]  时序平滑后的特征
        """
        # [T, D] → [1, D, T] for Conv1d
        z_t = z.permute(1, 0).unsqueeze(0)  # [1, D, T]
        z_t = self.conv(z_t)                 # [1, D, T]
        return z_t.squeeze(0).permute(1, 0)  # [T, D]


class TimeAwareProjector(nn.Module):
    """时间感知HI投影头: 融合信号特征 + 时间位置 → 健康指标.

    设计理由:
      显式注入归一化时间位置, 给模型"时间感"。
      信号特征来自时序平滑后的表征, 时间位置来自元数据。
      两者通过可学习的权重混合, 而非简单拼接 —
      模型自己决定在多大程度上依赖信号 vs 时间先验。

    Input:
      z:     [T, D]  时序平滑后的特征
      t_pos: [T]     归一化时间位置 [0, 1]
    Output:
      HI: [T]  健康指标, [0, 1]
    """

    def __init__(self, dim=LATENT_DIM, hidden=PROJECTOR_HIDDEN,
                 time_dim=TIME_EMBED_DIM):
        super().__init__()
        # 时间位置嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
        )
        # 信号特征投影
        self.signal_fc = nn.Linear(dim, hidden)
        # 时间特征投影
        self.time_fc = nn.Linear(time_dim, hidden)
        # 融合门控: 学习信号 vs 时间的混合权重
        self.gate = nn.Sequential(
            nn.Linear(dim + time_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        # 输出层
        self.out = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()

    def forward(self, z, t_pos):
        """
        Args:
            z:     [T, D]  时序平滑特征
            t_pos: [T]     归一化时间位置
        Returns:
            HI: [T]  健康指标
        """
        # 时间嵌入
        t_emb = self.time_embed(t_pos.unsqueeze(-1))  # [T, time_dim]

        # 信号分支
        h_signal = self.act(self.signal_fc(z))  # [T, hidden]

        # 时间分支
        h_time = self.act(self.time_fc(t_emb))  # [T, hidden]

        # 门控混合: gate ∈ [0, 1], gate=1 偏向时间, gate=0 偏向信号
        gate_input = torch.cat([z, t_emb], dim=-1)  # [T, D+time_dim]
        gate = self.gate(gate_input)                 # [T, 1]

        h = gate * h_time + (1 - gate) * h_signal    # [T, hidden]

        return self.out(h).squeeze(-1)  # [T]


class DeepHIModel(nn.Module):
    """完整的深度HI模型 (v2: +时序上下文)

    Pipeline:
      x [T,1,L] → Encoder → z [T,D] → TemporalConv1D → z' [T,D]
                                      ├→ Decoder → x̂ [T,1,L]
                                      └→ TimeAwareProjector(z', t_pos) → HI [T]
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.temporal_conv = TemporalConv1D()
        self.decoder = Decoder()
        self.projector = TimeAwareProjector()

    def forward(self, x, time_positions=None):
        """
        Args:
            x:              [T, 1, WINDOW_SIZE]
            time_positions: [T]  归一化时间位置 (可为None, 回退到旧行为)
        Returns:
            z:     [T, LATENT_DIM]  编码特征 (时序平滑前)
            x_hat: [T, 1, WINDOW_SIZE]
            HI:    [T]
        """
        z_raw = self.encoder(x)              # [T, D]
        z = self.temporal_conv(z_raw)        # [T, D]  时序平滑
        x_hat = self.decoder(z)
        if time_positions is None:
            time_positions = torch.linspace(0, 1, x.shape[0], device=x.device)
        HI = self.projector(z, time_positions)
        return z, x_hat, HI

    def extract_hi(self, x, time_positions=None):
        """推理时仅提取HI"""
        z_raw = self.encoder(x)
        z = self.temporal_conv(z_raw)
        if time_positions is None:
            time_positions = torch.linspace(0, 1, x.shape[0], device=x.device)
        return self.projector(z, time_positions)


if __name__ == '__main__':
    print("=" * 60)
    print("测试模型模块")
    print("=" * 60)

    model = DeepHIModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 测试前向传播
    x = torch.randn(16, 1, WINDOW_SIZE)
    z, x_hat, HI = model(x)
    print(f"\ninput:     {x.shape}")
    print(f"z:         {z.shape}")
    print(f"x_recon:   {x_hat.shape}")
    print(f"HI:        {HI.shape}, range=[{HI.min():.3f}, {HI.max():.3f}]")

    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} vs {x.shape}"
    print("Model test PASSED")
