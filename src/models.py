import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 一维 CBAM 注意力机制 (可解释性的核心)
# ==========================================
class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 共享的多层感知机
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM1D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention1D(in_planes, ratio)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        # 通道注意力加权
        x = x * self.ca(x)
        # 空间(时序)注意力加权
        x = x * self.sa(x)
        return x

# ==========================================
# 2. 完整 RUL 预测模型
# ==========================================
class BearingRULModel(nn.Module):
    def __init__(self, input_features=7, seq_len=30, hidden_dim=64, dropout_rate=0.3):
        super(BearingRULModel, self).__init__()
        
        # A. 多尺度 1D-CNN (捕捉不同频段的局部冲击特征)
        # 输入 shape: (Batch, Channels, Seq_len) -> PyTorch 中 1D CNN 的 Channel 在中间
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_features, out_channels=16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=input_features, out_channels=16, kernel_size=7, padding=3)
        
        # 拼接后的通道数: 16 + 16 + 16 = 48
        cnn_out_channels = 48
        
        # B. 接入 CBAM 注意力机制
        # ratio 设为 8 以适配 48 通道
        self.cbam = CBAM1D(in_planes=cnn_out_channels, ratio=8) 
        
        # C. BiGRU 时序建模
        # CNN 输出后，需要转置回 (Batch, Seq_len, Features) 给 RNN
        self.bigru = nn.GRU(input_size=cnn_out_channels, 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True)
        
        # D. 预测头与 MC Dropout (不确定性量化的核心)
        # BiGRU 输出维度是 hidden_dim * 2
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        
        # 关键点：这个 Dropout 层在推理时也必须保持激活！
        self.dropout = nn.Dropout(p=dropout_rate) 
        
        # 输出单点 RUL (使用 Softplus 保证输出为正数，辅助单调性)
        self.fc2 = nn.Linear(32, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x 初始形状: (Batch, Seq_len, Features)
        # 转换形状适配 1D CNN: (Batch, Features, Seq_len)
        x = x.permute(0, 2, 1)
        
        # 1. 多尺度特征提取
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))
        
        # 沿通道维度拼接
        x_multi = torch.cat([out1, out2, out3], dim=1) # Shape: (Batch, 48, Seq_len)
        
        # 2. 注意力加权 (后续可以提取 self.cbam.ca 和 sa 的权重做可视化)
        x_cbam = self.cbam(x_multi)
        
        # 3. 转换形状适配 BiGRU: (Batch, Seq_len, 48)
        x_rnn_in = x_cbam.permute(0, 2, 1)
        
        # RNN 输出: rnn_out (Batch, Seq_len, hidden_dim*2), h_n (num_layers*2, Batch, hidden_dim)
        rnn_out, _ = self.bigru(x_rnn_in)
        
        # 取序列的最后一个时间步特征作为整个窗口的表征
        last_step_out = rnn_out[:, -1, :] # Shape: (Batch, hidden_dim*2)
        
        # 4. 全连接与 MC Dropout 预测
        feat = F.relu(self.fc1(last_step_out))
        feat = self.dropout(feat) # MC Dropout 层
        
        rul_pred = self.fc2(feat)
        # 强制输出非负数 (物理约束的第一层)
        rul_pred = self.softplus(rul_pred) 
        
        # 降维 (Batch, 1) -> (Batch,)
        return rul_pred.squeeze()