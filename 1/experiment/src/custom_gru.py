import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomGRUCell(nn.Module):
    """
    自定义GRU单元，支持门控激活值提取

    GRU公式：
    r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  # 重置门
    z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  # 更新门
    n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  # 新记忆
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}  # 隐藏状态
    """
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 重置门参数
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)

        # 更新门参数
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)

        # 新记忆参数
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        """
        Args:
            x: (batch, input_size)
            h_prev: (batch, hidden_size)
        Returns:
            h_new: (batch, hidden_size)
            gates: dict with 'reset_gate' and 'update_gate'
        """
        # 重置门
        r_t = torch.sigmoid(self.W_ir(x) + self.W_hr(h_prev))

        # 更新门
        z_t = torch.sigmoid(self.W_iz(x) + self.W_hz(h_prev))

        # 新记忆
        n_t = torch.tanh(self.W_in(x) + r_t * self.W_hn(h_prev))

        # 新隐藏状态
        h_new = (1 - z_t) * n_t + z_t * h_prev

        # 返回门控值
        gates = {
            'reset_gate': r_t,
            'update_gate': z_t,
            'new_memory': n_t
        }

        return h_new, gates


class CustomGRU(nn.Module):
    """
    自定义GRU层，支持双向和多层
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=True, bidirectional=False):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 创建GRU单元
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                if layer == 0:
                    cell_input_size = input_size
                else:
                    cell_input_size = hidden_size * self.num_directions
                self.cells.append(CustomGRUCell(cell_input_size, hidden_size))

    def forward(self, x, h_0=None, return_gates=False):
        """
        Args:
            x: (batch, seq_len, input_size) if batch_first=True
            h_0: (num_layers * num_directions, batch, hidden_size)
            return_gates: 是否返回门控激活值
        Returns:
            output: (batch, seq_len, hidden_size * num_directions)
            h_n: (num_layers * num_directions, batch, hidden_size)
            gates: dict (仅当 return_gates=True)
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # 转为 batch_first

        # 初始化隐藏状态
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * self.num_directions,
                             batch_size, self.hidden_size,
                             device=x.device, dtype=x.dtype)

        # 存储所有时间步的输出和门控值
        outputs = []
        all_gates = {
            'reset_gate': [],
            'update_gate': [],
            'new_memory': []
        } if return_gates else None

        # 逐层处理
        layer_input = x
        h_n_list = []

        for layer in range(self.num_layers):
            layer_output_forward = []
            layer_output_backward = []

            # 前向传播
            h_prev = h_0[layer * self.num_directions]
            cell_idx = layer * self.num_directions

            for t in range(seq_len):
                h_new, gates = self.cells[cell_idx](layer_input[:, t, :], h_prev)
                layer_output_forward.append(h_new)

                if return_gates and layer == self.num_layers - 1:  # 只保存最后一层
                    all_gates['reset_gate'].append(gates['reset_gate'])
                    all_gates['update_gate'].append(gates['update_gate'])
                    all_gates['new_memory'].append(gates['new_memory'])

                h_prev = h_new

            h_n_list.append(h_prev.unsqueeze(0))

            # 反向传播（如果是双向）
            if self.bidirectional:
                h_prev = h_0[layer * self.num_directions + 1]
                cell_idx = layer * self.num_directions + 1

                for t in range(seq_len - 1, -1, -1):
                    h_new, _ = self.cells[cell_idx](layer_input[:, t, :], h_prev)
                    layer_output_backward.append(h_new)
                    h_prev = h_new

                h_n_list.append(h_prev.unsqueeze(0))
                layer_output_backward = layer_output_backward[::-1]  # 反转

            # 拼接前向和反向输出
            if self.bidirectional:
                layer_output = torch.stack([
                    torch.cat([layer_output_forward[t], layer_output_backward[t]], dim=1)
                    for t in range(seq_len)
                ], dim=1)
            else:
                layer_output = torch.stack(layer_output_forward, dim=1)

            layer_input = layer_output

        output = layer_output
        h_n = torch.cat(h_n_list, dim=0)

        # 整理门控值
        if return_gates:
            all_gates['reset_gate'] = torch.stack(all_gates['reset_gate'], dim=1)  # (batch, seq_len, hidden)
            all_gates['update_gate'] = torch.stack(all_gates['update_gate'], dim=1)
            all_gates['new_memory'] = torch.stack(all_gates['new_memory'], dim=1)
            return output, h_n, all_gates

        return output, h_n
