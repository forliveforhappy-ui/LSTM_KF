import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMKFNet(nn.Module):
    def __init__(self,
                 input_size=56,       # 28 个关键点 * 2
                 hidden_size=256,
                 num_layers=3,
                 bidirectional=True,
                 dropout=0.2,
                 mlp_hidden=512,
                 use_boundary_skip=True,
                 num_points=28,
                 eps=1e-6):
        super(LSTMKFNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_boundary_skip = use_boundary_skip
        self.num_points = num_points
        self.state_dim = 4  # x,y,vx,vy
        self.obs_dim = 2    # x,y
        self.eps = eps

        # LSTM 主干网络
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 池化最后一层后的特征维度
        feat_dim = hidden_size * self.num_directions
        if use_boundary_skip:
            feat_dim += input_size * 2

        # 头部网络
        # 先验状态: 预测 (num_points * state_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_points * self.state_dim)
        )

        # 预测 Q（过程噪声）的对角线（log尺度），每个关键点每个状态维度
        self.q_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_points * self.state_dim)  # log 方差
        )

        # 预测 R（观测噪声）的对角线（log尺度），每个关键点每个观测维度
        self.r_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_points * self.obs_dim)    # log 方差
        )

        self._init_weights()

        # 静态 H 矩阵（obs = H x）
        H = torch.zeros((self.obs_dim, self.state_dim), dtype=torch.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        self.register_buffer('H', H)  # (2,4)

    def _init_weights(self):
        # LSTM 初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 遗忘门
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

        # MLP 头部初始化
        for head in [self.prior_head, self.q_head, self.r_head]:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x, meas=None, return_dict=True):
        """
        x: (B, T, input_size)
        meas: 可选观测，形状为 (B, num_points*2) 或 (B, num_points, 2)
              观测需与 data_loader 的归一化坐标空间一致（即每个目标帧归一化坐标）

        返回包含先验/后验状态及相关矩阵的字典。
        """
        B = x.size(0)
        # LSTM 前向
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, hidden_size)
        h_n = h_n.view(self.num_layers, self.num_directions, B, self.hidden_size)
        last_layer_h = h_n[-1]  # (num_directions, B, hidden_size)
        last_hidden = last_layer_h.permute(1, 0, 2).reshape(B, -1)  # (B, hidden_size * num_directions)

        # 边界跳跃特征
        if self.use_boundary_skip:
            first_frame = x[:, 0, :]
            last_frame = x[:, -1, :]
            feats = torch.cat([last_hidden, first_frame, last_frame], dim=1)
        else:
            feats = last_hidden

        # 头部网络
        prior = self.prior_head(feats)  # (B, K*4)
        q_log = self.q_head(feats)      # (B, K*4)
        r_log = self.r_head(feats)      # (B, K*2)

        K = self.num_points
        # 变形
        x_prior = prior.view(B, K, self.state_dim)       # (B, K, 4)
        Q_diag = torch.exp(q_log).view(B, K, self.state_dim) + self.eps  # (B,K,4) 保证正数
        R_diag = torch.exp(r_log).view(B, K, self.obs_dim) + self.eps    # (B,K,2)

        # 方便使用的先验坐标
        coords_prior = x_prior[..., :2].reshape(B, K * 2)

        out = {
            'x_prior': x_prior,
            'coords_prior': coords_prior,
            'Q_diag': Q_diag,
            'R_diag': R_diag
        }

        if meas is None:
            if return_dict:
                return out
            else:
                return coords_prior

        # 处理观测
        if meas.dim() == 2:
            meas = meas.view(B, K, self.obs_dim)
        elif meas.dim() == 3:
            # 假定为 (B,K,2)
            pass
        else:
            raise ValueError(f"meas 必须为 (B,K,2) 或 (B,K*2)，但得到 {meas.shape}")

        z = meas

        # 构造 P_pred 为对角矩阵
        # P_pred: (B, K, 4, 4)
        P_pred = torch.diag_embed(Q_diag)  # 用 Q_diag 作为先验协方差对角线

        # R 矩阵: (B, K, 2, 2)
        R_mat = torch.diag_embed(R_diag)

        # 计算 PHT = P_pred @ H.T  -> (B,K,4,2)
        H_t = self.H.t()  # (4,2)
        PHT = torch.matmul(P_pred, H_t)  # 广播矩阵乘法

        # S = H @ PHT + R  -> (B,K,2,2)
        S = torch.matmul(self.H, PHT) + R_mat
        # 稳定 S，在对角线上加小的 eps
        eye2 = torch.eye(self.obs_dim, device=S.device, dtype=S.dtype).view(1, 1, self.obs_dim, self.obs_dim)
        S = S + self.eps * eye2

        # S 求逆
        S_inv = torch.inverse(S)

        # 卡尔曼增益 K_gain = PHT @ S_inv -> (B,K,4,2)
        K_gain = torch.matmul(PHT, S_inv)

        # 创新量 y = z - H x_prior
        x_prior_unsq = x_prior.unsqueeze(-1)  # (B,K,4,1)
        Hx = torch.matmul(self.H, x_prior_unsq).squeeze(-1)  # (B,K,2)
        y = z - Hx  # (B,K,2)

        # x_post = x_prior + K_gain @ y
        y_unsq = y.unsqueeze(-1)  # (B,K,2,1)
        K_y = torch.matmul(K_gain, y_unsq).squeeze(-1)  # (B,K,4)
        x_post = x_prior + K_y  # (B,K,4)

        # P_post = (I - K H) P_pred
        # KH = K_gain @ H -> (B,K,4,4)
        KH = torch.matmul(K_gain, self.H)  # (B,K,4,4)
        I4 = torch.eye(self.state_dim, device=KH.device, dtype=KH.dtype).view(1, 1, self.state_dim, self.state_dim)
        I_minus_KH = I4 - KH
        P_post = torch.matmul(I_minus_KH, P_pred)

        coords_post = x_post[..., :2].reshape(B, K * 2)

        out.update({
            'x_post': x_post,
            'coords_post': coords_post,
            'P_post': P_post,
            'K_gain': K_gain,
            'R_mat': R_mat
        })

        if return_dict:
            return out
        else:
            return coords_post


if __name__ == '__main__':
    
    B = 2
    T = 14
    K = 28
    input_size = 56
    model = LSTMKFNet(input_size=input_size, hidden_size=128, num_layers=2, bidirectional=True, num_points=K)
    seq = torch.randn(B, T, input_size)
    # 伪造的观测（归一化），形状为 (B, K*2)
    meas = torch.randn(B, K * 2)
    out = model(seq, meas)
    print('x_prior', out['x_prior'].shape)   # (B,K,4)
    print('coords_prior', out['coords_prior'].shape)  # (B, K*2)
    print('coords_post', out['coords_post'].shape)    # (B, K*2)
    print('P_post', out['P_post'].shape)  # (B,K,4,4)
    print('Q_diag', out['Q_diag'].shape)  # (B,K,4)
    print('R_diag', out['R_diag'].shape)  # (B,K,2)

