import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data_loader import KeypointDatasetForLSTMKF
from model import LSTMKFNet
from tqdm import tqdm
import datetime

# -----------------------
# 超参数设置
# -----------------------
DATA_DIR = 'dataset'
BATCH_SIZE = 256
NUM_EPOCHS = 200
LR = 2e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRINT_EVERY = 20

# Loss 权重
W_COORD = 1.0      # 后验坐标损失权重（主要目标）
W_VEL = 0.1        # 后验速度损失
W_Q_REG = 1e-3     # 对 predicted log-Q 的 L2 正则
W_R_REG = 1e-3     # 对 predicted log-R 的 L2 正则

# 模型超参数
INPUT_SIZE = 2 * 28   # 28个关键点，每个关键点2D坐标
OUTPUT_SIZE = 2 * 28
HIDDEN_SIZE = 256
NUM_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.2
MLP_HIDDEN = 512

# -----------------------
# 辅助函数
# -----------------------

def denormalize_coords(batch_coords, bboxes):
    """
    batch_coords: (B, K*2) normalized coords in [0,1]
    bboxes: (B,4) xyxy
    returns (B, K, 2) pixel coords
    """
    B = batch_coords.shape[0]
    K2 = batch_coords.shape[1]
    K = K2 // 2
    coords = batch_coords.reshape(B, K, 2)
    out = np.zeros_like(coords)
    for i in range(B):
        x1, y1, x2, y2 = bboxes[i]
        w = x2 - x1 if (x2 - x1) != 0 else 1e-6
        h = y2 - y1 if (y2 - y1) != 0 else 1e-6
        out[i] = coords[i] * np.array([w, h]) + np.array([x1, y1])
    return out

# -----------------------
# 训练/验证流程
# -----------------------

def main():
    # Dataset & DataLoader
    dataset = KeypointDatasetForLSTMKF(json_dir=DATA_DIR, pre_len=9, post_len=5, heatmap_size=64)
    N = len(dataset)
    if N == 0:
        raise RuntimeError(f"No samples found in {DATA_DIR}")

    idxs = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idxs)
    split = int(0.8 * N)
    train_idx, val_idx = idxs[:split], idxs[split:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Model
    model = LSTMKFNet(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                      bidirectional=BIDIRECTIONAL, dropout=DROPOUT, mlp_hidden=MLP_HIDDEN,
                      use_boundary_skip=True, num_points=28)
    model.to(DEVICE)

    # Losses and optimizer
    criterion_coord = nn.SmoothL1Loss()
    criterion_vel = nn.SmoothL1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # logging & saving
    run_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('LSTM_KF_saved_model', run_time)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')
    log_path = os.path.join(save_dir, 'log.txt')

    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,pixel_error,lr,time')

    best_val = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{NUM_EPOCHS}")

        for i, batch in pbar:
            # batch unroll: (input_seq, target_coord, target_offsets, target_bbox, target_state)
            seq, target_coord, target_offsets, target_bbox, target_state = batch
            seq = seq.to(DEVICE).float()
            target_coord = target_coord.to(DEVICE).float()
            target_state = target_state.to(DEVICE).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                out = model(seq, meas=target_coord)
                coords_post = out['coords_post']  # (B, K*2)
                x_post = out['x_post']            # (B, K, 4)
                Q_diag = out['Q_diag']            # (B, K, 4)
                R_diag = out['R_diag']            # (B, K, 2)

                # losses
                loss_c = criterion_coord(coords_post, target_coord)
                # velocity supervision: compare state vx,vy
                target_state_resh = target_state.view(target_state.size(0), -1, 4)  # (B,K,4)
                vel_tgt = target_state_resh[..., 2:4].reshape(target_state.size(0), -1)  # (B, K*2)
                vel_pred = x_post[..., 2:4].reshape(target_state.size(0), -1)
                loss_v = criterion_vel(vel_pred, vel_tgt)

                # regularize log-variances (we predicted Q_diag and R_diag as positive values already)
                # penalize extreme values by L2 on log (approx via log of Q_diag)
                reg_q = torch.mean(torch.log(Q_diag + 1e-6) ** 2)
                reg_r = torch.mean(torch.log(R_diag + 1e-6) ** 2)

                loss = W_COORD * loss_c + W_VEL * loss_v + W_Q_REG * reg_q + W_R_REG * reg_r

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if (i + 1) % PRINT_EVERY == 0:
                pbar.set_postfix({'loss': f"{epoch_loss / (i + 1):.4f}"})

        # validation
        model.eval()
        val_loss = 0.0
        total_pixel_error = 0.0
        total_frames = 0
        with torch.no_grad():
            for batch in val_loader:
                seq, target_coord, target_offsets, target_bbox, target_state = batch
                seq = seq.to(DEVICE).float()
                target_coord = target_coord.to(DEVICE).float()
                target_bbox = target_bbox.numpy()
                target_state = target_state.to(DEVICE).float()

                out = model(seq, meas=target_coord)
                coords_post = out['coords_post']  # (B, K*2)
                coords_post_np = coords_post.cpu().numpy()
                target_coord_np = target_coord.cpu().numpy()

                # compute pixel error per sample using target_bbox
                coords_px = denormalize_coords(coords_post_np, target_bbox)
                tgts_px = denormalize_coords(target_coord_np, target_bbox)
                # per-sample mean error
                batch_err = np.linalg.norm(coords_px - tgts_px, axis=2).mean(axis=1)  # (B,)
                total_pixel_error += batch_err.sum()
                total_frames += coords_px.shape[0]

                # loss for scheduler
                loss_c = nn.functional.smooth_l1_loss(coords_post, target_coord)
                # velocity loss
                target_state_resh = target_state.view(target_state.size(0), -1, 4)
                vel_tgt = target_state_resh[..., 2:4].reshape(target_state.size(0), -1)
                vel_pred = out['x_post'][..., 2:4].reshape(target_state.size(0), -1)
                loss_v = nn.functional.smooth_l1_loss(vel_pred.to(DEVICE), vel_tgt)

                reg_q = torch.mean(torch.log(out['Q_diag'] + 1e-6) ** 2)
                reg_r = torch.mean(torch.log(out['R_diag'] + 1e-6) ** 2)

                batch_loss = W_COORD * loss_c + W_VEL * loss_v + W_Q_REG * reg_q + W_R_REG * reg_r
                val_loss += batch_loss.item()

        val_loss = val_loss / max(1, len(val_loader))
        avg_pixel_error = total_pixel_error / max(1, total_frames)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        train_loss_avg = epoch_loss / max(1, len(train_loader))

        log_line = f"{epoch},{train_loss_avg:.4f},{val_loss:.4f},{avg_pixel_error:.2f},{optimizer.param_groups[0]['lr']:.2e},{elapsed:.1f}s\n"

        print(f"[Epoch {epoch}] train_loss: {train_loss_avg:.4f} | val_loss: {val_loss:.4f} | pixel_error: {avg_pixel_error:.2f} | time: {elapsed:.1f}s | lr: {optimizer.param_groups[0]['lr']:.2e}")
        with open(log_path, 'a') as f:
            f.write(log_line)

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss': val_loss
            }, save_path)
            print(f"Saved best model (val_loss={val_loss:.4f}) -> {save_path}")

    print('Training finished. Best val loss:', best_val)

if __name__ == '__main__':
    main()

