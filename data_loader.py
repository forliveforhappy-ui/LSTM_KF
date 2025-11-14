# data_loader.py  -- 支持 LSTM-KF 的数据格式（增加 target_state: [x,y,vx,vy]）
import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset

# 需要预测的非手指关键点索引（去掉左右手指 11-30, 34-53）
FILTER_IDX = [i for i in range(68) if not (11 <= i <= 30 or 34 <= i <= 53)]
NUM_POINTS = len(FILTER_IDX)  # 28 by design

def bbox_to_xyxy(bbox):
    """
    接受 bbox 格式为 [x1,y1,x2,y2] 或 [x,y,w,h]，返回 [x1,y1,x2,y2]。
    与原始 loader 保持兼容的逻辑。
    """
    if len(bbox) != 4:
        raise ValueError("bbox must be length 4")
    x0, y0, a, b = bbox
    # heuristic: if a and b look like x2/y2
    if (a > x0) and (b > y0) and (a - x0) < 2000 and (b - y0) < 2000:
        return [float(x0), float(y0), float(a), float(b)]
    x1 = float(x0)
    y1 = float(y0)
    x2 = float(x0) + float(a)
    y2 = float(y0) + float(b)
    return [x1, y1, x2, y2]

class KeypointDatasetForLSTMKF(Dataset):
    """
    针对 LSTM-KF 风格训练设计的数据集。
    对于每个中心帧 t ：
      - 输入：前 pre_len 帧（t-pre_len .. t-1）和后 post_len 帧（t+1 .. t+post_len）
      - 目标：第 t 帧

    每个样本返回：
      - input_seq: 张量 (input_len, NUM_POINTS*2)，每帧归一化（相对于各自 bbox）
      - target_coord: 张量 (NUM_POINTS*2,)，中心帧归一化关键点坐标
      - target_offsets: 张量 (NUM_POINTS*2,)，热力图上的小数偏移（与之前一致）
      - target_bbox: 张量 (4,)，中心帧的 bbox（xyxy 格式）
      - target_state: 张量 (NUM_POINTS*4,)，每个关键点的状态 [x,y,vx,vy]（归一化坐标和归一化速度）
    说明：
      - 速度 vx,vy 按归一化单位计算为 (x_t - x_{t-1}), (y_t - y_{t-1})，dt=1 帧。
      - 如果 t-1 无效（当前采样逻辑下不应发生），vx,vy 设为 0。
    """
    def __init__(self, json_dir, pre_len=9, post_len=5, heatmap_size=64):
        super().__init__()
        self.pre_len = int(pre_len)
        self.post_len = int(post_len)
        assert self.pre_len >= 1 and self.post_len >= 1
        self.input_len = self.pre_len + self.post_len
        self.heatmap_size = int(heatmap_size)
        self.samples = []

        files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
        for fname in files:
            path = os.path.join(json_dir, fname)
            try:
                with open(path, 'r') as f:
                    frames = json.load(f)
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}")
                continue

            if not isinstance(frames, list):
                print(f"Warning: {path} does not contain a list of frames, skipping.")
                continue

            n_frames = len(frames)
            if n_frames < (self.pre_len + 1 + self.post_len):
                continue

            for c in range(self.pre_len, n_frames - self.post_len):
                valid = True
                inputs_coords = []
                inputs_bboxes = []

                # collect previous frames (t-pre_len .. t-1)
                for idx in range(c - self.pre_len, c):
                    fr = frames[idx]
                    if not isinstance(fr, dict):
                        valid = False
                        break
                    bbox = fr.get('bbox', None)
                    kps = fr.get('output_keypoints_2d', None)
                    if bbox is None or kps is None:
                        valid = False
                        break
                    arr = np.array(kps, dtype=float)
                    if arr.size == 68 * 2:
                        pts68 = arr.reshape(68, 2)
                    elif arr.size == 68 * 3:
                        pts68 = arr.reshape(68, 3)[:, :2]
                    else:
                        valid = False
                        break
                    pts_filtered = pts68[FILTER_IDX]
                    inputs_coords.append(pts_filtered)
                    inputs_bboxes.append(bbox)
                if not valid:
                    continue

                # collect next frames (t+1 .. t+post_len)
                for idx in range(c + 1, c + 1 + self.post_len):
                    fr = frames[idx]
                    if not isinstance(fr, dict):
                        valid = False
                        break
                    bbox = fr.get('bbox', None)
                    kps = fr.get('output_keypoints_2d', None)
                    if bbox is None or kps is None:
                        valid = False
                        break
                    arr = np.array(kps, dtype=float)
                    if arr.size == 68 * 2:
                        pts68 = arr.reshape(68, 2)
                    elif arr.size == 68 * 3:
                        pts68 = arr.reshape(68, 3)[:, :2]
                    else:
                        valid = False
                        break
                    pts_filtered = pts68[FILTER_IDX]
                    inputs_coords.append(pts_filtered)
                    inputs_bboxes.append(bbox)
                if not valid:
                    continue

                # target frame (center)
                target_frame = frames[c]
                if not isinstance(target_frame, dict):
                    continue
                target_bbox_raw = target_frame.get('bbox', None)
                target_kps_raw = target_frame.get('output_keypoints_2d', None)
                if target_bbox_raw is None or target_kps_raw is None:
                    continue
                arr_t = np.array(target_kps_raw, dtype=float)
                if arr_t.size == 68 * 2:
                    pts68_t = arr_t.reshape(68, 2)
                elif arr_t.size == 68 * 3:
                    pts68_t = arr_t.reshape(68, 3)[:, :2]
                else:
                    continue
                target_pts_filtered = pts68_t[FILTER_IDX]  # (NUM_POINTS,2)

                # normalization: per-frame relative to its bbox
                def normalize_points(pts, bbox_raw):
                    x1, y1, x2, y2 = bbox_to_xyxy(bbox_raw)
                    w = x2 - x1 if (x2 - x1) != 0 else 1e-6
                    h = y2 - y1 if (y2 - y1) != 0 else 1e-6
                    norm = (pts - np.array([x1, y1])) / np.array([w, h])
                    return np.clip(norm, 0.0, 1.0)

                # normalize inputs (order: prev frames then next frames)
                norm_inputs = []
                for pts, bb in zip(inputs_coords, inputs_bboxes):
                    norm_inputs.append(normalize_points(pts, bb))
                norm_inputs = np.stack(norm_inputs, axis=0)  # (input_len, NUM_POINTS, 2)

                # normalize target coords
                norm_target = normalize_points(target_pts_filtered, target_bbox_raw)  # (NUM_POINTS,2)

                # compute fractional offsets for target on heatmap grid
                H = self.heatmap_size
                offsets = np.zeros((NUM_POINTS * 2,), dtype=np.float32)
                for p in range(NUM_POINTS):
                    x_norm = float(norm_target[p, 0])
                    y_norm = float(norm_target[p, 1])
                    x_cont = x_norm * (H - 1)
                    y_cont = y_norm * (H - 1)
                    x_int = int(math.floor(x_cont))
                    y_int = int(math.floor(y_cont))
                    if x_int >= H - 1:
                        x_int = H - 2
                    if y_int >= H - 1:
                        y_int = H - 2
                    dx = x_cont - x_int
                    dy = y_cont - y_int
                    offsets[2 * p] = float(np.clip(dx, 0.0, 1.0))
                    offsets[2 * p + 1] = float(np.clip(dy, 0.0, 1.0))

                # --- New part: compute target_state [x,y,vx,vy] in normalized space ---
                # Use last previous frame (t-1) to compute velocity: vx = x_t - x_{t-1}
                # last previous frame is inputs_coords[self.pre_len - 1]
                prev_pts = inputs_coords[self.pre_len - 1]  # (NUM_POINTS,2) absolute coords
                # normalize previous
                prev_bbox = inputs_bboxes[self.pre_len - 1]
                prev_norm = normalize_points(prev_pts, prev_bbox)  # (NUM_POINTS,2)
                # velocity in normalized units (dt=1)
                vel = norm_target - prev_norm  # (NUM_POINTS,2)
                # stack to state (NUM_POINTS,4)
                state_pts = np.concatenate([norm_target, vel], axis=1)  # (NUM_POINTS,4)

                # store sample (note: keep previous fields for backward compatibility, plus target_state)
                sample = {
                    'input_seq': norm_inputs.astype(np.float32),      # (input_len, NUM_POINTS,2)
                    'target_coord': norm_target.astype(np.float32),   # (NUM_POINTS,2)
                    'target_offsets': offsets.astype(np.float32),     # (NUM_POINTS*2,)
                    'target_bbox': np.array(bbox_to_xyxy(target_bbox_raw), dtype=np.float32),
                    'target_state': state_pts.astype(np.float32)      # (NUM_POINTS,4)
                }
                self.samples.append(sample)

        if len(self.samples) == 0:
            print("Warning: no valid samples found in", json_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # flatten input per frame: (input_len, NUM_POINTS*2)
        input_seq = s['input_seq'].reshape(self.input_len, -1)   # (input_len, NUM_POINTS*2)
        target = s['target_coord'].reshape(-1)                   # (NUM_POINTS*2,)
        target_offsets = s['target_offsets']                     # (NUM_POINTS*2,)
        target_bbox = s['target_bbox']                           # (4,)
        target_state = s['target_state'].reshape(-1)             # (NUM_POINTS*4,)
        # Return order: keep previous 4 items, add target_state as 5th
        return (torch.from_numpy(input_seq), torch.from_numpy(target),
                torch.from_numpy(target_offsets), torch.from_numpy(target_bbox),
                torch.from_numpy(target_state))
