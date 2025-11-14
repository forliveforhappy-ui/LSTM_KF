# inference.py (LSTM-KF 版本)
import os
import json
import torch
import numpy as np
import math
import re
from typing import List, Tuple
import time

# 使用 LSTMKFNet 作为 ModelClass
from model import LSTMKFNet as ModelClass

# 非手部关键点索引（与 data_loader 保持一致）
NON_HAND_INDICES = [i for i in range(68) if not (11 <= i <= 30 or 34 <= i <= 53)]
NUM_POINTS = len(NON_HAND_INDICES)  # 28


def bbox_to_xyxy(bbox):
    """
    Accept bbox in either [x1,y1,x2,y2] or [x,y,w,h] and return [x1,y1,x2,y2].
    Same heuristic as data_loader.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must be length 4")
    x0, y0, a, b = bbox
    if (a > x0) and (b > y0) and (a - x0) < 2000 and (b - y0) < 2000:
        return [float(x0), float(y0), float(a), float(b)]
    x1 = float(x0)
    y1 = float(y0)
    x2 = float(x0) + float(a)
    y2 = float(y0) + float(b)
    return [x1, y1, x2, y2]


def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _is_state_dict_like(d: dict):
    """判断一个 dict 是否像 state_dict（value 为 tensor 的集合）"""
    if not isinstance(d, dict) or len(d) == 0:
        return False
    for v in d.values():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            return True
    return False


def load_model_from_weights(weights_path: str, device=None, model_kwargs: dict = None):
    """
    加载模型权重并返回 model, device。和之前版本基本一致，但 ModelClass 为 LSTMKFNet。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    ckpt = torch.load(weights_path, map_location=device)

    sd = None
    ckpt_model_cfg = None
    if isinstance(ckpt, dict):
        if 'model_state' in ckpt and _is_state_dict_like(ckpt['model_state']):
            sd = ckpt['model_state']
        elif _is_state_dict_like(ckpt) and any(k.startswith('lstm.') or k.startswith('prior_head.') for k in ckpt.keys()):
            sd = ckpt
        elif all(isinstance(k, str) and k.startswith('module.') for k in ckpt.keys()) and _is_state_dict_like(ckpt):
            sd = {k[len('module.'):]: v for k, v in ckpt.items()}
        if 'model_cfg' in ckpt:
            ckpt_model_cfg = ckpt['model_cfg']

    if sd is None and _is_state_dict_like(ckpt):
        sd = ckpt

    if sd is None:
        raise RuntimeError(f"Cannot find a state_dict inside checkpoint: {weights_path}")

    sd = { (k[len('module.'): ] if k.startswith('module.') else k) : v for k,v in sd.items() }

    model_kwargs = dict(model_kwargs) if model_kwargs is not None else {}
    if ckpt_model_cfg and isinstance(ckpt_model_cfg, dict):
        for kk, vv in ckpt_model_cfg.items():
            model_kwargs.setdefault(kk, vv)

    ckpt_has_reverse = any(('_reverse' in k) for k in sd.keys())
    lstm_w_keys = [k for k in sd.keys() if k.startswith('lstm.weight_ih_l')]
    hidden_size_from_ckpt = None
    input_dim_from_ckpt = None
    num_layers_from_ckpt = None
    if len(lstm_w_keys) > 0:
        key0 = None
        for candidate in lstm_w_keys:
            if candidate.startswith('lstm.weight_ih_l0'):
                key0 = candidate
                break
        if key0 is None:
            key0 = lstm_w_keys[0]
        w0 = sd[key0]
        if isinstance(w0, torch.Tensor):
            r0, c0 = w0.shape
        elif isinstance(w0, np.ndarray):
            r0, c0 = w0.shape
        else:
            r0, c0 = None, None
        if r0 is not None:
            hidden_size_from_ckpt = r0 // 4
            input_dim_from_ckpt = c0
        max_layer = -1
        for k in sd.keys():
            m = re.match(r'lstm\.weight_ih_l(\d+)', k)
            if m:
                idx = int(m.group(1))
                if idx > max_layer:
                    max_layer = idx
        if max_layer >= 0:
            num_layers_from_ckpt = max_layer + 1

    if 'input_size' not in model_kwargs:
        if input_dim_from_ckpt is not None:
            model_kwargs['input_size'] = int(input_dim_from_ckpt)
        else:
            model_kwargs['input_size'] = 56
    if 'hidden_size' not in model_kwargs:
        model_kwargs['hidden_size'] = int(hidden_size_from_ckpt) if hidden_size_from_ckpt is not None else 256
    if 'num_layers' not in model_kwargs:
        model_kwargs['num_layers'] = int(num_layers_from_ckpt) if num_layers_from_ckpt is not None else 3
    if 'bidirectional' not in model_kwargs:
        model_kwargs['bidirectional'] = bool(ckpt_has_reverse)
    # ensure num_points present for LSTMKFNet
    model_kwargs.setdefault('num_points', NUM_POINTS)
    model_kwargs.setdefault('use_boundary_skip', model_kwargs.get('use_boundary_skip', True))

    model = ModelClass(**model_kwargs).to(device)

    try:
        model.load_state_dict(sd)
    except Exception as e:
        sd_fixed = {}
        for k, v in sd.items():
            newk = k
            if k.startswith('module.'):
                newk = k[len('module.'):]
            sd_fixed[newk] = v
        try:
            model.load_state_dict(sd_fixed)
        except Exception as e2:
            try:
                model.load_state_dict(sd_fixed, strict=False)
                print("Warning: loaded checkpoint in non-strict mode (missing/unexpected keys).")
            except Exception as e3:
                raise RuntimeError(
                    "Failed to load checkpoint into model. Please ensure model hyperparams match.\n"
                    f"Auto-inferred kwargs: {model_kwargs}\n"
                    f"Errors: {e}, {e2}, {e3}"
                )

    model.eval()
    return model, device


def _ensure_batch_and_flat(input_sequence: np.ndarray) -> torch.Tensor:
    """
    Accept inputs:
      - (T, 28, 2) -> (1, T, 56)
      - (B, T, 28, 2) -> (B, T, 56)
      - (T, 56) -> (1, T, 56)
      - (B, T, 56) -> (B, T, 56)
    """
    arr = np.asarray(input_sequence)
    if arr.ndim == 3 and arr.shape[1:] == (28, 2):  # (T,28,2)
        arr = arr.reshape(1, arr.shape[0], -1)
    elif arr.ndim == 4 and arr.shape[2:] == (28, 2):  # (B,T,28,2)
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    elif arr.ndim == 2 and arr.shape[1] == 56:  # (T,56)
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    elif arr.ndim == 3 and arr.shape[2] == 56:  # (B,T,56)
        pass
    else:
        raise ValueError(f"Unsupported input shape: {arr.shape}")
    return torch.from_numpy(arr).float()


def parse_keypoints_field(kps_field):
    arr = np.array(kps_field, dtype=float)
    if arr.ndim == 2 and arr.shape == (68, 2):
        return arr
    if arr.ndim == 1 and arr.size == 68 * 2:
        return arr.reshape(68, 2)
    raise ValueError(f"Unsupported keypoints format with shape {arr.shape}")


def find_missing_position(frames: List[dict], pre_len=9, post_len=5) -> Tuple[int, List[dict]]:
    idx_list = []
    have_idx = True
    for i, fr in enumerate(frames):
        if isinstance(fr, dict):
            if 'frame_index' in fr:
                idx_list.append((int(fr['frame_index']), i, fr))
            elif 'frame_id' in fr:
                idx_list.append((int(fr['frame_id']), i, fr))
            else:
                have_idx = False
                break
        else:
            have_idx = False
            break

    if have_idx:
        idx_list.sort(key=lambda x: x[0])
        sorted_frames = [t[2] for t in idx_list]
        sorted_indices = [t[0] for t in idx_list]
        insert_pos = None
        for i in range(len(sorted_indices) - 1):
            if sorted_indices[i + 1] > sorted_indices[i] + 1:
                missing_index = sorted_indices[i] + 1
                insert_pos = i + 1
                break
        if insert_pos is None:
            raise ValueError("Cannot find missing frame index gap in provided frames.")
        if insert_pos - pre_len < 0 or insert_pos + post_len > len(sorted_frames):
            raise ValueError("Not enough surrounding frames to form input (need pre_len before and post_len after).")
        return insert_pos, sorted_frames
    else:
        if len(frames) == (pre_len + post_len):
            insert_pos = pre_len
            return insert_pos, frames.copy()
        else:
            raise ValueError("Frames do not contain frame_index/frame_id and length != pre_len+post_len; cannot infer missing position.")


def build_input_sequence_from_window(pre_frames: List[dict], post_frames: List[dict]) -> np.ndarray:
    seq_frames = pre_frames + post_frames
    seq_norm = []
    for fr in seq_frames:
        if 'output_keypoints_2d' in fr:
            kps = parse_keypoints_field(fr['output_keypoints_2d'])
        elif 'keypoints' in fr:
            kps = parse_keypoints_field(fr['keypoints'])
        else:
            raise ValueError("Frame missing keypoints field (output_keypoints_2d or keypoints).")

        pts28 = kps[NON_HAND_INDICES, :2]  # (28,2)

        if 'bbox' not in fr:
            raise ValueError("Frame missing bbox field.")
        x1, y1, x2, y2 = bbox_to_xyxy(fr['bbox'])
        w = x2 - x1 if (x2 - x1) != 0 else 1e-6
        h = y2 - y1 if (y2 - y1) != 0 else 1e-6
        norm = (pts28 - np.array([x1, y1])) / np.array([w, h])
        norm = np.clip(norm, 0.0, 1.0)
        seq_norm.append(norm.astype(np.float32))
    return np.stack(seq_norm, axis=0)  # (T, 28, 2)


def predict_next_frame(model, input_sequence, device=None, use_posterior_if_meas=False):
    """
    输入 input_sequence: np array (T,28,2) or (1,T,56) 等
    返回 pred_coords (28,2) 归一化坐标

    对于 LSTM-KF：
      - 若没有观测（缺帧场景），使用 model(..., meas=None) 并返回 'coords_prior'（先验预测）
      - 若手头有观测 meas（比如来自一个检测器）并希望得到后验，请传 meas 并设置 use_posterior_if_meas=True
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = _ensure_batch_and_flat(input_sequence).to(device)  # (1, T, 56)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x, meas=None)  # returns dict
        # choose prior coords (since we don't have measurement for missing center)
        if 'coords_prior' in out:
            coords = out['coords_prior']  # (B, K*2)
        else:
            coords = out['coords_post']
        coords = coords.clamp(0.0, 1.0)
        coords_np = coords.view(-1, NUM_POINTS, 2).cpu().numpy()[0]  # (28,2)
    return coords_np


def save_single_frame_json(pred_coords_norm: np.ndarray, bbox_xyxy: List[float], template_frame: dict, out_path: str, missing_index: int = None):
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1

    abs_coords = pred_coords_norm.copy()
    abs_coords[:, 0] = abs_coords[:, 0] * w + x1
    abs_coords[:, 1] = abs_coords[:, 1] * h + y1

    full_kps = [[0.0, 0.0] for _ in range(68)]
    for idx, j in enumerate(NON_HAND_INDICES):
        full_kps[j] = [float(abs_coords[idx, 0]), float(abs_coords[idx, 1])]

    bbox_xywh = xyxy_to_xywh(bbox_xyxy)

    new_frame = {
        "frame_index": (int(missing_index) if missing_index is not None else template_frame.get("frame_index", None)),
        "image_id": template_frame.get("image_id", ""),
        "bbox": bbox_xywh,
        "score": template_frame.get("score", 1.0),
        "output_keypoints_2d": full_kps
    }

    with open(out_path, 'w') as f:
        json.dump([new_frame], f, indent=4)
    print(f"Saved predicted single frame to: {out_path}")


# ---------------------------
# Main: 读取 14 帧 JSON -> 推理 -> 存单帧 JSON
# ---------------------------
if __name__ == "__main__":
    # 配置：修改为你的路径
    json_path = "test_data/0_8_10_14.json"   # 输入：包含 14 帧（缺第10帧）的 JSON
    weights = "/home/liangguomin/.vscode-server/prediction/LSTM_KF_saved_model/20251015_101908/best_model.pth"   # 训练好的权重文件
    output = "pred_data/predicted_frame10_7.json"                  # 输出文件（包含 1 帧的 list）

    # 若训练时使用了特定的模型超参（例如 num_points/hidden_size/ bidirectional），可显式提供：
    model_kwargs = None
    model, device = load_model_from_weights(weights_path=weights, device=None, model_kwargs=model_kwargs)

    # load frames
    with open(json_path, 'r') as f:
        frames = json.load(f)
    if not isinstance(frames, list):
        raise ValueError("Input json must contain a list of frames.")

    PRE_LEN = 9
    POST_LEN = 5
    insert_pos, sorted_frames = find_missing_position(frames, pre_len=PRE_LEN, post_len=POST_LEN)
    pre_frames = sorted_frames[insert_pos - PRE_LEN: insert_pos]
    post_frames = sorted_frames[insert_pos: insert_pos + POST_LEN]

    input_seq = build_input_sequence_from_window(pre_frames, post_frames)  # (14,28,2)

    start_time = time.time()
    pred_norm = predict_next_frame(model, input_seq, device=device)  # (28,2)
    end_time = time.time()
    print(f"模型推理耗时: {(end_time - start_time)*1000:.6f} ms")

    prev_bbox_xyxy = bbox_to_xyxy(pre_frames[-1]['bbox'])
    next_bbox_xyxy = bbox_to_xyxy(post_frames[0]['bbox'])
    bbox_avg = [(prev_bbox_xyxy[i] + next_bbox_xyxy[i]) / 2.0 for i in range(4)]

    missing_idx = None
    try:
        if 'frame_index' in pre_frames[-1] or 'frame_id' in pre_frames[-1]:
            prev_idx = int(pre_frames[-1].get('frame_index', pre_frames[-1].get('frame_id')))
            missing_idx = prev_idx + 1
    except Exception:
        missing_idx = None

    save_single_frame_json(pred_norm, bbox_avg, pre_frames[-1], output, missing_index=missing_idx)

    print("Done.")
