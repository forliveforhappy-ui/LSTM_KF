import json
import numpy as np
import matplotlib.pyplot as plt

# 非手部关键点索引
NON_HAND_INDICES = [i for i in range(68) if not (11 <= i <= 30 or 34 <= i <= 53)]

# 连线规则
pose_line_map_body_with_hands = {
    'body': [
        [0, 1], [0, 2], [1, 3], [2, 4], [0, 5],
        [6, 7], [7, 8], [8, 9], [9, 10],
        [7, 31], [31, 32], [32, 33],
        [7, 54], [54, 66],
        [66, 67], [67, 55], [55, 56], [56, 57], [57, 58],
        [58, 59], [59, 60], [55, 61], [61, 62], [62, 63],
        [63, 64], [64, 65]
    ]
}

def load_keypoints_from_json(json_path):
    """加载 68 个关键点 (x, y)"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 取列表的第一个元素
    if not data or "output_keypoints_2d" not in data[0]:
        raise ValueError(f"JSON 文件中未找到 'output_keypoints_2d': {json_path}")

    keypoints = np.array(data[0]["output_keypoints_2d"], dtype=np.float32)
    if keypoints.shape != (68, 2):
        raise ValueError(f"关键点格式错误，期望(68,2)，实际 {keypoints.shape}")

    return keypoints
def visualize_two_json_side_by_side(json_file_1, json_file_2, save_path=None):
    """左右两侧各自独立坐标轴显示非手部关键点并连线"""
    kps1 = load_keypoints_from_json(json_file_1)
    kps2 = load_keypoints_from_json(json_file_2)

    kps1_nonhand = kps1[NON_HAND_INDICES]
    kps2_nonhand = kps2[NON_HAND_INDICES]

    # 创建左右两侧子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    def plot_keypoints_with_lines(ax, keypoints, color='green'):
        # 画关键点
        ax.scatter(keypoints[:, 0], keypoints[:, 1], c=color)
        for idx, (x, y) in enumerate(keypoints):
            ax.text(x + 2, y + 2, str(NON_HAND_INDICES[idx]), fontsize=8, color=color)

        # 画连线
        for a, b in pose_line_map_body_with_hands['body']:
            if a in NON_HAND_INDICES and b in NON_HAND_INDICES:
                idx_a = NON_HAND_INDICES.index(a)
                idx_b = NON_HAND_INDICES.index(b)
                x_vals = [keypoints[idx_a, 0], keypoints[idx_b, 0]]
                y_vals = [keypoints[idx_a, 1], keypoints[idx_b, 1]]
                ax.plot(x_vals, y_vals, c=color, linewidth=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Y轴向下为正

    # 左边
    plot_keypoints_with_lines(ax1, kps1_nonhand, color='green')
    ax1.set_title("JSON 1 (Ground Truth)")

    # 右边
    plot_keypoints_with_lines(ax2, kps2_nonhand, color='blue')
    ax2.set_title("JSON 2 (Prediction)")

    plt.suptitle("Comparison of Non-Hand Keypoints with Pose Lines")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    json_1 = "GT.json"
    json_2 = "pred_data/predicted_frame10_7.json"
    visualize_two_json_side_by_side(json_1, json_2, save_path="show_pic/compare_non_hand_lines7.png")
