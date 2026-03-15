"""
工具函数模块 - 角度计算和坐标转换
"""

import numpy as np
import math
import cv2


def calculate_angle(a, b, c):
    """
    计算三点之间的有向角度 (以 b 为顶点)
    适用于图像坐标系 (x向右, y向下)

    预期结果（肩部角度 髋-肩-肘）:
    - ~180°: 手臂下垂（肘在肩下方）
    - ~90°: 手臂水平前伸
    - ~0°: 手臂上举（肘在肩上方）

    Args:
        a, b, c: 三个点的坐标 [(x, y), (x, y), (x, y)]

    Returns:
        角度（度），范围 [0, 360)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # 计算向量 ba (从b指向a) 和 bc (从b指向c)
    # 注意：在图像坐标系中，y向下为正
    ba = a - b
    bc = c - b

    # 计算角度（使用标准数学坐标系，y向上为正）
    # 所以需要反转y分量
    angle_ba = np.arctan2(-ba[1], ba[0])  # 反转y以匹配标准坐标系
    angle_bc = np.arctan2(-bc[1], bc[0])

    # 计算有向角度差（逆时针为正，符合标准数学）
    angle_diff = (angle_bc - angle_ba) * 180.0 / np.pi

    # 归一化到 [0, 360) 范围
    if angle_diff < 0:
        angle_diff += 360

    return angle_diff  # 范围: [0, 360)


def calculate_shoulder_angle_simple(shoulder, elbow):
    """
    计算上臂与垂直方向的夹角（简化版，不依赖髋部关键点）

    图像坐标系: x向右(+), y向下(+)
    垂直向下向量: (0, 1)

    预期结果:
    - ~0°: 手臂下垂（肘在肩下方，上臂与垂直方向一致）
    - ~90°: 手臂水平前伸
    - ~180°: 手臂上举（肘在肩上方，上臂与垂直方向相反）

    Args:
        shoulder: 肩膀坐标 (x, y)
        elbow: 肘部坐标 (x, y)

    Returns:
        角度（度），范围 [0, 180]，0°表示垂直向下，90°表示水平
    """
    shoulder = np.array(shoulder)
    elbow = np.array(elbow)

    # 上臂向量（从肩膀指向肘部）
    arm_vector = elbow - shoulder

    # 垂直向下向量 (0, 1) 在图像坐标系中
    vertical = np.array([0, 1])

    # 计算夹角（使用点积公式）
    # cos(θ) = (a·b) / (|a| * |b|)
    arm_length = np.linalg.norm(arm_vector)
    if arm_length < 1e-6:  # 避免除零
        return 0

    cos_angle = np.dot(arm_vector, vertical) / arm_length
    # 限制在 [-1, 1] 范围内避免数值误差
    cos_angle = np.clip(cos_angle, -1, 1)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    # 返回 [0, 180] 范围的角度
    # 0° = 垂直向下，90° = 水平，180° = 垂直向上
    return angle_deg


def calculate_joint_angles(landmarks):
    """
    从 landmarks 计算手臂关节角度（简化版，不依赖髋部关键点）

    Args:
        landmarks: landmarks 对象（兼容 MediaPipe 格式）

    Returns:
        dict: {
            'left_shoulder': 角度,
            'left_elbow': 角度,
            'right_shoulder': 角度,
            'right_elbow': 角度
        }
    """
    # 关键点索引
    # 左臂: 11(肩), 13(肘), 15(腕)
    # 右臂: 12(肩), 14(肘), 16(腕)

    def get_coord(idx):
        return [landmarks[idx].x, landmarks[idx].y]

    # 左臂角度
    left_shoulder_simple = calculate_shoulder_angle_simple(
        get_coord(11),  # 肩膀
        get_coord(13)   # 肘部
    )
    # 检测手臂方向：肘部在肩部左侧还是右侧
    left_shoulder_x = get_coord(11)[0]
    left_elbow_x = get_coord(13)[0]
    left_direction = -1 if left_elbow_x < left_shoulder_x else 1  # 负数表示向左

    left_elbow_angle = calculate_angle(
        get_coord(11),  # 肩膀
        get_coord(13),  # 肘部
        get_coord(15)   # 手腕
    )

    # 右臂角度
    right_shoulder_simple = calculate_shoulder_angle_simple(
        get_coord(12),  # 肩膀
        get_coord(14)   # 肘部
    )
    # 检测手臂方向：肘部在肩部左侧还是右侧
    right_shoulder_x = get_coord(12)[0]
    right_elbow_x = get_coord(14)[0]
    right_direction = -1 if right_elbow_x < right_shoulder_x else 1  # 负数表示向左

    right_elbow_angle = calculate_angle(
        get_coord(12),  # 肩膀
        get_coord(14),  # 肘部
        get_coord(16)   # 手腕
    )

    return {
        'left_shoulder': left_shoulder_simple,
        'left_elbow': left_elbow_angle,
        'right_shoulder': right_shoulder_simple,
        'right_elbow': right_elbow_angle,
        'left_direction': left_direction,
        'right_direction': right_direction
    }


def map_arm_to_qarm(arm_angles, side='left', arm_position=None):
    """
    将手臂关节角度映射到 Qarm 机械臂关节

    Qarm 有 4 个关节:
    - Joint 1: 底座旋转 (-170 ~ 170°)
    - Joint 2: 肩部 (-85 ~ 85°)
    - Joint 3: 肘部 (-95 ~ 75°)
    - Joint 4: 腕部 (-160 ~ 160°)

    坐标系说明：
    - 图像坐标系: x向右(+), y向下(+), 原点在左上角
    - 视觉上的"上"对应图像y的负方向
    - 视觉上的"下"对应图像y的正方向
    - 视觉上的"左"对应图像x的负方向
    - 视觉上的"右"对应图像x的正方向

    角度范围说明 (简化后的角度):
    - 肩部角度 (上臂与垂直方向): 0°=下垂, 90°=水平, 180°=上举
    - 肘部角度 (肩-肘-腕): ~180°=伸直, ~90°=前弯, ~270°=后弯

    人体手臂 → Qarm 映射:
    - 手臂水平方向 → Joint 1 (底座)
    - 肩部角度 → Joint 2 (0°=下垂 -> -90°, 90°=水平 -> 0°)
    - 肘部角度 → Joint 3 (使用 0-360° 范围检测小臂旋转方向)
    - 手腕相对前臂角度 → Joint 4

    Args:
        arm_angles: 字典，包含关节角度
        side: 'left' 或 'right'，选择哪只手臂
        arm_position: 字典，包含关键点坐标 {'shoulder': (x,y), 'elbow': (x,y), 'wrist': (x,y)}
                    如果提供，则计算 Joint 1 和 Joint 4

    Returns:
        list: [joint1, joint2, joint3, joint4] 角度（度）
    """
    if side == 'left':
        shoulder = arm_angles['left_shoulder']
        elbow = arm_angles['left_elbow']
        direction = arm_angles.get('left_direction', 1)
    else:
        shoulder = arm_angles['right_shoulder']
        elbow = arm_angles['right_elbow']
        direction = arm_angles.get('right_direction', 1)

    # Joint 2: 肩部 pitch 角度映射
    # 新的角度定义: 0°(下垂) → 90°(水平) → 180°(上举)
    # Qarm J2: -90°(下垂) → 0°(水平) → 85°(上举)
    # 线性映射: [0, 180] -> [-90, 85]
    joint2 = np.clip(shoulder * (175.0/180.0) - 90, -85, 85)

    # Joint 3: 肘部角度映射 - 支持小臂旋转检测
    # 图像角度 [0-360): 180°(伸直) → 90°(向前弯90°) → 270°(向后弯90°)
    # Qarm J3: 0°(伸直) → 90°(向前弯) → -90°(向后弯)
    if elbow <= 180:
        joint3 = np.clip(180 - elbow, 0, 90)  # 向前弯曲 (0-90°)
    else:
        joint3 = np.clip(180 - elbow, -95, 0)  # 向后弯曲 (0 到 -95°)

    # Joint 1: 底座旋转（如果提供了位置信息）
    if arm_position and 'shoulder' in arm_position and 'elbow' in arm_position:
        shoulder_pos = np.array(arm_position['shoulder'])
        elbow_pos = np.array(arm_position['elbow'])

        # 计算手臂向量（从肩膀指向肘部）
        arm_vector = elbow_pos - shoulder_pos
        dx = arm_vector[0]  # x差值
        dy = arm_vector[1]  # y差值

        # 计算水平角度: 以正前方为0°, 左转为负, 右转为正
        horizontal_angle = np.degrees(np.arctan2(dx, -dy))

        # 映射到 -170 到 170° 范围（以正前方为0°）
        joint1 = np.clip(horizontal_angle * 2, -170, 170)
    else:
        joint1 = 0  # 默认正前方

    # Joint 4: 腕部旋转 - 使用有向角度检测旋转方向
    if arm_position and 'elbow' in arm_position and 'wrist' in arm_position:
        elbow_pos = np.array(arm_position['elbow'])
        wrist_pos = np.array(arm_position['wrist'])

        # 前臂向量
        forearm_vector = wrist_pos - elbow_pos

        if 'shoulder' in arm_position:
            shoulder_pos = np.array(arm_position['shoulder'])
            upper_arm_vector = shoulder_pos - elbow_pos

            # 计算上臂到前臂的角度
            angle_upper = np.arctan2(upper_arm_vector[1], upper_arm_vector[0])
            angle_fore = np.arctan2(forearm_vector[1], forearm_vector[0])
            angle_diff = (angle_fore - angle_upper) * 180.0 / np.pi

            # 归一化到 -180 到 180
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360

            joint4 = np.clip(angle_diff, -160, 160)
        else:
            vertical_angle = np.degrees(np.arctan2(forearm_vector[0], -forearm_vector[1]))
            joint4 = np.clip(vertical_angle, -160, 160)
    else:
        joint4 = 0  # 默认位置（伸直）

    return [joint1, joint2, joint3, joint4]


def normalize_coords(coords, frame_shape):
    """
    将归一化坐标 (0-1) 转换为像素坐标

    Args:
        coords: (x, y) 归一化坐标
        frame_shape: (height, width)

    Returns:
        (x, y) 像素坐标
    """
    h, w = frame_shape[:2]
    return (int(coords[0] * w), int(coords[1] * h))


def draw_skeleton(img, landmarks):
    """
    在图像上绘制骨架

    Args:
        img: OpenCV 图像
        landmarks: MediaPipe Pose landmarks

    Returns:
        绘制后的图像
    """
    h, w = img.shape[:2]

    # 骨架连接定义
    connections = [
        # 左臂
        (11, 12),  # 肩膀连线
        (11, 13),  # 左肩-左肘
        (13, 15),  # 左肘-左腕
        # 右臂
        (12, 14),  # 右肩-右肘
        (14, 16),  # 右肘-右腕
        # 躯干
        (11, 23),  # 左肩-左髋
        (12, 24),  # 右肩-右髋
        (23, 24),  # 髋部连线
    ]

    # 关键点（用圆圈标记）
    key_points = [11, 12, 13, 14, 15, 16, 23, 24]

    for connection in connections:
        start_idx, end_idx = connection
        start = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))

        # 绘制连接线
        color = (0, 255, 0) if start_idx < 13 else (255, 0, 0)  # 左臂绿色，右臂蓝色
        cv2 = __import__('cv2')
        cv2.line(img, start, end, color, 3)

    # 绘制关键点
    for idx in key_points:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        color = (0, 255, 0) if idx in [11, 13, 15, 23] else (255, 0, 0)
        cv2 = __import__('cv2')
        cv2.circle(img, (x, y), 6, color, -1)
        cv2.circle(img, (x, y), 8, (255, 255, 255), 2)

    return img


if __name__ == "__main__":
    # 测试角度计算
    print("测试角度计算函数...")

    # 直角测试 (90度)
    angle = calculate_angle((0, 0), (1, 0), (1, 1))
    print(f"直角测试: {angle}° (期望 90°)")

    # 平角测试 (180度)
    angle = calculate_angle((0, 0), (1, 0), (2, 0))
    print(f"平角测试: {angle}° (期望 180°)")


def draw_skeleton_yolo(img, keypoints, conf):
    """
    在图像上绘制 YOLOv8-pose 骨架

    Args:
        img: OpenCV 图像
        keypoints: (17, 2) 关键点坐标
        conf: (17,) 置信度

    Returns:
        绘制后的图像
    """
    # YOLOv8-pose 骨架连接
    connections = [
        (5, 7),   # 左肩-左肘
        (7, 9),   # 左肘-左腕
        (6, 8),   # 右肩-右肘
        (8, 10),  # 右肘-右腕
        (5, 6),   # 肩膀连线
        (5, 11),  # 左肩-左髋
        (6, 12),  # 右肩-右髋
        (11, 12), # 髋部连线
    ]

    # 绘制连接线
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))

            # 检查置信度
            if conf[start_idx] > 0.3 and conf[end_idx] > 0.3:
                color = (0, 255, 0) if start_idx in [5, 7, 9, 11] else (255, 0, 0)
                cv2.line(img, start, end, color, 3)

    # 绘制关键点
    key_points = [5, 6, 7, 8, 9, 10, 11, 12]
    for idx in key_points:
        if idx < len(keypoints) and conf[idx] > 0.3:
            x = int(keypoints[idx][0])
            y = int(keypoints[idx][1])

            color = (0, 255, 0) if idx in [5, 7, 9, 11] else (255, 0, 0)
            cv2.circle(img, (x, y), 6, color, -1)
            cv2.circle(img, (x, y), 8, (255, 255, 255), 2)

    return img
