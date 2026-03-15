"""
Hand/Body Pose Tracking Module - Using YOLOv8-pose
Right Arm Only Detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
from utils import calculate_joint_angles, map_arm_to_qarm


class PoseTracker:
    """
    Pose tracker using YOLOv8-pose for body keypoint detection
    Right arm only mode
    """

    # YOLOv8-pose keypoint mapping (RIGHT ARM ONLY)
    # YOLO: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
    #       5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
    #       9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip, ...
    #
    # Right arm keypoints (ONLY THESE ARE USED):
    # 6: right_shoulder
    # 8: right_elbow
    # 10: right_wrist
    RIGHT_ARM_KEYPOINTS = {
        'shoulder': 6,   # Right shoulder
        'elbow': 8,      # Right elbow
        'wrist': 10      # Right wrist
    }

    KEYPOINT_MAP = {
        5: 11,  # Left shoulder -> MediaPipe 11 (not used in right-arm mode)
        6: 12,  # Right shoulder -> MediaPipe 12
        7: 13,  # Left elbow -> MediaPipe 13 (not used in right-arm mode)
        8: 14,  # Right elbow -> MediaPipe 14
        9: 15,  # Left wrist -> MediaPipe 15 (not used in right-arm mode)
        10: 16, # Right wrist -> MediaPipe 16
        11: 23, # Left hip -> MediaPipe 23
        12: 24, # Right hip -> MediaPipe 24
    }

    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5):
        """
        Initialize pose tracker

        Args:
            static_image_mode: Whether to use static image mode
            model_complexity: Model complexity (not applicable to YOLO)
            min_detection_confidence: Minimum detection confidence
        """
        print("Loading YOLOv8-pose model...")
        self.model = YOLO('yolov8n-pose.pt')  # Auto-download model
        self.conf_threshold = min_detection_confidence
        print("YOLOv8-pose model loaded successfully")
        print("Mode: RIGHT ARM ONLY")

    def process(self, frame):
        """
        Process a frame for pose detection (right arm only)

        Args:
            frame: OpenCV image (BGR format)

        Returns:
            dict: {
                'landmarks': landmarks object,
                'image': image with drawn skeleton,
                'angles': joint angles dictionary,
                'qarm_joints': Qarm joint angles
            }
        """
        image_drawn = frame.copy()
        h, w = frame.shape[:2]

        # Run YOLOv8-pose inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        if results and len(results) > 0 and results[0].keypoints is not None:
            # Check if keypoints detected
            if results[0].keypoints.xy is None or len(results[0].keypoints.xy) == 0:
                return self._get_empty_result(image_drawn)

            # Select closest person (by bounding box area)
            best_person_idx = self._select_closest_person(results[0])

            if best_person_idx is None:
                return self._get_empty_result(image_drawn)

            keypoints = results[0].keypoints.xy[best_person_idx].cpu().numpy()  # (17, 2)
            conf = results[0].keypoints.conf[best_person_idx].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

            # Convert to MediaPipe-compatible format
            class Landmark:
                def __init__(self, x, y, z=0, visibility=0.5):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility

            # Create 33 keypoints list (MediaPipe format)
            landmarks = [Landmark(0, 0, 0, 0) for _ in range(33)]

            # Map YOLO keypoints to MediaPipe format
            for yolo_idx, mp_idx in self.KEYPOINT_MAP.items():
                if yolo_idx < len(keypoints):
                    x, y = keypoints[yolo_idx]
                    c = conf[yolo_idx] if yolo_idx < len(conf) else 0.5
                    landmarks[mp_idx] = Landmark(x / w, y / h, 0, c)

            # Draw skeleton (RIGHT ARM ONLY)
            image_drawn = self._draw_right_arm_skeleton(image_drawn, keypoints, conf)

            # Calculate joint angles
            angles = calculate_joint_angles(landmarks)

            # Get right arm keypoint positions (for Joint 1 and Joint 4 calculation)
            arm_position = {
                'shoulder': (keypoints[6][0], keypoints[6][1]),  # Right shoulder
                'elbow': (keypoints[8][0], keypoints[8][1]),      # Right elbow
                'wrist': (keypoints[10][0], keypoints[10][1])     # Right wrist
            }

            # Map to Qarm (RIGHT ARM)
            qarm_joints = map_arm_to_qarm(angles, side='right', arm_position=arm_position)

            return {
                'landmarks': landmarks,
                'image': image_drawn,
                'angles': angles,
                'qarm_joints': qarm_joints,
                'detected': True
            }

        return {
            'landmarks': None,
            'image': image_drawn,
            'angles': None,
            'qarm_joints': [0, -90, 0, 0],  # Default: natural hanging pose
            'detected': False
        }

    def _select_closest_person(self, result):
        """
        Select the person closest to the camera (by bounding box area)

        Args:
            result: YOLOv8 detection result

        Returns:
            int: Index of closest person, None if none found
        """
        if result.boxes is None or len(result.boxes) == 0:
            return None

        max_area = 0
        best_idx = 0

        for i, box in enumerate(result.boxes):
            # box.xyxy is [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)

            # Check keypoint confidence (avoid partial occlusion)
            # Focus on RIGHT ARM keypoints
            if result.keypoints.conf is not None and i < len(result.keypoints.conf):
                conf = result.keypoints.conf[i].cpu().numpy()
                # Check right arm keypoints (right shoulder, elbow, wrist)
                key_indices = [6, 8, 10]  # Right shoulder, elbow, wrist
                key_conf = conf[key_indices]
                avg_conf = np.mean(key_conf[key_conf > 0])

                # Skip if confidence too low
                if avg_conf < 0.3:
                    continue

            if area > max_area:
                max_area = area
                best_idx = i

        return best_idx

    def _get_empty_result(self, image_drawn):
        """Return empty detection result"""
        return {
            'landmarks': None,
            'image': image_drawn,
            'angles': None,
            'qarm_joints': [0, -90, 0, 0],  # Default: natural hanging pose
            'detected': False
        }

    def _draw_coordinate_system(self, img, origin, scale=20, label=""):
        """
        在图像上画3D坐标系

        Args:
            img: OpenCV图像
            origin: 原点坐标 (x, y)
            scale: 轴长度（像素）
            label: 坐标系标签

        坐标系定义（图像坐标系）:
            X轴（红色）- 水平向右
            Y轴（绿色）- 垂直向下
            Z轴（蓝色）- 垂直画面向外（用圆点表示）
        """
        x, y = int(origin[0]), int(origin[1])

        # X轴 - 红色 - 水平向右
        cv2.arrowedLine(img, (x, y), (x + scale, y), (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(img, "X", (x + scale + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Y轴 - 绿色 - 垂直向下
        cv2.arrowedLine(img, (x, y), (x, y + scale), (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(img, "Y", (x - 5, y + scale + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Z轴 - 蓝色 - 垂直画面向外（用实心圆表示）
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        cv2.putText(img, "Z", (x - 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # 可选：画一个小圆圈表示坐标系原点
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)

    def _draw_right_arm_skeleton(self, img, keypoints, conf):
        """
        Draw ONLY right arm skeleton on image

        Args:
            img: OpenCV image
            keypoints: (17, 2) keypoint coordinates
            conf: (17,) confidence values

        Returns:
            Drawn image
        """
        h, w = img.shape[:2]

        # RIGHT ARM connections only
        right_arm_connections = [
            (6, 8),   # Right shoulder - Right elbow
            (8, 10),  # Right elbow - Right wrist
        ]

        # Arm color (blue for right arm)
        arm_color = (255, 0, 0)  # Blue
        joint_color = (255, 100, 0)  # Orange

        # Draw connections
        for start_idx, end_idx in right_arm_connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))

                # Check confidence
                if conf[start_idx] > 0.3 and conf[end_idx] > 0.3:
                    cv2.line(img, start, end, arm_color, 4)

        # Draw RIGHT ARM keypoints only
        right_arm_points = [6, 8, 10]  # Right shoulder, elbow, wrist
        for idx in right_arm_points:
            if idx < len(keypoints) and conf[idx] > 0.3:
                x = int(keypoints[idx][0])
                y = int(keypoints[idx][1])

                # Outer circle (white)
                cv2.circle(img, (x, y), 10, (255, 255, 255), 2)
                # Inner circle (orange)
                cv2.circle(img, (x, y), 8, joint_color, -1)

                # Draw coordinate system at each joint
                label = "S" if idx == 6 else "E" if idx == 8 else "W"
                self._draw_coordinate_system(img, (x, y), scale=25, label=label)

        # Add "RIGHT ARM" label
        label_pos = (int(keypoints[6][0]) - 40, int(keypoints[6][1]) - 20) if 6 < len(keypoints) and conf[6] > 0.3 else (w - 150, 50)
        cv2.putText(img, "RIGHT ARM", label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, arm_color, 2)

        return img

    def close(self):
        """Close pose tracker"""
        pass


def get_camera_stream(camera_id=0, width=640, height=480):
    """
    Get camera video stream

    Args:
        camera_id: Camera ID (default 0)
        width: Image width
        height: Image height

    Returns:
        cv2.VideoCapture object
    """
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


if __name__ == "__main__":
    # Test pose tracking (RIGHT ARM ONLY)
    print("Starting camera test for pose detection...")
    print("Press 'q' to quit")
    print("Mode: RIGHT ARM ONLY")

    cap = get_camera_stream()
    tracker = PoseTracker()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = tracker.process(frame)

            # Display angle information
            if result['detected']:
                angles = result['angles']
                joints = result['qarm_joints']

                # Display angles on image (RIGHT ARM ONLY)
                info_text = [
                    f"RIGHT ARM ONLY MODE",
                    f"Right Shoulder: {angles['right_shoulder']:.1f} deg",
                    f"Right Elbow: {angles['right_elbow']:.1f} deg",
                    f"",
                    f"Qarm Joints: [{joints[0]:.1f}, {joints[1]:.1f}, {joints[2]:.1f}, {joints[3]:.1f}]"
                ]

                for i, text in enumerate(info_text):
                    cv2.putText(result['image'], text, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow('Pose Tracking Test - Right Arm Only (YOLOv8)', result['image'])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
