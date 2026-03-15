"""
Hand Pose Control Qarm Demo - Right Arm Only (Optimized)
Uses matplotlib with incremental updates for smooth performance

Left: Camera hand pose capture (YOLOv8-pose, RIGHT ARM ONLY)
Right: Qarm 3D simulation following

Key Controls:
    '1' - Front view
    '2' - Side view
    '3' - Top view
    '4' - 3D (isometric) view
    'a' / 'd' - Rotate view left/right
    'q' - Quit
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from hand_tracker import PoseTracker, get_camera_stream
from qarm_sim import QarmSimulator


class HandControlQarmDemo:
    """Hand pose control demo - optimized with incremental updates"""

    def __init__(self, camera_id=0):
        """Initialize demo"""
        self.cap = get_camera_stream(camera_id, width=480, height=360)

        self.tracker = PoseTracker(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )

        self.sim = QarmSimulator(create_figure=False)

        # State
        self.running = True
        self.frame_count = 0
        self.camera_enabled = True  # Camera on/off switch

        # View state
        self.elev = 20
        self.azim = 45
        self.current_view = 'default'  # Track current view for toggle
        self.preset_views = {
            'front': (0, 0),
            'back': (0, 180),
            'left': (0, 90),
            'right': (0, -90),
            'top': (90, -90),
            'bottom': (-90, -90),
            'iso': (30, 45)
        }

        # Setup matplotlib figure
        plt.ion()
        self.fig = plt.figure(figsize=(12, 5))
        self.fig.canvas.manager.set_window_title('Hand Pose Control Qarm - Right Arm (Auto Mode)')
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.1)

        # Create subplots (only once!)
        self.ax_cam = self.fig.add_subplot(1, 2, 1)
        self.ax_3d = self.fig.add_subplot(1, 2, 2, projection='3d')

        # Initialize camera image display
        ret, frame = self.cap.read()
        if ret:
            img = cv2.cvtColor(cv2.resize(frame, (480, 360)), cv2.COLOR_BGR2RGB)
            self.cam_img_obj = self.ax_cam.imshow(img)
        else:
            # Create blank image
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            self.cam_img_obj = self.ax_cam.imshow(blank)

        self.ax_cam.set_title("Camera Feed - Right Arm Tracking", fontsize=11, fontweight='bold')
        self.ax_cam.axis('off')

        # Initialize text overlays (empty initially)
        self.text_status = self.ax_cam.text(0.02, 0.95, "", transform=self.ax_cam.transAxes,
                                            fontsize=11, fontweight='bold',
                                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        self.text_mode = self.ax_cam.text(0.02, 0.89, "", transform=self.ax_cam.transAxes,
                                         fontsize=11, fontweight='bold',
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        self.text_shoulder = self.ax_cam.text(0.02, 0.83, "", transform=self.ax_cam.transAxes,
                                             fontsize=9, color='blue',
                                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        self.text_elbow = self.ax_cam.text(0.02, 0.77, "", transform=self.ax_cam.transAxes,
                                          fontsize=9, color='blue',
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Help text - key controls (at top to avoid blocking camera view)
        help_text = "1:FRONT  2:LEFT  3:TOP  4:ISO  C:CAM  P:PHYS  Q:QUIT"
        self.text_help = self.ax_cam.text(0.5, 0.98, help_text, transform=self.ax_cam.transAxes,
                                          fontsize=10, ha='center', fontweight='bold',
                                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        # Initialize 3D plot objects (create once, update later)
        self._init_3d_objects()

        # Connect keyboard event
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        # Position window
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()

            window_width = int(screen_width * 0.8)
            window_height = int(screen_height * 0.65)
            window_x = int(screen_width * 0.1)
            window_y = int(screen_height * 0.15)

            backend = plt.get_backend()
            if 'TkAgg' in backend:
                self.fig.canvas.manager.window.wm_geometry(f"+{window_x}+{window_y}")
                self.fig.canvas.manager.window.wm_geometry(f"{window_width}x{window_height}")
        except:
            pass

        self._print_controls()

    def _init_3d_objects(self):
        """Initialize 3D plot objects (created once, updated in loop)"""
        # Set view angle
        self.ax_3d.view_init(elev=self.elev, azim=self.azim)

        # Set axis limits (once)
        max_reach = 0.35
        limit = 0.3
        self.ax_3d.set_xlim(-limit, limit)
        self.ax_3d.set_ylim(-limit * 0.3, limit)
        self.ax_3d.set_zlim(0, 0.6)
        self.ax_3d.set_xlabel('X (m)', fontsize=7)
        self.ax_3d.set_ylabel('Y (m)', fontsize=7)
        self.ax_3d.set_zlabel('Z (m)', fontsize=7)
        self.ax_3d.grid(True, alpha=0.3)

        # Store line objects for updating
        self.line_base, = self.ax_3d.plot([], [], [], 'k-', linewidth=6, alpha=0.7)
        self.line_upper, = self.ax_3d.plot([], [], [], 'b-', linewidth=5, marker='o', markersize=6)
        self.line_fore, = self.ax_3d.plot([], [], [], 'g-', linewidth=4, marker='o', markersize=5)
        self.line_finger1, = self.ax_3d.plot([], [], [], color='cyan', linewidth=2, marker='o', markersize=3)
        self.line_finger2, = self.ax_3d.plot([], [], [], color='magenta', linewidth=2, marker='o', markersize=3)

        # Joint markers (scatter - but create once)
        self.scatter_should = self.ax_3d.scatter([], [], [], color='red', s=100, marker='s', alpha=0.8)
        self.scatter_elbow = self.ax_3d.scatter([], [], [], color='purple', s=80, marker='o', alpha=0.8)
        self.scatter_wrist = self.ax_3d.scatter([], [], [], color='brown', s=60, marker='o', alpha=0.8)

        # Title
        self.title_3d = self.ax_3d.set_title('Qarm 3D Simulation', fontsize=11, fontweight='bold')

    def _print_controls(self):
        """Print control instructions"""
        print("=" * 50)
        print("  HAND POSE CONTROL QARM - AUTO MODE")
        print("=" * 50)
        print("\n  MODE: AUTO (Right Arm Tracking)")
        print("\n  KEYBOARD:")
        print("  ═════════")
        print("  VIEW CONTROLS:")
        print("    1    - Front view")
        print("    2    - Side view")
        print("    3    - Top view")
        print("    4    - 3D (isometric) view")
        print("    a/d  - Rotate view left/right")
        print("  OTHER:")
        print("    c    - Toggle camera ON/OFF")
        print("    p    - Toggle physics limits (ON: hardware / OFF: free ±180°)")
        print("    q    - Quit")
        print("  ═════════")
        print("  >>> CLICK THE PLOT WINDOW TO ENABLE KEYBOARD <<<")
        print("=" * 50 + "\n")

    def _on_key_press(self, event):
        """Handle keyboard input"""
        if event.key is None:
            return

        key = event.key.lower()

        if key == 'q':
            self.running = False

        # Camera toggle
        elif key == 'c':
            self.camera_enabled = not self.camera_enabled
            if self.camera_enabled:
                print("Camera: ON")
                self.text_mode.set_text("MODE: AUTO")
                self.text_mode.set_color('green')
            else:
                print("Camera: OFF - Resetting to initial pose")
                self.text_mode.set_text("MODE: OFF")
                self.text_mode.set_color('red')
                # Reset to initial pose (arms down)
                self.sim.set_joints([0, -90, 0, 0])

        # Physics limits toggle
        elif key == 'p':
            self.sim.physics_enabled = not self.sim.physics_enabled
            status = "ON (Hardware)" if self.sim.physics_enabled else "OFF (Free ±180°)"
            print(f"Physics Limits: {status}")

        # View controls (with toggle support)
        elif key == '1':
            # Toggle between front and back
            if self.current_view == 'front':
                target = 'back'
            else:
                target = 'front'
            self.elev, self.azim = self.preset_views[target]
            self.current_view = target
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
            self._update_help_text()
            print(f"View: {target.upper()}")
        elif key == '2':
            # Toggle between left and right
            if self.current_view == 'left':
                target = 'right'
            else:
                target = 'left'
            self.elev, self.azim = self.preset_views[target]
            self.current_view = target
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
            self._update_help_text()
            print(f"View: {target.upper()}")
        elif key == '3':
            # Toggle between top and bottom
            if self.current_view == 'top':
                target = 'bottom'
            else:
                target = 'top'
            self.elev, self.azim = self.preset_views[target]
            self.current_view = target
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
            self._update_help_text()
            print(f"View: {target.upper()}")
        elif key == '4':
            # ISO stays ISO
            self.elev, self.azim = self.preset_views['iso']
            self.current_view = 'iso'
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
            self._update_help_text()
            print("View: ISO")
        elif key == 'a':
            self.azim = (self.azim - 10) % 360
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
        elif key == 'd':
            self.azim = (self.azim + 10) % 360
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)

    def process_camera(self):
        """Capture and process camera frame"""
        if not self.camera_enabled:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Resize for performance
        frame = cv2.resize(frame, (480, 360))

        # Process pose detection
        result = self.tracker.process(frame)

        # Auto mode: follow right arm
        if result['detected']:
            from utils import map_arm_to_qarm
            joints = map_arm_to_qarm(result['angles'], side='right',
                                   arm_position=result.get('arm_position', {}))
            self.sim.set_joints(joints)

        return result

    def update_camera_view(self, result):
        """Update camera view (incremental, no clear)"""
        if result is None:
            return

        # Update image
        img = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
        self.cam_img_obj.set_data(img)

        # Update text overlays
        status = "Pose Detected" if result['detected'] else "No Pose"
        color = 'green' if result['detected'] else 'red'
        self.text_status.set_text(f"Camera: {status}")
        self.text_status.set_color(color)

        self.text_mode.set_text("MODE: AUTO")

        if result['detected'] and result['angles']:
            self.text_shoulder.set_text(f"Shoulder: {result['angles']['right_shoulder']:.0f}°")
            self.text_elbow.set_text(f"Elbow: {result['angles']['right_elbow']:.0f}°")
        else:
            self.text_shoulder.set_text("")
            self.text_elbow.set_text("")

    def update_3d_view(self):
        """Update 3D view (incremental, no clear)"""
        positions = self.sim.forward_kinematics(self.sim.joints)

        # Update lines
        self.line_base.set_data([positions['base'][0], positions['shoulder'][0]],
                               [positions['base'][1], positions['shoulder'][1]])
        self.line_base.set_3d_properties([positions['base'][2], positions['shoulder'][2]])

        self.line_upper.set_data([positions['shoulder'][0], positions['elbow'][0]],
                                [positions['shoulder'][1], positions['elbow'][1]])
        self.line_upper.set_3d_properties([positions['shoulder'][2], positions['elbow'][2]])

        self.line_fore.set_data([positions['elbow'][0], positions['wrist'][0]],
                               [positions['elbow'][1], positions['wrist'][1]])
        self.line_fore.set_3d_properties([positions['elbow'][2], positions['wrist'][2]])

        # Update gripper (simplified V-shape)
        wrist_pos = np.array(positions['wrist'])
        end_pos = np.array(positions['end'])
        gripper_vector = end_pos - wrist_pos
        gripper_length = np.linalg.norm(gripper_vector)
        if gripper_length > 0:
            gripper_dir = gripper_vector / gripper_length
        else:
            gripper_dir = np.array([0, 0, -1])

        # Simple perpendicular for gripper opening
        if abs(gripper_dir[2]) < 0.9:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([0, 1, 0])
        right = np.cross(gripper_dir, ref)
        right = right / np.linalg.norm(right)

        # Apply wrist rotation
        j4_rad = np.radians(self.sim.joints[3])
        cos_j4 = np.cos(j4_rad)
        sin_j4 = np.sin(j4_rad)

        # Finger 1
        finger1_end = end_pos + 0.04 * (0.7 * gripper_dir + 0.7 * (cos_j4 * right + sin_j4 * np.cross(gripper_dir, right)))
        # Finger 2
        finger2_end = end_pos + 0.04 * (0.7 * gripper_dir - 0.7 * (cos_j4 * right + sin_j4 * np.cross(gripper_dir, right)))

        self.line_finger1.set_data([wrist_pos[0], finger1_end[0]], [wrist_pos[1], finger1_end[1]])
        self.line_finger1.set_3d_properties([wrist_pos[2], finger1_end[2]])

        self.line_finger2.set_data([wrist_pos[0], finger2_end[0]], [wrist_pos[1], finger2_end[1]])
        self.line_finger2.set_3d_properties([wrist_pos[2], finger2_end[2]])

        # Update scatter positions
        self.scatter_should._offsets3d = ([positions['shoulder'][0]], [positions['shoulder'][1]], [positions['shoulder'][2]])
        self.scatter_elbow._offsets3d = ([positions['elbow'][0]], [positions['elbow'][1]], [positions['elbow'][2]])
        self.scatter_wrist._offsets3d = ([positions['wrist'][0]], [positions['wrist'][1]], [positions['wrist'][2]])

        # Update title
        joint_info = f"J1:{self.sim.joints[0]:.0f} J2:{self.sim.joints[1]:.0f} J3:{self.sim.joints[2]:.0f} J4:{self.sim.joints[3]:.0f}"
        view_name = self._get_view_name()
        physics_status = "ON" if self.sim.physics_enabled else "OFF"
        self.title_3d.set_text(f'Qarm 3D Simulation | {view_name}\n{joint_info} | Physics: {physics_status}')

    def _update_help_text(self):
        """Update help text based on current view state"""
        # Get label for each key based on current view
        key_labels = {
            '1': 'BACK' if self.current_view == 'front' else 'FRONT',
            '2': 'RIGHT' if self.current_view == 'left' else 'LEFT',
            '3': 'BOTTOM' if self.current_view == 'top' else 'TOP',
            '4': 'ISO'
        }
        help_text = f"1:{key_labels['1']}  2:{key_labels['2']}  3:{key_labels['3']}  4:{key_labels['4']}  C:CAM  Q:QUIT"
        self.text_help.set_text(help_text)

    def _get_view_name(self):
        """Get the name of the current view"""
        # Check preset views
        for name, (e, a) in self.preset_views.items():
            if abs(self.elev - e) < 1 and abs(self.azim - a) < 1:
                return name.upper()
        # Custom view
        return f"CUSTOM ({self.elev:.0f}°, {self.azim:.0f}°)"

    def run(self):
        """Run main loop with incremental updates"""
        print("\n>>> Starting main loop...\n")

        try:
            while self.running and plt.fignum_exists(self.fig.number):
                # Process camera
                result = self.process_camera()

                # Update views (no clear, just update data)
                self.update_camera_view(result)
                self.update_3d_view()

                # Redraw canvas
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

                # Keep event loop alive
                plt.pause(0.03)  # ~30 FPS target

                # FPS counter
                self.frame_count += 1
                if self.frame_count % 60 == 0:
                    print(f"Running... frame: {self.frame_count} | FPS: ~{30}")

        except KeyboardInterrupt:
            print("\nProgram interrupted")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.tracker.close()
        plt.ioff()
        plt.close('all')
        print("Program ended")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Hand Pose Control Qarm - Right Arm Auto Mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    args = parser.parse_args()

    print(f"Using camera: {args.camera}")
    print("Mode: RIGHT ARM AUTO TRACKING\n")

    demo = HandControlQarmDemo(camera_id=args.camera)
    demo.run()


if __name__ == "__main__":
    main()
