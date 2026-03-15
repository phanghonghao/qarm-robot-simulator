"""
Hand Pose Control Qarm Demo - Right Arm Only (Optimized)
Uses matplotlib with incremental updates for smooth performance

Left: Camera hand pose capture (YOLOv8-pose, RIGHT ARM ONLY)
Right: Qarm 3D simulation - FOUR VIEW DISPLAY (FRONT, LEFT, TOP, ISO)

Key Controls:
    'c' - Toggle camera ON/OFF
    'p' - Toggle physics limits
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
        self.fig = plt.figure(figsize=(14, 6))
        self.fig.canvas.manager.set_window_title('Hand Pose Control Qarm - Four View Display')
        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05, wspace=0.15, hspace=0.2)

        # Create grid layout: left side camera, right side 2x2 3D views
        # Left: Camera (1 column)
        self.ax_cam = self.fig.add_subplot(1, 2, 1)

        # Right: 2x2 grid for 3D views
        # Position: [left, bottom, width, height]
        self.ax_front = self.fig.add_axes([0.55, 0.52, 0.20, 0.38], projection='3d')  # FRONT (top-left)
        self.ax_left = self.fig.add_axes([0.77, 0.52, 0.20, 0.38], projection='3d')   # LEFT (top-right)
        self.ax_top = self.fig.add_axes([0.55, 0.08, 0.20, 0.38], projection='3d')    # TOP (bottom-left)
        self.ax_iso = self.fig.add_axes([0.77, 0.08, 0.20, 0.38], projection='3d')    # ISO (bottom-right)

        # Store axes in dictionary for easy access
        self.axes_3d = {
            'front': self.ax_front,
            'left': self.ax_left,
            'top': self.ax_top,
            'iso': self.ax_iso
        }

        # Four view configurations
        self.view_configs = {
            'front': {'elev': 0, 'azim': 0, 'title': 'FRONT'},
            'left': {'elev': 0, 'azim': 90, 'title': 'LEFT'},
            'top': {'elev': 90, 'azim': -90, 'title': 'TOP'},
            'iso': {'elev': 30, 'azim': 45, 'title': 'ISO'}
        }

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
        help_text = "C:CAM  P:PHYS  Q:QUIT"
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
        """Initialize 3D plot objects for all four views (created once, updated in loop)"""
        # Axis limits (same for all views)
        limit = 0.3
        z_limit = 0.6

        # Store line and scatter objects for each view
        self.lines_3d = {}
        self.scatters_3d = {}
        self.titles_3d = {}

        for view_name, ax in self.axes_3d.items():
            config = self.view_configs[view_name]

            # Set view angle
            ax.view_init(elev=config['elev'], azim=config['azim'])

            # Set axis limits
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit * 0.3, limit)
            ax.set_zlim(0, z_limit)
            ax.set_xlabel('X', fontsize=6)
            ax.set_ylabel('Y', fontsize=6)
            ax.set_zlabel('Z', fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # Store line objects for this view
            self.lines_3d[view_name] = {
                'base': ax.plot([], [], [], 'k-', linewidth=5, alpha=0.7)[0],
                'upper': ax.plot([], [], [], 'b-', linewidth=4, marker='o', markersize=4)[0],
                'fore': ax.plot([], [], [], 'g-', linewidth=3, marker='o', markersize=3)[0],
                'finger1': ax.plot([], [], [], color='cyan', linewidth=1.5, marker='o', markersize=2)[0],
                'finger2': ax.plot([], [], [], color='magenta', linewidth=1.5, marker='o', markersize=2)[0]
            }

            # Store scatter objects for this view
            self.scatters_3d[view_name] = {
                'shoulder': ax.scatter([], [], [], color='red', s=50, marker='s', alpha=0.8),
                'elbow': ax.scatter([], [], [], color='purple', s=40, marker='o', alpha=0.8),
                'wrist': ax.scatter([], [], [], color='brown', s=30, marker='o', alpha=0.8)
            }

            # Title
            self.titles_3d[view_name] = ax.set_title(f"Qarm {config['title']}", fontsize=9, fontweight='bold')

    def _print_controls(self):
        """Print control instructions"""
        print("=" * 50)
        print("  HAND POSE CONTROL QARM - AUTO MODE")
        print("=" * 50)
        print("\n  MODE: AUTO (Right Arm Tracking)")
        print("  DISPLAY: Four View (FRONT, LEFT, TOP, ISO)")
        print("\n  KEYBOARD:")
        print("  ═════════")
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
        """Update all four 3D views (incremental, no clear)"""
        positions = self.sim.forward_kinematics(self.sim.joints)

        # Calculate gripper positions (shared by all views)
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

        # Finger positions
        finger1_end = end_pos + 0.04 * (0.7 * gripper_dir + 0.7 * (cos_j4 * right + sin_j4 * np.cross(gripper_dir, right)))
        finger2_end = end_pos + 0.04 * (0.7 * gripper_dir - 0.7 * (cos_j4 * right + sin_j4 * np.cross(gripper_dir, right)))

        # Update each view
        for view_name, ax in self.axes_3d.items():
            lines = self.lines_3d[view_name]
            scatters = self.scatters_3d[view_name]

            # Update lines
            lines['base'].set_data([positions['base'][0], positions['shoulder'][0]],
                                   [positions['base'][1], positions['shoulder'][1]])
            lines['base'].set_3d_properties([positions['base'][2], positions['shoulder'][2]])

            lines['upper'].set_data([positions['shoulder'][0], positions['elbow'][0]],
                                    [positions['shoulder'][1], positions['elbow'][1]])
            lines['upper'].set_3d_properties([positions['shoulder'][2], positions['elbow'][2]])

            lines['fore'].set_data([positions['elbow'][0], positions['wrist'][0]],
                                   [positions['elbow'][1], positions['wrist'][1]])
            lines['fore'].set_3d_properties([positions['elbow'][2], positions['wrist'][2]])

            # Update gripper
            lines['finger1'].set_data([wrist_pos[0], finger1_end[0]], [wrist_pos[1], finger1_end[1]])
            lines['finger1'].set_3d_properties([wrist_pos[2], finger1_end[2]])

            lines['finger2'].set_data([wrist_pos[0], finger2_end[0]], [wrist_pos[1], finger2_end[1]])
            lines['finger2'].set_3d_properties([wrist_pos[2], finger2_end[2]])

            # Update scatter positions
            scatters['shoulder']._offsets3d = ([positions['shoulder'][0]], [positions['shoulder'][1]], [positions['shoulder'][2]])
            scatters['elbow']._offsets3d = ([positions['elbow'][0]], [positions['elbow'][1]], [positions['elbow'][2]])
            scatters['wrist']._offsets3d = ([positions['wrist'][0]], [positions['wrist'][1]], [positions['wrist'][2]])

            # Update title with joint info
            joint_info = f"J1:{self.sim.joints[0]:.0f} J2:{self.sim.joints[1]:.0f} J3:{self.sim.joints[2]:.0f} J4:{self.sim.joints[3]:.0f}"
            physics_status = "ON" if self.sim.physics_enabled else "OFF"
            self.titles_3d[view_name].set_text(f"{self.view_configs[view_name]['title']}\n{joint_info} | P:{physics_status}")

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
