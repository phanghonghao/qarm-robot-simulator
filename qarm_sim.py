"""
Qarm Robotic Arm Simulator - 3D Visualization (Wall-Mounted Configuration)
Wall-mounted: Base fixed above, similar to human shoulder embedded in torso
"""

import numpy as np
import csv
import matplotlib
# Use interactive backend for keyboard support
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.widgets import Button, Slider


class Arrow3D(FancyArrowPatch):
    """3D arrow class for drawing coordinate axes"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class QarmSimulator:
    """
    Qarm 4-DOF Robotic Arm Simulator - Wall-Mounted Version
    Uses Matplotlib 3D for visualization

    Wall-mounted configuration:
    - Base fixed to wall above (like human shoulder embedded in torso)
    - Joint 1: Shoulder rotation (left/right swing)
    - Joint 2: Upper arm pitch (forward/backward, 0deg = vertical downward)
    - Joint 3: Forearm bend relative to upper arm
    - Joint 4: Wrist rotation
    """

    # Qarm link lengths (meters, based on Quanser Qarm specifications)
    LINK_LENGTHS = {
        'base_height': 0.08,           # Base height
        'shoulder_box': [0.35, 0.1, 0.14],  # Shoulder joint box (X, Y, Z dimensions)
        'link1': 0.3536,               # Link 1 (upper arm: shoulder to elbow)
        'link2': 0.25,                 # Link 2 (forearm: elbow to wrist)
        'link3': 0.16,                 # Link 3 (V-gripper: wrist to end, length = 0.16m)
    }

    # Qarm joint limits (degrees, based on hardware specifications)
    JOINT_LIMITS = {
        'joint1': (-170, 170),   # Base rotation
        'joint2': (-85, 85),     # Shoulder pitch
        'joint3': (-95, 75),     # Elbow bend
        'joint4': (-160, 160),   # Wrist rotation
    }

    def __init__(self, wall_mount_height=0.5, create_figure=True):
        """
        Initialize simulator

        Args:
            wall_mount_height: Wall mounting height (m), base distance from ground
            create_figure: Whether to create figure and axes (default True)
        """
        # Initial joint angles [degrees] - Natural hanging position
        # Joint 1: 0 deg (facing forward)
        # Joint 2: -90 deg (vertical downward, natural hanging)
        # Joint 3: 0 deg (forearm straight)
        # Joint 4: 0 deg (wrist flat)
        self.joints = [0, -90, 0, 0]

        self.wall_mount_height = wall_mount_height

        # Create figure and 3D axes only if requested
        self.fig = None
        self.ax = None
        if create_figure:
            self.fig = plt.figure(figsize=(6, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')

        self.zoom = 1.0  # Zoom factor
        self.view_angle = 45  # Initial viewing angle (azimuth)
        self.elev = 15  # Initial elevation angle
        self.azim = 45  # Initial azimuth angle
        self._is_paused = False

        # Debug info
        self.last_key = "None"
        self.last_action = "No action"
        self.key_count = 0

        # Physics constraint toggle (press 'p' to toggle)
        # When False: joints can move freely in ±180° range
        # When True: joints are constrained to hardware limits
        self.physics_enabled = False

        # Command history for display
        self.command_history = []  # List of (command, action) tuples
        self.max_history = 8  # Max lines to display

        # View animation state
        self.is_animating = False
        self.animation_target = None
        self.view_buttons = []  # Store button references
        self.view_buttons_dict = {}  # Store buttons by view name for label updates

        # Current view state for toggle functionality
        self.current_view = 'default'  # 'front', 'back', 'left', 'right', 'top', 'bottom', 'default'

        # Joint view mode (press 'v' to cycle)
        self.joint_view_mode = None  # None, 1, 2, 3, or 4
        self.joint_view_configs = {
            1: {'name': 'J1 Base (Yaw)', 'elev': 90, 'azim': 0, 'zoom': 1.2, 'axis': 'Z'},
            2: {'name': 'J2 Shoulder (Pitch)', 'elev': 0, 'azim': 90, 'zoom': 1.5, 'axis': 'Horizontal'},
            3: {'name': 'J3 Elbow (Pitch)', 'elev': 0, 'azim': 90, 'zoom': 2.0, 'axis': 'Horizontal'},
            4: {'name': 'J4 Wrist (Roll)', 'elev': 0, 'azim': 0, 'zoom': 2.5, 'axis': 'Forearm'},
        }

        # Target point for IK (display only)
        self.target_point = None

        # Load precomputed reachable targets from CSV
        self.reachable_targets = self._load_targets() if create_figure else []

    def forward_kinematics(self, joints):
        """
        Forward kinematics: Calculate joint positions (wall-mounted configuration)

        Args:
            joints: [joint1, joint2, joint3, joint4] angles (degrees)
                    joint1: Base rotation (horizontal plane, 0deg = +Y direction, perpendicular to wall)
                    joint2: Shoulder pitch (-90deg = vertical down, 0deg = horizontal, positive = up)
                    joint3: Elbow bend (relative to upper arm, positive = bend up)
                    joint4: Wrist rotation

        Returns:
            dict: (x, y, z) coordinates of each joint
        """
        # Convert to radians
        j1, j2, j3, j4 = [np.radians(angle) for angle in joints]

        # Base position (fixed to wall above)
        # Assume wall is in YZ plane (X=0), base mounted on wall
        base = np.array([0, 0, self.wall_mount_height])

        # Base top (shoulder joint position)
        shoulder_pos = base + np.array([0, 0, self.LINK_LENGTHS['base_height']])

        # Joint 1 (shoulder) rotation direction
        # j1 controls rotation in horizontal plane, 0deg faces +Y (perpendicular to wall)
        horizontal_angle = j1

        # Joint 2 (shoulder pitch):
        # j2 = -90deg: vertical downward (natural hanging)
        # j2 = 0deg: horizontal forward
        # j2 = 90deg: vertical upward
        # Note: j2 is already in radians from the conversion above
        shoulder_pitch = j2  # Already in radians!

        # Upper arm direction using spherical coordinates
        # pitch_angle: angle from horizontal plane
        # -π/2 (down), 0 (horizontal), π/2 (up)

        # Calculate components:
        # When j2=-π/2 (down): cos(-π/2)=0, sin(-π/2)=-1 -> Y=0, Z=-L
        # When j2=0 (forward): cos(0)=1, sin(0)=0 -> Y=L, Z=0
        elbow_local = np.array([
            self.LINK_LENGTHS['link1'] * np.cos(shoulder_pitch) * np.sin(horizontal_angle),  # X
            self.LINK_LENGTHS['link1'] * np.cos(shoulder_pitch) * np.cos(horizontal_angle),  # Y (forward)
            self.LINK_LENGTHS['link1'] * np.sin(shoulder_pitch),                           # Z (up/down, negative is down)
        ])
        elbow_pos = shoulder_pos + elbow_local

        # Joint 3 (elbow bend): bend angle relative to upper arm
        # j3=0 means forearm continues in upper arm direction
        # j3>0 means forearm bends upward
        elbow_bend = j3  # Already in radians from line 137
        forearm_pitch = shoulder_pitch + elbow_bend

        # Forearm end (wrist position)
        wrist_local = np.array([
            self.LINK_LENGTHS['link2'] * np.cos(forearm_pitch) * np.sin(horizontal_angle),
            self.LINK_LENGTHS['link2'] * np.cos(forearm_pitch) * np.cos(horizontal_angle),
            self.LINK_LENGTHS['link2'] * np.sin(forearm_pitch),
        ])
        wrist_pos = elbow_pos + wrist_local

        # Joint 4 (wrist): end effector
        # j4 affects end effector direction, simplified here
        wrist_bend = j4  # Already in radians from line 137
        end_pitch = forearm_pitch + wrist_bend
        end_local = np.array([
            self.LINK_LENGTHS['link3'] * np.cos(end_pitch) * np.sin(horizontal_angle),
            self.LINK_LENGTHS['link3'] * np.cos(end_pitch) * np.cos(horizontal_angle),
            self.LINK_LENGTHS['link3'] * np.sin(end_pitch),
        ])
        end_pos = wrist_pos + end_local

        return {
            'base': base,
            'shoulder': shoulder_pos,
            'elbow': elbow_pos,
            'wrist': wrist_pos,
            'end': end_pos
        }

    def _get_joint_limits(self, joint_name):
        """
        Get effective joint limits based on physics_enabled setting.

        Args:
            joint_name: 'joint1', 'joint2', 'joint3', or 'joint4'

        Returns:
            (min_limit, max_limit) tuple in degrees
        """
        if self.physics_enabled:
            # Use hardware limits
            return self.JOINT_LIMITS[joint_name]
        else:
            # Free movement: ±180 degrees
            return (-180, 180)

    def set_joints(self, joints):
        """
        Set joint angles

        Args:
            joints: [joint1, joint2, joint3, joint4] angles (degrees)
        """
        self.joints = joints

    def inverse_kinematics(self, target_pos, elbow_up=True):
        """
        Inverse kinematics: Calculate joint angles to reach target position
        Uses analytical geometric solution for 4-DOF arm.

        Args:
            target_pos: (x, y, z) target position for end effector
            elbow_up: If True, prefer elbow-up solution; else elbow-down

        Returns:
            [j1, j2, j3, j4] joint angles in degrees, or None if unreachable
        """
        x, y, z = target_pos

        # Shoulder position
        shoulder_z = self.wall_mount_height + self.LINK_LENGTHS['base_height']
        shoulder_pos = np.array([0, 0, shoulder_z])

        # Vector from shoulder to target
        dx = x - shoulder_pos[0]
        dy = y - shoulder_pos[1]
        dz = z - shoulder_pos[2]

        # Link lengths
        L1 = self.LINK_LENGTHS['link1']  # Upper arm
        L2 = self.LINK_LENGTHS['link2']  # Forearm
        L3 = self.LINK_LENGTHS['link3']  # Hand/end effector

        # Calculate horizontal distance and total distance
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        total_dist = np.sqrt(horizontal_dist**2 + dz**2)

        # Adjust target to wrist position (subtract end effector offset)
        # We want the wrist to be at a position that allows end effector to reach target
        # For simplicity, target the wrist position directly
        target_dist = max(0.01, total_dist - L3 * 0.5)  # Back off slightly for L3

        # Check reachability for wrist position
        max_reach = L1 + L2
        min_reach = abs(L1 - L2)

        if target_dist > max_reach or target_dist < min_reach:
            return None  # Unreachable

        # Joint 1: Base rotation (yaw) - angle from Y axis towards X
        j1 = np.degrees(np.arctan2(dx, dy))
        j1 = np.clip(j1, *self._get_joint_limits('joint1'))

        # Two-link IK for joints 2 and 3 (shoulder and elbow)
        # Using law of cosines

        # Elbow angle (j3)
        cos_j3 = (target_dist**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_j3 = np.clip(cos_j3, -1, 1)

        if elbow_up:
            j3_rad = np.arccos(cos_j3)
        else:
            j3_rad = -np.arccos(cos_j3)

        j3 = np.degrees(j3_rad)

        # Apply joint limits to j3
        j3 = np.clip(j3, *self._get_joint_limits('joint3'))

        # Shoulder pitch (j2)
        # Angle to target from horizontal plane
        angle_to_target = np.arctan2(dz, horizontal_dist)

        # Angle offset due to elbow
        cos_offset = (L1**2 + target_dist**2 - L2**2) / (2 * L1 * target_dist)
        cos_offset = np.clip(cos_offset, -1, 1)
        offset_angle = np.arccos(cos_offset)

        if elbow_up:
            j2_rad = angle_to_target + offset_angle - np.pi/2
        else:
            j2_rad = angle_to_target - offset_angle - np.pi/2

        j2 = np.degrees(j2_rad)

        # Apply joint limits to j2
        j2 = np.clip(j2, *self._get_joint_limits('joint2'))

        # Joint 4: Wrist rotation
        # Align end effector with target direction
        j4 = 0  # Simplified - keep wrist flat

        return [j1, j2, j3, j4]

    def _load_targets(self, csv_file='reachable_targets.csv'):
        """
        Load precomputed reachable targets from CSV file.

        Args:
            csv_file: Path to CSV file with reachable targets

        Returns:
            List of dicts with 'pos' (np.array) and 'joints' (list)
        """
        targets = []
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    targets.append({
                        'pos': np.array([float(row['x']), float(row['y']), float(row['z'])]),
                        'joints': [float(row['j1']), float(row['j2']),
                                  float(row['j3']), float(row['j4'])]
                    })
            print(f"Loaded {len(targets)} reachable targets from {csv_file}")
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found.")
            print(f"Generating fallback targets... (run 'python precompute_targets.py' for more)")
            targets = self._generate_fallback_targets()
        except Exception as e:
            print(f"Error loading targets: {e}")
            print(f"Generating fallback targets...")
            targets = self._generate_fallback_targets()

        return targets

    def _generate_fallback_targets(self, num_samples=100):
        """
        Generate fallback reachable targets using FK sampling.
        Used when CSV file is not available.

        Args:
            num_samples: Number of targets to generate

        Returns:
            List of dicts with 'pos' and 'joints'
        """
        targets = []
        for _ in range(num_samples):
            j1 = np.random.uniform(*self._get_joint_limits('joint1'))
            j2 = np.random.uniform(*self._get_joint_limits('joint2'))
            j3 = np.random.uniform(*self._get_joint_limits('joint3'))
            j4 = np.random.uniform(*self._get_joint_limits('joint4'))
            joints = [j1, j2, j3, j4]
            pos = self.forward_kinematics(joints)['end']
            targets.append({'pos': pos, 'joints': joints})
        return targets

    def generate_random_target(self):
        """
        Generate a random reachable target point in workspace
        Uses FK verification to ensure the target is actually reachable.

        Returns:
            (x, y, z) target position, or None if generation fails
        """
        max_attempts = 200
        L1 = self.LINK_LENGTHS['link1']
        L2 = self.LINK_LENGTHS['link2']
        L3 = self.LINK_LENGTHS['link3']

        shoulder_z = self.wall_mount_height + self.LINK_LENGTHS['base_height']

        # Precise workspace bounds
        min_reach = max(0.02, abs(L1 - L2) - L3)  # Minimum reach
        max_reach = (L1 + L2 + L3) * 0.95  # Maximum reach (95% for safety)

        for attempt in range(max_attempts):
            # Generate random spherical coordinates
            r = np.random.uniform(min_reach, max_reach)

            # Azimuth (for j1): limit to realistic joint range
            theta = np.radians(np.random.uniform(-140, 140))

            # Elevation: mostly forward and downward (natural workspace)
            phi = np.radians(np.random.uniform(-70, 40))

            # Convert to Cartesian
            x = r * np.cos(phi) * np.sin(theta)
            y = r * np.cos(phi) * np.cos(theta)
            z = shoulder_z + r * np.sin(phi)

            target = (x, y, z)

            # Try IK with both elbow configurations
            for elbow_up in [True, False]:
                joints = self.inverse_kinematics(target, elbow_up=elbow_up)

                if joints is not None:
                    # CRITICAL: Verify with FK
                    fk_result = self.forward_kinematics(joints)
                    fk_end = fk_result['end']

                    # Calculate error
                    error = np.linalg.norm(fk_end - np.array(target))

                    # Only accept if error is small (< 1.5cm)
                    if error < 0.015:
                        return target

        return None

    def animate_to_target(self, target_joints, duration=1.5):
        """
        Smooth animation to target joint angles

        Args:
            target_joints: [j1, j2, j3, j4] target angles in degrees
            duration: Animation duration in seconds
        """
        if self.is_animating:
            return

        self.is_animating = True
        start_joints = self.joints.copy()

        diffs = [target_joints[i] - start_joints[i] for i in range(4)]

        fps = 50
        num_frames = int(duration * fps)
        if num_frames < 10:
            num_frames = 10

        for i in range(num_frames + 1):
            t = i / num_frames
            t_smooth = t * t * (3 - 2 * t)  # Smoothstep

            self.joints = [start_joints[j] + diffs[j] * t_smooth for j in range(4)]

            self.ax.clear()
            positions = self.forward_kinematics(self.joints)
            self._redraw_quick(positions)
            plt.pause(1 / fps)

        # Final position
        self.joints = target_joints
        self.ax.clear()
        positions = self.forward_kinematics(self.joints)
        self._redraw_quick(positions)

        self.is_animating = False

    def render(self, show=True, block=False):
        """
        Render current state as 3D image (matplotlib 3D)

        Args:
            show: Whether to display immediately
            block: Whether to block window

        Returns:
            matplotlib Figure: Rendered figure object
        """
        self.ax.clear()

        # Calculate joint positions
        positions = self.forward_kinematics(self.joints)

        # Draw wall (reference)
        self._draw_wall()

        # Draw ground
        self._draw_ground()

        # Draw base mount (wall to shoulder joint)
        base_x = [positions['base'][0], positions['shoulder'][0]]
        base_y = [positions['base'][1], positions['shoulder'][1]]
        base_z = [positions['base'][2], positions['shoulder'][2]]
        self.ax.plot(base_x, base_y, base_z, 'k-', linewidth=8, alpha=0.7, label='Base Mount')

        # Draw robotic arm links
        # Upper arm (shoulder to elbow) - blue
        self.ax.plot(
            [positions['shoulder'][0], positions['elbow'][0]],
            [positions['shoulder'][1], positions['elbow'][1]],
            [positions['shoulder'][2], positions['elbow'][2]],
            'b-', linewidth=6, marker='o', markersize=10, label='Upper Arm'
        )

        # Forearm (elbow to wrist) - green
        self.ax.plot(
            [positions['elbow'][0], positions['wrist'][0]],
            [positions['elbow'][1], positions['wrist'][1]],
            [positions['elbow'][2], positions['wrist'][2]],
            'g-', linewidth=5, marker='o', markersize=8, label='Forearm'
        )

        # Draw V-shaped gripper (shows wrist rotation clearly)
        self._draw_gripper(positions['wrist'], positions['end'], self.joints)

        # Draw joints
        # Draw shoulder joint box
        self._draw_shoulder_box(positions['shoulder'])

        # Shoulder joint
        self.ax.scatter(
            [positions['shoulder'][0]], [positions['shoulder'][1]], [positions['shoulder'][2]],
            color='red', s=200, marker='s', alpha=0.8, label='Shoulder'
        )
        # Elbow joint
        self.ax.scatter(
            [positions['elbow'][0]], [positions['elbow'][1]], [positions['elbow'][2]],
            color='purple', s=150, marker='o', alpha=0.8, label='Elbow'
        )
        # Wrist joint
        self.ax.scatter(
            [positions['wrist'][0]], [positions['wrist'][1]], [positions['wrist'][2]],
            color='brown', s=100, marker='o', alpha=0.8, label='Wrist'
        )

        # Draw axis indicator at each joint
        for joint_name in ['shoulder', 'elbow', 'wrist']:
            self._draw_axis_indicator(positions[joint_name])

        # Draw joint rotation axis if in joint view mode
        if self.joint_view_mode is not None:
            self._draw_joint_rotation_axis(positions, self.joint_view_mode)

        # Draw target point if exists (use plot for more reliable 3D rendering)
        if self.target_point is not None:
            self.ax.plot(
                [self.target_point[0]], [self.target_point[1]], [self.target_point[2]],
                'r*', markersize=25, markeredgecolor='darkred',
                markeredgewidth=2, label='Target', zorder=100
            )

        # Set axis ranges - based on maximum workspace
        # Calculate maximum reach of the arm
        max_reach = (self.LINK_LENGTHS['link1'] +
                     self.LINK_LENGTHS['link2'] +
                     self.LINK_LENGTHS['link3'])
        # Use max reach with safety margin, applied to zoom
        limit = max(max_reach * 1.3, 0.4) / self.zoom
        # X range: centered, allows full rotation
        self.ax.set_xlim(-limit, limit)
        # Y range: forward reach (arm extends in +Y direction)
        self.ax.set_ylim(-limit * 0.3, limit)
        # Z range: from ground to above base
        self.ax.set_zlim(0, max(self.wall_mount_height + max_reach * 0.5, 0.6))

        self.ax.set_xlabel('X (m)', fontsize=10)
        self.ax.set_ylabel('Y (m)', fontsize=10)
        self.ax.set_zlabel('Z (m)', fontsize=10)

        # Title
        joint_info = f"J1:{self.joints[0]:.0f} J2:{self.joints[1]:.0f} J3:{self.joints[2]:.0f} J4:{self.joints[3]:.0f}"
        view_name = self.get_view_name()
        physics_status = "ON (Hardware)" if self.physics_enabled else "OFF (Free ±180°)"
        self.ax.set_title(f'Qarm Wall-Mounted Simulation\n{joint_info}\nZoom: {self.zoom:.1f}x | View: {view_name} ({self.elev:.0f}°, {self.azim:.0f}°)\nPhysics Limits: {physics_status}', fontsize=12)

        # Set view angle
        self.ax.view_init(elev=self.elev, azim=self.azim)

        # Legend
        self.ax.legend(loc='upper left', fontsize=8)

        # Grid
        self.ax.grid(True, alpha=0.3)

        # Always draw the canvas to update display
        self.fig.canvas.draw()

        if show:
            plt.show(block=block)

        return self.fig

    def _draw_wall(self):
        """Draw wall (X=0 plane)"""
        y = np.linspace(0, 0.5, 10)
        z = np.linspace(0, self.wall_mount_height + 0.2, 10)
        Y, Z = np.meshgrid(y, z)
        X = np.zeros_like(Y)
        self.ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')

        # Draw wall frame
        self.ax.plot([0, 0], [0, 0.5], [0, 0], 'k-', linewidth=2)
        self.ax.plot([0, 0], [0, 0.5], [self.wall_mount_height + 0.2, self.wall_mount_height + 0.2], 'k-', linewidth=2)
        self.ax.plot([0, 0], [0, 0], [0, self.wall_mount_height + 0.2], 'k-', linewidth=2)
        self.ax.plot([0, 0], [0.5, 0.5], [0, self.wall_mount_height + 0.2], 'k-', linewidth=2)

        # Wall label
        self.ax.text(0, 0.25, self.wall_mount_height + 0.25, 'Wall', fontsize=10, ha='center')

    def _draw_ground(self):
        """Draw ground"""
        x = np.linspace(-0.3, 0.3, 10)
        y = np.linspace(0, 0.5, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.05, color='green')

    def _draw_shoulder_box(self, shoulder_pos):
        """
        Draw shoulder joint as a rectangular box

        Args:
            shoulder_pos: Center position of shoulder joint (x, y, z)
        """
        sx, sy, sz = self.LINK_LENGTHS['shoulder_box']
        x, y, z = shoulder_pos

        # Define the 8 vertices of the box
        # Box is centered at shoulder_pos
        vertices = np.array([
            [x - sx/2, y - sy/2, z - sz/2],  # 0: back-bottom-left
            [x + sx/2, y - sy/2, z - sz/2],  # 1: back-bottom-right
            [x + sx/2, y + sy/2, z - sz/2],  # 2: back-top-right
            [x - sx/2, y + sy/2, z - sz/2],  # 3: back-top-left
            [x - sx/2, y - sy/2, z + sz/2],  # 4: front-bottom-left
            [x + sx/2, y - sy/2, z + sz/2],  # 5: front-bottom-right
            [x + sx/2, y + sy/2, z + sz/2],  # 6: front-top-right
            [x - sx/2, y + sy/2, z + sz/2],  # 7: front-top-left
        ])

        # Define the 12 edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # back face
            [4, 5], [5, 6], [6, 7], [7, 4],  # front face
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting edges
        ]

        # Draw edges
        for edge in edges:
            self.ax.plot3D(
                [vertices[edge[0], 0], vertices[edge[1], 0]],
                [vertices[edge[0], 1], vertices[edge[1], 1]],
                [vertices[edge[0], 2], vertices[edge[1], 2]],
                'k-', linewidth=2, alpha=0.6
            )

        # Draw semi-transparent faces
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Define faces using vertex indices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # back
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # front
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # bottom
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # top
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        ]

        # Add faces as a polygon collection
        face_collection = Poly3DCollection(faces, alpha=0.2, facecolor='lightblue', edgecolor='black')
        self.ax.add_collection3d(face_collection)

        # Add label
        self.ax.text(x, y, z + sz/2 + 0.05, 'Shoulder', fontsize=9, ha='center')

    def _draw_axis_indicator(self, origin):
        """Draw 3D axis indicator at specified position"""
        axis_length = 0.08

        # X axis (red)
        self.ax.quiver(
            origin[0], origin[1], origin[2],
            axis_length, 0, 0,
            color='red', arrow_length_ratio=0.3, linewidth=2
        )
        self.ax.text(origin[0] + axis_length * 1.2, origin[1], origin[2], 'X', color='red', fontsize=10, fontweight='bold')

        # Y axis (green)
        self.ax.quiver(
            origin[0], origin[1], origin[2],
            0, axis_length, 0,
            color='green', arrow_length_ratio=0.3, linewidth=2
        )
        self.ax.text(origin[0], origin[1] + axis_length * 1.2, origin[2], 'Y', color='green', fontsize=10, fontweight='bold')

        # Z axis (blue)
        self.ax.quiver(
            origin[0], origin[1], origin[2],
            0, 0, axis_length,
            color='blue', arrow_length_ratio=0.3, linewidth=2
        )
        self.ax.text(origin[0], origin[1], origin[2] + axis_length * 1.2, 'Z', color='blue', fontsize=10, fontweight='bold')

    def _draw_joint_rotation_axis(self, positions, joint_num):
        """
        Draw rotation axis and direction for a specific joint

        Args:
            positions: Joint positions dict from forward_kinematics
            joint_num: 1, 2, 3, or 4 (which joint to visualize)
        """
        j1, j2, j3, j4 = [np.radians(angle) for angle in self.joints]

        if joint_num == 1:
            # J1: Z-axis rotation at shoulder
            origin = positions['shoulder']
            axis_dir = np.array([0, 0, 1])  # Z axis
            color = 'magenta'
            label = 'J1: Yaw (Z-axis)'

        elif joint_num == 2:
            # J2: Horizontal axis rotation at shoulder (perpendicular to arm direction)
            origin = positions['shoulder']
            # Axis perpendicular to the horizontal projection of the arm
            arm_horizontal = np.array([np.sin(j1), np.cos(j1), 0])
            axis_dir = np.cross(arm_horizontal, np.array([0, 0, 1]))
            color = 'orange'
            label = 'J2: Pitch'

        elif joint_num == 3:
            # J3: Horizontal axis rotation at elbow
            origin = positions['elbow']
            # Same orientation as J2 axis
            arm_horizontal = np.array([np.sin(j1), np.cos(j1), 0])
            axis_dir = np.cross(arm_horizontal, np.array([0, 0, 1]))
            color = 'cyan'
            label = 'J3: Pitch'

        elif joint_num == 4:
            # J4: Along forearm axis rotation at wrist
            origin = positions['wrist']
            # Axis along the forearm direction
            forearm_dir = positions['wrist'] - positions['elbow']
            forearm_dir = forearm_dir / np.linalg.norm(forearm_dir)
            axis_dir = forearm_dir
            color = 'purple'
            label = 'J4: Roll'

        else:
            return

        # Draw rotation axis (thick line with arrows)
        axis_length = 0.12
        start = origin - axis_dir * axis_length
        end = origin + axis_dir * axis_length

        self.ax.plot(
            [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
            color=color, linewidth=4, alpha=0.8, linestyle='--'
        )

        # Draw axis arrow
        self.ax.quiver(
            origin[0], origin[1], origin[2],
            axis_dir[0], axis_dir[1], axis_dir[2],
            length=axis_length * 0.8, color=color, arrow_length_ratio=0.3,
            linewidth=3, alpha=0.9
        )

        # Draw rotation direction arc (curved arrow)
        # Create arc points
        theta = np.linspace(0, np.pi/2, 20)
        arc_radius = axis_length * 0.6

        # Find perpendicular vectors for arc plane
        if abs(axis_dir[2]) > 0.9:  # Z-axis
            v1 = np.array([1, 0, 0])
            v2 = np.array([0, 1, 0])
        else:
            v1 = np.array([0, 0, 1])
            v2 = np.cross(axis_dir, v1)
            v2 = v2 / np.linalg.norm(v2)
            v1 = np.cross(axis_dir, v2)

        arc_points = []
        for t in theta:
            pt = origin + arc_radius * (np.cos(t) * v1 + np.sin(t) * v2)
            arc_points.append(pt)

        arc_points = np.array(arc_points)
        self.ax.plot(
            arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
            color=color, linewidth=3, alpha=0.7
        )

        # Arrow head for rotation direction
        arrow_pt = arc_points[-1]
        tangent = v2 * arc_radius * 0.3
        self.ax.quiver(
            arrow_pt[0], arrow_pt[1], arrow_pt[2],
            tangent[0], tangent[1], tangent[2],
            color=color, arrow_length_ratio=0.5, linewidth=2
        )

        # Label
        self.ax.text(
            origin[0] + axis_length, origin[1], origin[2] + axis_length,
            label, color=color, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    def _draw_gripper(self, wrist_pos, end_pos, joints):
        """
        Draw V-shaped gripper (140° opening angle, 0.16m length)

        Args:
            wrist_pos: Wrist joint position (x, y, z)
            end_pos: End effector position (x, y, z)
            joints: Current joint angles [j1, j2, j3, j4]
        """
        # V-gripper parameters
        v_angle = 140  # V opening angle in degrees
        v_angle_rad = np.radians(v_angle / 2)  # Half angle for each side

        # Calculate gripper direction vector (from wrist to end)
        gripper_vector = np.array(end_pos) - np.array(wrist_pos)
        gripper_length = np.linalg.norm(gripper_vector)
        if gripper_length > 0:
            gripper_dir = gripper_vector / gripper_length
        else:
            gripper_dir = np.array([0, 0, -1])

        # Get wrist rotation angle (j4)
        j4_rad = np.radians(joints[3])

        # Create a coordinate system at the wrist
        forward = gripper_dir

        # Create a perpendicular vector for the gripper opening plane
        if abs(forward[2]) < 0.9:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([0, 1, 0])

        # Right direction (perpendicular to forward and reference)
        right = np.cross(forward, ref)
        right_length = np.linalg.norm(right)
        if right_length > 0:
            right = right / right_length

        # Up direction (perpendicular to both)
        up = np.cross(right, forward)

        # Apply wrist rotation to right and up vectors
        cos_j4 = np.cos(j4_rad)
        sin_j4 = np.sin(j4_rad)
        right_rot = cos_j4 * right + sin_j4 * up
        up_rot = -sin_j4 * right + cos_j4 * up

        # Calculate V-gripper finger directions
        # Each finger opens at v_angle/2 from the forward direction
        # Finger 1 (cyan) - opens to the left
        finger1_dir = np.cos(v_angle_rad) * forward + np.sin(v_angle_rad) * right_rot
        finger1_dir = finger1_dir / np.linalg.norm(finger1_dir)

        # Finger 2 (magenta) - opens to the right
        finger2_dir = np.cos(v_angle_rad) * forward - np.sin(v_angle_rad) * right_rot
        finger2_dir = finger2_dir / np.linalg.norm(finger2_dir)

        # Finger 1 (cyan): from wrist to end at 70° angle
        finger1_end = np.array(wrist_pos) + gripper_length * finger1_dir
        self.ax.plot(
            [wrist_pos[0], finger1_end[0]],
            [wrist_pos[1], finger1_end[1]],
            [wrist_pos[2], finger1_end[2]],
            color='cyan', linewidth=4, marker='o', markersize=3
        )

        # Finger 2 (magenta): from wrist to end at -70° angle
        finger2_end = np.array(wrist_pos) + gripper_length * finger2_dir
        self.ax.plot(
            [wrist_pos[0], finger2_end[0]],
            [wrist_pos[1], finger2_end[1]],
            [wrist_pos[2], finger2_end[2]],
            color='magenta', linewidth=4, marker='o', markersize=3
        )

        # Draw end effector marker (at the tip of the V, not actual end_pos)
        # The tip of V is where both fingers meet if extended back
        # But we show the marker at the visual center between finger tips

    def set_zoom(self, zoom_delta):
        """
        Adjust zoom factor

        Args:
            zoom_delta: Zoom change amount (positive = zoom in, negative = zoom out)
        """
        self.zoom = np.clip(self.zoom + zoom_delta, 0.3, 3.0)

    def rotate_view(self, angle_delta):
        """
        Rotate view angle

        Args:
            angle_delta: Angle change amount
        """
        self.view_angle = (self.view_angle + angle_delta) % 360
        self.azim = self.view_angle  # Keep azim in sync

    def set_preset_view(self, view_name):
        """
        Set a preset view

        Args:
            view_name: Preset view name ('front', 'side', 'top', 'iso')
        """
        preset_views = {
            'front': (0, 0),          # Front view
            'side': (0, 90),          # Side view
            'top': (90, -90),         # Top view
            'iso': (30, 45),          # Isometric view
            'default': (15, 45),      # Default view
        }

        if view_name in preset_views:
            self.elev, self.azim = preset_views[view_name]
            self.view_angle = self.azim  # Keep view_angle in sync for a/d rotation

        return preset_views.get(view_name, preset_views['default'])

    def get_view_name(self):
        """Get the name of the current view"""
        view_map = {
            (0, 0): 'FRONT',
            (0, 180): 'BACK',
            (0, 90): 'LEFT SIDE',
            (0, -90): 'RIGHT SIDE',
            (90, -90): 'TOP',
            (-90, -90): 'BOTTOM',
            (30, 45): 'ISOMETRIC',
            (15, 45): 'DEFAULT',
        }
        # Round to handle floating point
        key = (round(self.elev), round(self.azim))
        return view_map.get(key, 'CUSTOM')

    def _add_joint_sliders(self):
        """Add joint control sliders at the bottom of the figure (2x2 grid layout)"""
        # Slider configuration: [label, joint_index, x_position, y_position, color]
        # 2x2 grid layout - adjusted positions to avoid overlap
        slider_configs = [
            ('J1: Base', 0, 0.06, 0.30, 'lightcoral'),   # Top-left
            ('J2: Shoulder', 1, 0.54, 0.30, 'lightblue'),  # Top-right
            ('J3: Elbow', 2, 0.06, 0.22, 'lightgreen'),   # Bottom-left
            ('J4: Wrist', 3, 0.54, 0.22, 'lightyellow'),  # Bottom-right
        ]

        # Store sliders and their axes for hover detection
        self.joint_sliders = []
        self.slider_axes = []  # Store slider axes for hover detection

        for label, joint_idx, x_pos, y_pos, color in slider_configs:
            # Create slider axis [left, bottom, width, height]
            ax_slider = self.fig.add_axes([x_pos, y_pos, 0.40, 0.025])

            # Get joint limits (based on physics_enabled setting)
            j_min, j_max = self._get_joint_limits(f'joint{joint_idx + 1}')

            # Create slider
            slider = Slider(
                ax=ax_slider,
                label=label,
                valmin=j_min,
                valmax=j_max,
                valinit=self.joints[joint_idx],
                color=color,
                valfmt='%0.1f°'
            )

            # Connect callback (with joint index)
            slider.on_changed(lambda val, idx=joint_idx: self._slider_callback(idx, val))

            self.joint_sliders.append(slider)
            self.slider_axes.append({'ax': ax_slider, 'slider': slider, 'index': joint_idx})

        # Enable hover sliding
        self._hover_slider_active = None  # Track which slider is being hovered
        self._last_mouse_x = None
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_slider_hover)
        self.fig.canvas.mpl_connect('button_release_event', self._on_slider_release)

    def _on_slider_hover(self, event):
        """Handle mouse/touch hover over sliders for click-free control"""
        if event.inaxes is None:
            return

        # Check if mouse is over any slider
        for slider_info in self.slider_axes:
            ax = slider_info['ax']
            if event.inaxes == ax:
                slider = slider_info['slider']
                joint_idx = slider_info['index']

                # Convert x position to slider value
                x_min, x_max = ax.get_xlim()
                val = slider.valmin + (event.xdata - x_min) / (x_max - x_min) * (slider.valmax - slider.valmin)
                val = np.clip(val, slider.valmin, slider.valmax)

                # Update joint angle
                self.joints[joint_idx] = val
                slider.set_val(val)

                # Redraw
                self.ax.clear()
                positions = self.forward_kinematics(self.joints)
                self._redraw_quick(positions)
                self.fig.canvas.draw_idle()
                break

    def _on_slider_release(self, event):
        """Handle mouse button release"""
        pass  # Can be used for additional logic if needed

    def _slider_callback(self, joint_idx, value):
        """Callback for joint slider changes"""
        self.joints[joint_idx] = value

        # Quick redraw without full render
        self.ax.clear()
        positions = self.forward_kinematics(self.joints)
        self._redraw_quick(positions)
        self.fig.canvas.draw_idle()

    def _update_slider_ranges(self):
        """Update slider ranges based on current physics_enabled setting"""
        if not hasattr(self, 'joint_sliders'):
            return

        for i, slider in enumerate(self.joint_sliders):
            joint_name = f'joint{i + 1}'
            j_min, j_max = self._get_joint_limits(joint_name)
            slider.valmin = j_min
            slider.valmax = j_max
            # Reset slider to current joint value, clipped to new range
            current_val = np.clip(self.joints[i], j_min, j_max)
            self.joints[i] = current_val
            slider.set_val(current_val)
        # Refresh display
        self.fig.canvas.draw_idle()

    def _add_view_buttons(self):
        """Add view control buttons at the bottom of the figure"""
        # Button configuration: [label, view_name, x_position]
        button_configs = [
            ('Front', 'front', 0.10),
            ('Left', 'left', 0.24),
            ('Top', 'top', 0.38),
            ('Reset', 'reset', 0.55),
        ]

        # Store button references with their view names for label updates
        self.view_buttons_dict = {}

        for label, view_name, x_pos in button_configs:
            # Create button axis
            ax_btn = self.fig.add_axes([x_pos, 0.02, 0.12, 0.06])
            btn = Button(ax_btn, label, color='lightblue', hovercolor='skyblue')

            # Connect callback
            if view_name == 'reset':
                btn.on_clicked(self._reset_callback)
                self.view_buttons_dict['reset'] = btn
            else:
                btn.on_clicked(lambda event, vn=view_name: self._view_button_callback(vn, btn))
                self.view_buttons_dict[view_name] = btn

            self.view_buttons.append(btn)

    def _update_button_labels(self):
        """Update button labels based on current view"""
        # Label pairs for toggling
        label_pairs = {
            'front': 'Back',
            'back': 'Front',
            'left': 'Right',
            'right': 'Left',
            'top': 'Bottom',
            'bottom': 'Top',
        }

        # Update Front/Back button
        if self.current_view in ['front', 'back']:
            self.view_buttons_dict['front'].label.set_text(label_pairs[self.current_view])
        else:
            self.view_buttons_dict['front'].label.set_text('Front')

        # Update Left/Right button
        if self.current_view in ['left', 'right']:
            self.view_buttons_dict['left'].label.set_text(label_pairs[self.current_view])
        else:
            self.view_buttons_dict['left'].label.set_text('Left')

        # Update Top/Bottom button
        if self.current_view in ['top', 'bottom']:
            self.view_buttons_dict['top'].label.set_text(label_pairs[self.current_view])
        else:
            self.view_buttons_dict['top'].label.set_text('Top')

        self.fig.canvas.draw_idle()

    def _view_button_callback(self, view_name, btn):
        """Callback for view buttons - toggle between opposite views on second click"""
        if self.is_animating:
            return  # Ignore if already animating

        # View toggle mappings
        view_toggle_map = {
            'front': 'back',
            'back': 'front',
            'left': 'right',
            'right': 'left',
            'top': 'bottom',
            'bottom': 'top',
        }

        # Determine target view based on current view
        if self.current_view == view_name:
            # Currently on this view, toggle to opposite
            target_view = view_toggle_map.get(view_name, view_name)
        else:
            # Go to the requested view
            target_view = view_name

        # Get target view angles
        preset_views = {
            'front': (0, 0),
            'back': (0, 180),
            'left': (0, 90),
            'right': (0, -90),
            'top': (90, -90),
            'bottom': (-90, -90),
            'default': (15, 45),
        }

        if target_view in preset_views:
            target_elev, target_azim = preset_views[target_view]
            self.current_view = target_view
            self.animate_view_transition(target_elev, target_azim)
            self._update_button_labels()

    def _reset_callback(self, event):
        """Callback for reset button"""
        if self.is_animating:
            return

        self.joints = [0, -90, 0, 0]
        self.zoom = 1.0
        self.current_view = 'default'

        # Reset sliders to initial positions
        for i, slider in enumerate(self.joint_sliders):
            slider.set_val(self.joints[i])
            slider.eventson = False  # Disable callback during reset
            slider.eventson = True   # Re-enable callback

        self.animate_view_transition(15, 45)

    def animate_view_transition(self, target_elev, target_azim):
        """
        Animate smooth transition to target view

        Args:
            target_elev: Target elevation angle
            target_azim: Target azimuth angle
        """
        self.is_animating = True

        # Disable buttons during animation
        for btn in self.view_buttons:
            btn.active = False

        start_elev = self.elev
        start_azim = self.azim

        # Calculate angle differences (handle wrap-around for azimuth)
        elev_diff = target_elev - start_elev
        azim_diff = target_azim - start_azim

        # Handle azimuth wrap-around (take shortest path)
        if azim_diff > 180:
            azim_diff -= 360
        elif azim_diff < -180:
            azim_diff += 360

        # Calculate number of frames (step size = 8 degrees for faster animation)
        max_diff = max(abs(elev_diff), abs(azim_diff))
        num_frames = int(max_diff / 8) + 1
        if num_frames < 4:
            num_frames = 4  # Minimum frames for smoothness
        elif num_frames > 30:
            num_frames = 30  # Maximum frames for quick response

        # Animate
        for i in range(num_frames + 1):
            t = i / num_frames

            # Easing function for smooth animation (ease-in-out)
            t_smooth = t * t * (3 - 2 * t)  # Smoothstep

            # Interpolate angles
            self.elev = start_elev + elev_diff * t_smooth
            self.azim = start_azim + azim_diff * t_smooth
            self.view_angle = self.azim

            # Re-render
            self.ax.clear()
            positions = self.forward_kinematics(self.joints)
            self._redraw_quick(positions)

            plt.pause(0.015)  # ~66fps for smoother, faster animation

        # Ensure we end exactly at target
        self.elev = target_elev
        self.azim = target_azim
        self.view_angle = self.azim

        # Re-enable buttons
        for btn in self.view_buttons:
            btn.active = True

        self.is_animating = False

    def interactive_mode(self):
        """Enable interactive mode with SLIDERS and keyboard control"""
        # Turn on interactive mode
        plt.ion()

        # Adjust figure layout to make room for sliders and buttons at bottom
        self.fig.subplots_adjust(bottom=0.35, left=0.1, right=0.95)

        # Position window on the right side (60% width for better visibility)
        try:
            # Get screen size
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()

            # Calculate window position (right side, 60% width)
            window_width = int(screen_width * 0.6)
            window_height = int(screen_height * 0.8)
            window_x = int(screen_width * 0.4)
            window_y = int(screen_height * 0.1)

            # Set window position
            backend = plt.get_backend()
            if 'TkAgg' in backend:
                self.fig.canvas.manager.window.wm_geometry(f"+{window_x}+{window_y}")
                self.fig.canvas.manager.window.wm_geometry(f"{window_width}x{window_height}")
        except:
            print("Could not set window position")

        # Add joint control sliders
        self._add_joint_sliders()

        # Add view control buttons
        self._add_view_buttons()

        # Connect keyboard event for REAL-TIME control
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        # Also connect key release for continuous movement support
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)

        print("\n" + "=" * 60)
        print("   QARM INTERACTIVE CONTROL MODE - SLIDERS")
        print("=" * 60)
        print("\n  CONTROLS:")
        print("  ════════════════════════════════════════════════")
        print("  SLIDERS (Bottom):")
        print("    J1: Base rotation     (-170° ~ +170°)")
        print("    J2: Shoulder pitch    (-85° ~ +85°)")
        print("    J3: Elbow bend        (-95° ~ +75°)")
        print("    J4: Wrist rotation    (-160° ~ +160°)")
        print("\n  KEYBOARD (Alternative):")
        print("    1/2  - Joint 1  ±30°    3/4  - Joint 2  ±30°")
        print("    5/6  - Joint 3  ±30°    7/8  - Joint 4  ±30°")
        print("\n  VIEW:")
        print("    a/d  - Rotate view     or click view buttons")
        print("    +/-  - Zoom in/out     r    - Reset pose")
        print("    g    - Random target + IK + animation")
        print("    p    - Toggle physics limits (ON: hardware / OFF: free ±180°)")
        print("    q    - Quit")
        print("  ════════════════════════════════════════════════")
        print("  >>> USE SLIDERS OR KEYBOARD TO CONTROL JOINTS <<<")
        print("=" * 60 + "\n")

        # Initial render
        self._add_to_history("START", "Real-time mode ready")
        self.render(show=False)

        # Keep window alive with event loop
        print(">>> Waiting for keyboard input (click plot window)...\n")

        try:
            while plt.fignum_exists(self.fig.number):
                # Use pause to keep the event loop alive
                plt.pause(0.05)
        except KeyboardInterrupt:
            print("\nExiting...")

        # Turn off interactive mode
        plt.ioff()
        print("Interactive mode closed.")

    def _process_command(self, cmd):
        """Process command string and return action description"""
        step = 30  # Angle step (increased from 5 to 30 degrees)
        action = "Unknown command"

        if cmd == 'q':
            self._add_to_history("q", "Quit")
            plt.close(self.fig)
            return "QUIT"

        elif cmd == 'r':
            old_joints = self.joints.copy()
            self.joints = [0, -90, 0, 0]
            self.zoom = 1.0
            self.view_angle = 45
            action = f"RESET: {old_joints} -> {self.joints}"

        elif cmd == '1':
            old_val = self.joints[0]
            self.joints[0] = np.clip(self.joints[0] - step, *self._get_joint_limits('joint1'))
            action = f"J1: {old_val:.0f} -> {self.joints[0]:.0f} (-)"
        elif cmd == '2':
            old_val = self.joints[0]
            self.joints[0] = np.clip(self.joints[0] + step, *self._get_joint_limits('joint1'))
            action = f"J1: {old_val:.0f} -> {self.joints[0]:.0f} (+)"

        elif cmd == '3':
            old_val = self.joints[1]
            self.joints[1] = np.clip(self.joints[1] - step, *self._get_joint_limits('joint2'))
            action = f"J2: {old_val:.0f} -> {self.joints[1]:.0f} (-)"
        elif cmd == '4':
            old_val = self.joints[1]
            self.joints[1] = np.clip(self.joints[1] + step, *self._get_joint_limits('joint2'))
            action = f"J2: {old_val:.0f} -> {self.joints[1]:.0f} (+)"

        elif cmd == '5':
            old_val = self.joints[2]
            self.joints[2] = np.clip(self.joints[2] - step, *self._get_joint_limits('joint3'))
            action = f"J3: {old_val:.0f} -> {self.joints[2]:.0f} (-)"
        elif cmd == '6':
            old_val = self.joints[2]
            self.joints[2] = np.clip(self.joints[2] + step, *self._get_joint_limits('joint3'))
            action = f"J3: {old_val:.0f} -> {self.joints[2]:.0f} (+)"

        elif cmd == '7':
            old_val = self.joints[3]
            self.joints[3] = np.clip(self.joints[3] - step, *self._get_joint_limits('joint4'))
            action = f"J4: {old_val:.0f} -> {self.joints[3]:.0f} (-)"
        elif cmd == '8':
            old_val = self.joints[3]
            self.joints[3] = np.clip(self.joints[3] + step, *self._get_joint_limits('joint4'))
            action = f"J4: {old_val:.0f} -> {self.joints[3]:.0f} (+)"

        elif cmd in ['+', '=']:
            self.set_zoom(0.1)
            action = f"Zoom: {self.zoom:.1f}x"
        elif cmd in ['-', '_']:
            self.set_zoom(-0.1)
            action = f"Zoom: {self.zoom:.1f}x"

        elif cmd == 'a':
            self.rotate_view(-5)
            action = f"View: {self.view_angle:.0f} deg"
        elif cmd == 'd':
            self.rotate_view(5)
            action = f"View: {self.view_angle:.0f} deg"

        elif cmd == 'p':
            # Toggle physics constraints
            self.physics_enabled = not self.physics_enabled
            status = "ON" if self.physics_enabled else "OFF"
            limits_mode = "Hardware" if self.physics_enabled else "Free (±180°)"
            action = f"Physics: {status} ({limits_mode})"
            print(f"  -> Physics constraints: {status}")
            print(f"  -> Joint limits: {limits_mode}")
            # Update slider ranges to reflect new limits
            self._update_slider_ranges()

        elif cmd == 'h':
            print("\n--- Commands (30° per step) ---")
            print("1/2: J1 +/- | 3/4: J2 +/- | 5/6: J3 +/- | 7/8: J4 +/-")
            print("+/-: Zoom    | a/d: View   | v: Joint View | g: Random Target")
            print("r: Reset     | p: Physics  | q: Quit")
            print("---------------\n")
            action = "Help displayed"

        else:
            action = f"Unknown: '{cmd}'"

        # Add to history
        self._add_to_history(cmd, action)

        # Print to terminal
        print(f"  -> {action}")

        return action

    def _add_to_history(self, cmd, action):
        """Add command to history"""
        self.command_history.append((cmd, action))
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)

    def _on_key_press(self, event):
        """Keyboard event handler"""
        if event.key is None:
            return

        # Update debug info
        self.last_key = f"'{event.key}'"
        self.key_count += 1

        # Debug output to terminal
        print(f"[DEBUG] Key #{self.key_count}: '{event.key}' | Joints: [{self.joints[0]:.0f}, {self.joints[1]:.0f}, {self.joints[2]:.0f}, {self.joints[3]:.0f}]")

        if event.key == 'q':
            self.last_action = "Quit"
            plt.close(self.fig)
            return

        step = 30  # Angle step per adjustment (increased from 5 to 30)
        action = None

        if event.key == '1':
            old_val = self.joints[0]
            self.joints[0] = np.clip(self.joints[0] - step, *self._get_joint_limits('joint1'))
            action = f"J1: {old_val:.0f} -> {self.joints[0]:.0f} (DECREASE)"
        elif event.key == '2':
            old_val = self.joints[0]
            self.joints[0] = np.clip(self.joints[0] + step, *self._get_joint_limits('joint1'))
            action = f"J1: {old_val:.0f} -> {self.joints[0]:.0f} (INCREASE)"
        elif event.key == '3':
            old_val = self.joints[1]
            self.joints[1] = np.clip(self.joints[1] - step, *self._get_joint_limits('joint2'))
            action = f"J2: {old_val:.0f} -> {self.joints[1]:.0f} (DECREASE)"
        elif event.key == '4':
            old_val = self.joints[1]
            self.joints[1] = np.clip(self.joints[1] + step, *self._get_joint_limits('joint2'))
            action = f"J2: {old_val:.0f} -> {self.joints[1]:.0f} (INCREASE)"
        elif event.key == '5':
            old_val = self.joints[2]
            self.joints[2] = np.clip(self.joints[2] - step, *self._get_joint_limits('joint3'))
            action = f"J3: {old_val:.0f} -> {self.joints[2]:.0f} (DECREASE)"
        elif event.key == '6':
            old_val = self.joints[2]
            self.joints[2] = np.clip(self.joints[2] + step, *self._get_joint_limits('joint3'))
            action = f"J3: {old_val:.0f} -> {self.joints[2]:.0f} (INCREASE)"
        elif event.key == '7':
            old_val = self.joints[3]
            self.joints[3] = np.clip(self.joints[3] - step, *self._get_joint_limits('joint4'))
            action = f"J4: {old_val:.0f} -> {self.joints[3]:.0f} (DECREASE)"
        elif event.key == '8':
            old_val = self.joints[3]
            self.joints[3] = np.clip(self.joints[3] + step, *self._get_joint_limits('joint4'))
            action = f"J4: {old_val:.0f} -> {self.joints[3]:.0f} (INCREASE)"
        elif event.key == '+' or event.key == '=':
            self.set_zoom(0.1)
            action = f"Zoom: {self.zoom:.1f}x"
        elif event.key == '-' or event.key == '_':
            self.set_zoom(-0.1)
            action = f"Zoom: {self.zoom:.1f}x"
        elif event.key == 'a':
            self.rotate_view(-5)
            action = f"View: {self.view_angle:.0f} deg"
        elif event.key == 'd':
            self.rotate_view(5)
            action = f"View: {self.view_angle:.0f} deg"
        elif event.key == 'r':
            self.joints = [0, -90, 0, 0]
            self.zoom = 1.0
            self.view_angle = 45
            self.azim = 45
            self.elev = 15
            self.target_point = None
            action = "RESET to default pose"
        elif event.key == 'v':
            # Cycle joint view mode
            if self.joint_view_mode is None:
                self.joint_view_mode = 1
            else:
                self.joint_view_mode = self.joint_view_mode + 1
                if self.joint_view_mode > 4:
                    self.joint_view_mode = None
            # Apply view configuration
            if self.joint_view_mode is not None:
                config = self.joint_view_configs[self.joint_view_mode]
                self.elev = config['elev']
                self.azim = config['azim']
                self.zoom = config['zoom']
                action = f"Joint View: {config['name']}"
            else:
                self.elev = 15
                self.azim = 45
                self.zoom = 1.0
                action = "Joint View: OFF"
        elif event.key == 'g':
            # Select random target from precomputed library
            if self.reachable_targets and len(self.reachable_targets) > 0:
                # Random selection from library
                target = np.random.choice(self.reachable_targets)
                self.target_point = target['pos']
                target_joints = target['joints']

                # Animate directly (no IK needed - already verified)
                self.animate_to_target(target_joints, duration=1.2)

                pos = self.target_point
                action = f"Target: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                self._add_to_history('g', action)
            else:
                action = "No targets - run precompute_targets.py"
                print("Warning: No target library available. Run 'python precompute_targets.py' first.")
        else:
            action = "Unknown key - no action"

        self.last_action = action

        # Re-render after key press - use simple refresh
        self.ax.clear()
        positions = self.forward_kinematics(self.joints)
        self._redraw_quick(positions)

    def _redraw_quick(self, positions):
        """Quick redraw without full setup"""
        # Draw wall
        self._draw_wall()

        # Draw ground
        self._draw_ground()

        # Draw base mount
        self.ax.plot(
            [positions['base'][0], positions['shoulder'][0]],
            [positions['base'][1], positions['shoulder'][1]],
            [positions['base'][2], positions['shoulder'][2]],
            'k-', linewidth=8, alpha=0.7
        )

        # Draw links
        self.ax.plot(
            [positions['shoulder'][0], positions['elbow'][0]],
            [positions['shoulder'][1], positions['elbow'][1]],
            [positions['shoulder'][2], positions['elbow'][2]],
            'b-', linewidth=6, marker='o', markersize=10
        )
        self.ax.plot(
            [positions['elbow'][0], positions['wrist'][0]],
            [positions['elbow'][1], positions['wrist'][1]],
            [positions['elbow'][2], positions['wrist'][2]],
            'g-', linewidth=5, marker='o', markersize=8
        )

        # Draw V-shaped gripper (shows wrist rotation clearly)
        self._draw_gripper(positions['wrist'], positions['end'], self.joints)

        # Draw shoulder joint box
        self._draw_shoulder_box(positions['shoulder'])

        # Draw joints (shoulder, elbow, wrist - end effector is drawn in gripper)
        for key, color, size in [('shoulder', 'red', 200), ('elbow', 'purple', 150),
                                  ('wrist', 'brown', 100)]:
            self.ax.scatter(
                [positions[key][0]], [positions[key][1]], [positions[key][2]],
                color=color, s=size, marker='s' if key == 'shoulder' else 'o', alpha=0.8
            )

        # Draw axis indicator at each joint
        for joint_name in ['shoulder', 'elbow', 'wrist']:
            self._draw_axis_indicator(positions[joint_name])

        # Draw joint rotation axis if in joint view mode
        if self.joint_view_mode is not None:
            self._draw_joint_rotation_axis(positions, self.joint_view_mode)

        # Draw target point if exists (use plot for more reliable 3D rendering)
        if self.target_point is not None:
            self.ax.plot(
                [self.target_point[0]], [self.target_point[1]], [self.target_point[2]],
                'r*', markersize=25, markeredgecolor='darkred',
                markeredgewidth=2, zorder=100
            )

        # Update title with joint info
        joint_info = f"J1:{self.joints[0]:.0f} J2:{self.joints[1]:.0f} J3:{self.joints[2]:.0f} J4:{self.joints[3]:.0f}"
        view_name = self.get_view_name()

        # Add joint view mode indicator
        if self.joint_view_mode is not None:
            config = self.joint_view_configs[self.joint_view_mode]
            joint_view_text = f" | [{config['name']}]"
        else:
            joint_view_text = ""

        # Physics limits status
        physics_status = "ON (Hardware)" if self.physics_enabled else "OFF (Free ±180°)"

        title_text = (
            f'Qarm Simulation (Real-Time Mode)\n'
            f'{joint_info} | Zoom: {self.zoom:.1f}x | View: {view_name} ({self.elev:.0f}°, {self.azim:.0f}°){joint_view_text}\n'
            f'Physics: {physics_status}'
        )
        self.ax.set_title(title_text, fontsize=10, fontweight='bold')

        # Add command history as text overlay (2D text in 3D plot)
        history_text = "COMMAND HISTORY:\n"
        for cmd, action in self.command_history:
            history_text += f"  {cmd:3s} -> {action}\n"

        # Position text in 2D coordinates (bottom-left of the plot)
        self.ax.text2D(0.02, 0.02, history_text,
                      transform=self.ax.transAxes,
                      fontsize=8,
                      verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Set limits - based on maximum workspace
        max_reach = (self.LINK_LENGTHS['link1'] +
                     self.LINK_LENGTHS['link2'] +
                     self.LINK_LENGTHS['link3'])
        limit = max(max_reach * 1.3, 0.4) / self.zoom
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit * 0.3, limit)
        self.ax.set_zlim(0, max(self.wall_mount_height + max_reach * 0.5, 0.6))
        self.ax.view_init(elev=self.elev, azim=self.azim)

        # Refresh canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _on_key_release(self, event):
        """Keyboard release event handler (for future continuous movement support)"""
        pass  # Currently not used, but ready for implementation

    def animate(self, trajectory, interval=100):
        """
        Animate trajectory

        Args:
            trajectory: List of joint angles, each element is [j1, j2, j3, j4]
            interval: Frame interval (milliseconds)
        """
        def update(frame):
            self.set_joints(trajectory[frame])
            return self.ax,

        anim = FuncAnimation(
            self.fig, update, frames=len(trajectory),
            interval=interval, blit=False, repeat=True
        )
        plt.show()
        return anim

    def get_end_effector_position(self):
        """Get end effector position"""
        positions = self.forward_kinematics(self.joints)
        return positions['end']


def create_human_arm_comparison():
    """
    Create comparison diagram of human arm and robotic arm structure
    Figure 1: 60% width, positioned on the left
    """
    # Create figure (60% of screen width)
    fig = plt.figure(figsize=(14, 8))

    # Position window on the left side (60% width)
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        # Calculate window position (left side, 60% width)
        window_width = int(screen_width * 0.6)
        window_height = int(screen_height * 0.8)
        window_x = 50  # Small offset from left edge
        window_y = int(screen_height * 0.1)

        # Set window position
        backend = plt.get_backend()
        if 'TkAgg' in backend:
            fig.canvas.manager.window.wm_geometry(f"+{window_x}+{window_y}")
            fig.canvas.manager.window.wm_geometry(f"{window_width}x{window_height}")
    except:
        print("Could not set window position for comparison figure")

    # Left: Human arm structure
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Human Arm Structure (Reference)', fontsize=14, fontweight='bold')

    # Human torso (simplified)
    # Shoulder embedded in torso
    torso_y = np.linspace(-0.1, 0.1, 5)
    torso_z = np.linspace(0.3, 1.3, 10)
    Torso_Y, Torso_Z = np.meshgrid(torso_y, torso_z)
    Torso_X = np.zeros_like(Torso_Y) - 0.05  # Torso inside wall
    ax1.plot_surface(Torso_X, Torso_Y, Torso_Z, alpha=0.3, color='peachpuff')

    # Human arm joint positions (right arm)
    human_shoulder = np.array([0, 0, 1.2])  # Shoulder joint position
    human_elbow = np.array([0, 0.05, 1.0])   # Elbow joint position (slightly forward when hanging)
    human_wrist = np.array([0, 0.08, 0.8])   # Wrist joint position
    human_hand = np.array([0, 0.1, 0.75])    # Hand position

    # Draw human arm
    ax1.plot(
        [human_shoulder[0], human_elbow[0]],
        [human_shoulder[1], human_elbow[1]],
        [human_shoulder[2], human_elbow[2]],
        'tomato', linewidth=8, marker='o', markersize=12, label='Upper Arm (Humerus)'
    )
    ax1.plot(
        [human_elbow[0], human_wrist[0]],
        [human_elbow[1], human_wrist[1]],
        [human_elbow[2], human_wrist[2]],
        'lightgreen', linewidth=6, marker='o', markersize=10, label='Forearm (Ulna/Radius)'
    )
    ax1.plot(
        [human_wrist[0], human_hand[0]],
        [human_wrist[1], human_hand[1]],
        [human_wrist[2], human_hand[2]],
        'orange', linewidth=4, marker='*', markersize=15, label='Hand'
    )

    # Draw joints
    ax1.scatter([human_shoulder[0]], [human_shoulder[1]], [human_shoulder[2]],
                color='red', s=300, marker='s', alpha=0.9, label='Shoulder (Ball Socket)')
    ax1.scatter([human_elbow[0]], [human_elbow[1]], [human_elbow[2]],
                color='purple', s=200, marker='o', alpha=0.9, label='Elbow (Hinge)')
    ax1.scatter([human_wrist[0]], [human_wrist[1]], [human_wrist[2]],
                color='brown', s=150, marker='o', alpha=0.9, label='Wrist (Ellipsoid)')

    # Wall
    y = np.linspace(-0.1, 0.3, 10)
    z = np.linspace(0, 1.5, 10)
    Y, Z = np.meshgrid(y, z)
    X = np.zeros_like(Y) - 0.1
    ax1.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_xlim(-0.2, 0.3)
    ax1.set_ylim(-0.1, 0.3)
    ax1.set_zlim(0, 1.5)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.view_init(elev=10, azim=60)

    # Right: Qarm robotic arm
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Qarm Robotic Arm (Wall-Mounted)', fontsize=14, fontweight='bold')

    # Create simulator without creating its own figure
    sim = QarmSimulator(wall_mount_height=0.5, create_figure=False)
    sim.ax = ax2
    sim.fig = fig

    # Set natural hanging pose (similar to human arm)
    sim.set_joints([0, -90, 0, 0])

    # Draw
    positions = sim.forward_kinematics(sim.joints)

    # Draw wall
    sim._draw_wall()

    # Draw base mount
    ax2.plot(
        [positions['base'][0], positions['shoulder'][0]],
        [positions['base'][1], positions['shoulder'][1]],
        [positions['base'][2], positions['shoulder'][2]],
        'k-', linewidth=10, alpha=0.7, label='Base Mount'
    )

    # Draw robotic arm
    ax2.plot(
        [positions['shoulder'][0], positions['elbow'][0]],
        [positions['shoulder'][1], positions['elbow'][1]],
        [positions['shoulder'][2], positions['elbow'][2]],
        'b-', linewidth=8, marker='o', markersize=12, label='Upper Arm (Link1)'
    )
    ax2.plot(
        [positions['elbow'][0], positions['wrist'][0]],
        [positions['elbow'][1], positions['wrist'][1]],
        [positions['elbow'][2], positions['wrist'][2]],
        'g-', linewidth=6, marker='o', markersize=10, label='Forearm (Link2)'
    )
    ax2.plot(
        [positions['wrist'][0], positions['end'][0]],
        [positions['wrist'][1], positions['end'][1]],
        [positions['wrist'][2], positions['end'][2]],
        color='orange', linewidth=4, marker='*', markersize=15, label='End Effector'
    )

    # Draw joints
    ax2.scatter([positions['shoulder'][0]], [positions['shoulder'][1]], [positions['shoulder'][2]],
                color='red', s=300, marker='s', alpha=0.9, label='Shoulder (J1+J2)')
    ax2.scatter([positions['elbow'][0]], [positions['elbow'][1]], [positions['elbow'][2]],
                color='purple', s=200, marker='o', alpha=0.9, label='Elbow (J3)')
    ax2.scatter([positions['wrist'][0]], [positions['wrist'][1]], [positions['wrist'][2]],
                color='brown', s=150, marker='o', alpha=0.9, label='Wrist (J4)')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_xlim(-0.3, 0.4)
    ax2.set_ylim(0, 0.4)
    ax2.set_zlim(0, 0.7)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.view_init(elev=15, azim=45)

    # Add explanation text
    fig.text(0.5, 0.02,
             'Structure Comparison: Human shoulder embedded in torso, Qarm shoulder fixed to wall mount\n'
             'Both use hanging configuration with arm naturally hanging downward',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('human_arm_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison diagram saved as human_arm_comparison.png")
    # Use non-blocking show so the program continues
    plt.show(block=False)
    plt.pause(0.1)  # Small delay to let window appear


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Qarm Robotic Arm Simulator')
    parser.add_argument('--compare', action='store_true',
                       help='Show human arm vs robotic arm comparison diagram')
    args = parser.parse_args()

    print("=" * 50)
    print("Qarm Robotic Arm Simulator - Wall-Mounted Version")
    print("=" * 50)

    # Only show comparison if explicitly requested with --compare flag
    if args.compare:
        print("\nGenerating human arm vs robotic arm comparison diagram...")
        create_human_arm_comparison()

    print("\n" + "=" * 50)
    print("Starting interactive simulation...")
    print("Tip: Use --compare flag to show human arm comparison")
    print("=" * 50)

    # Create simulator and start interactive mode
    sim = QarmSimulator(wall_mount_height=0.5)
    sim.interactive_mode()
