"""
Qarm Robotic Arm Simulator - Web Version
Access from any device with a web browser (PC, tablet, mobile)

Usage:
    python web_app.py
    Open browser: http://localhost:5000
"""

import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Import simulator
from qarm_sim import QarmSimulator

app = Flask(__name__)

# Global simulator instance
sim = QarmSimulator(create_figure=False)


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get current robot status"""
    positions = sim.forward_kinematics(sim.joints)
    return jsonify({
        'joints': sim.joints,
        'end_effector': positions['end'].tolist(),
        'view': {'elev': sim.elev, 'azim': sim.azim, 'zoom': sim.zoom}
    })


@app.route('/api/plot')
def get_plot():
    """Generate 3D plot and return as base64 image"""
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Render robot
    _render_robot(ax, sim.joints, sim.elev, sim.azim)

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return jsonify({'image': img_base64})


@app.route('/api/joint', methods=['POST'])
def set_joint():
    """Set a specific joint angle"""
    data = request.json
    joint_idx = int(data.get('joint', 0))
    delta = float(data.get('delta', 0))

    if 0 <= joint_idx < 4:
        sim.joints[joint_idx] = np.clip(
            sim.joints[joint_idx] + delta,
            *list(sim.JOINT_LIMITS.values())[joint_idx]
        )

    return get_status()


@app.route('/api/view', methods=['POST'])
def set_view():
    """Change view angle"""
    data = request.json
    view_type = data.get('view')

    preset_views = {
        'front': (0, 0),
        'side': (0, 90),
        'top': (90, -90),
        'iso': (30, 45)
    }

    if view_type in preset_views:
        sim.elev, sim.azim = preset_views[view_type]

    return jsonify({'elev': sim.elev, 'azim': sim.azim})


@app.route('/api/reset', methods=['POST'])
def reset_robot():
    """Reset to default pose"""
    sim.joints = [0, -90, 0, 0]
    sim.elev = 15
    sim.azim = 45
    sim.zoom = 1.0
    sim.target_point = None
    return get_status()


@app.route('/api/random_target', methods=['POST'])
def random_target():
    """Generate random target and move to it"""
    if sim.reachable_targets and len(sim.reachable_targets) > 0:
        target = np.random.choice(sim.reachable_targets)
        sim.target_point = target['pos']
        sim.joints = target['joints']

    return get_status()


def _render_robot(ax, joints, elev, azim):
    """Render robot on given axes"""
    positions = sim.forward_kinematics(joints)

    # Draw wall
    Y, Z = np.meshgrid(np.linspace(0, 0.5, 2), np.linspace(0, sim.wall_mount_height + 0.2, 2))
    X = np.zeros_like(Y) - 0.05
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    # Draw ground
    X, Y = np.meshgrid(np.linspace(-0.3, 0.3, 2), np.linspace(0, 0.5, 2))
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.05, color='green')

    # Draw links
    ax.plot([positions['shoulder'][0], positions['elbow'][0]],
            [positions['shoulder'][1], positions['elbow'][1]],
            [positions['shoulder'][2], positions['elbow'][2]],
            'b-', linewidth=6, marker='o', markersize=10, label='Upper Arm')

    ax.plot([positions['elbow'][0], positions['wrist'][0]],
            [positions['elbow'][1], positions['wrist'][1]],
            [positions['elbow'][2], positions['wrist'][2]],
            'g-', linewidth=5, marker='o', markersize=8, label='Forearm')

    ax.plot([positions['wrist'][0], positions['end'][0]],
            [positions['wrist'][1], positions['end'][1]],
            [positions['wrist'][2], positions['end'][2]],
            color='orange', linewidth=4, marker='*', markersize=15, label='Hand')

    # Draw joints
    for key, color, size in [('shoulder', 'red', 200), ('elbow', 'purple', 150),
                              ('wrist', 'brown', 100)]:
        ax.scatter([positions[key][0]], [positions[key][1]], [positions[key][2]],
                   color=color, s=size, marker='s' if key == 'shoulder' else 'o', alpha=0.8)

    # Draw target if exists
    if sim.target_point is not None:
        ax.plot([sim.target_point[0]], [sim.target_point[1]], [sim.target_point[2]],
                'r*', markersize=20, markeredgecolor='darkred', markeredgewidth=2, zorder=100)

    # Set limits
    max_reach = (sim.LINK_LENGTHS['link1'] + sim.LINK_LENGTHS['link2'] + sim.LINK_LENGTHS['link3'])
    limit = max(max_reach * 1.3, 0.4) / sim.zoom
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit * 0.3, limit)
    ax.set_zlim(0, max(sim.wall_mount_height + max_reach * 0.5, 0.6))
    ax.view_init(elev=elev, azim=azim)

    # Labels
    joint_info = f"J1:{joints[0]:.0f} J2:{joints[1]:.0f} J3:{joints[2]:.0f} J4:{joints[3]:.0f}"
    ax.set_title(f'Qarm Simulator Web\n{joint_info}', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left', fontsize=8)


if __name__ == '__main__':
    print("=" * 50)
    print("Qarm Simulator - Web Server")
    print("=" * 50)
    print("Server running at:")
    print("  Local:   http://localhost:5000")
    print("  Network: http://YOUR_PC_IP:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 50)

    # Run on all network interfaces so mobile can access
    app.run(host='0.0.0.0', port=5000, debug=True)
