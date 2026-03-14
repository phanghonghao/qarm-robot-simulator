# Qarm Robotic Arm Simulator

A 4-DOF robotic arm simulator with 3D visualization, inverse kinematics, and interactive control.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **3D Visualization**: Real-time 3D rendering of the Qarm robotic arm
- **Forward & Inverse Kinematics**: Move the arm to any reachable position
- **Interactive Control**: Keyboard shortcuts for joint control
- **Joint View Mode**: Visualize rotation axes for each joint
- **Random Target Mode**: Automatically generate and reach random targets
- **Precomputed Workspace**: 2000+ verified reachable target positions

## Quick Start

### Desktop Version

```bash
# Install dependencies
pip install -r requirements.txt

# Precompute reachable targets (run once)
python precompute_targets.py

# Run the simulator
python qarm_sim.py
```

### Web Version (Mobile Compatible)

```bash
# Run the web server
python web_app.py

# Open browser to http://localhost:5000
# Or access from mobile: http://YOUR_PC_IP:5000
```

## Controls

| Key | Function |
|-----|----------|
| `1`/`2` | Joint 1 (Base rotation) ±30° |
| `3`/`4` | Joint 2 (Shoulder pitch) ±30° |
| `5`/`6` | Joint 3 (Elbow bend) ±30° |
| `7`/`8` | Joint 4 (Wrist rotation) ±30° |
| `a`/`d` | Rotate view |
| `+`/`-` | Zoom in/out |
| `v` | Cycle joint view mode (J1→J2→J3→J4→OFF) |
| `g` | Generate random target and animate |
| `r` | Reset to default pose |
| `q` | Quit |

## Robot Specifications

**Qarm 4-DOF (RRRR Configuration)**

| Joint | Type | Range | Motion |
|-------|------|-------|--------|
| J1 | Revolute | -170° ~ +170° | Base rotation (Yaw) |
| J2 | Revolute | -85° ~ +85° | Shoulder pitch |
| J3 | Revolute | -95° ~ +75° | Elbow bend |
| J4 | Revolute | -160° ~ +160° | Wrist rotation |

**Link Lengths:**
- Base height: 0.08 m
- Link 1 (upper arm): 0.14 m
- Link 2 (forearm): 0.14 m
- Link 3 (hand): 0.06 m

## Project Structure

```
SRT/
├── qarm_sim.py              # Main simulator (desktop)
├── web_app.py               # Flask web server
├── precompute_targets.py    # Generate reachable target library
├── reachable_targets.csv    # Precomputed target positions
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Demo

- [Video Demo](https://youtu.be/) - Add your demo video link here

## License

MIT License - feel free to use this project for learning and development.

## Author

Created for robotics simulation and control research.
