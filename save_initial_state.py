"""
Save initial state image to docs directory
"""
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from qarm_sim import QarmSimulator

# Create simulator with default joint angles
sim = QarmSimulator()

# Set initial pose (default)
sim.set_joints([0, -90, 0, 0])

# Render and save
fig = sim.render(show=False, block=False)

# Save to docs directory
output_path = 'docs/initial_state.png'
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Initial state saved to: {output_path}")

plt.close(fig)
