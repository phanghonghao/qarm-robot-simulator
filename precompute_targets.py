"""
Generate reachable target points for Qarm simulator.
Run this once to create the CSV file with reachable targets.

Usage: python precompute_targets.py
"""

import numpy as np
import csv
import sys

# Import simulator
from qarm_sim import QarmSimulator


def precompute_workspace_targets(output_file='reachable_targets.csv',
                                 num_samples=5000):
    """
    Precompute reachable target points by sampling joint space.
    Uses Forward Kinematics to guarantee reachability.

    Args:
        output_file: CSV file path to save targets
        num_samples: Number of random joint configurations to sample
    """
    # Create a temporary simulator instance for FK (no figure needed)
    print("Creating simulator instance...")
    sim = QarmSimulator(create_figure=False)

    targets = []
    duplicates = 0

    print(f"\nSampling {num_samples} random joint configurations...")
    print("This may take a moment...\n")

    for i in range(num_samples):
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{num_samples} ({100*(i+1)/num_samples:.0f}%)")

        # Generate random joint angles within limits
        j1 = np.random.uniform(*sim.JOINT_LIMITS['joint1'])
        j2 = np.random.uniform(*sim.JOINT_LIMITS['joint2'])
        j3 = np.random.uniform(*sim.JOINT_LIMITS['joint3'])
        j4 = np.random.uniform(*sim.JOINT_LIMITS['joint4'])
        joints = [j1, j2, j3, j4]

        # Use FK to get end effector position (guaranteed reachable)
        pos = sim.forward_kinematics(joints)['end']

        # Check if duplicate (too close to existing points)
        is_duplicate = False
        for t in targets:
            if np.linalg.norm(pos - t['pos']) < 0.015:  # 1.5cm threshold
                is_duplicate = True
                duplicates += 1
                break

        if not is_duplicate:
            targets.append({
                'pos': pos,
                'joints': joints
            })

    # Write to CSV file
    print(f"\nWriting {len(targets)} unique targets to {output_file}...")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['x', 'y', 'z', 'j1', 'j2', 'j3', 'j4'])

        # Data rows
        for t in targets:
            writer.writerow([
                f"{t['pos'][0]:.6f}",
                f"{t['pos'][1]:.6f}",
                f"{t['pos'][2]:.6f}",
                f"{t['joints'][0]:.4f}",
                f"{t['joints'][1]:.4f}",
                f"{t['joints'][2]:.4f}",
                f"{t['joints'][3]:.4f}"
            ])

    print(f"\n" + "="*50)
    print(f"Precomputation Complete!")
    print(f"="*50)
    print(f"  Total samples tried:     {num_samples}")
    print(f"  Unique targets found:    {len(targets)}")
    print(f"  Duplicates filtered:     {duplicates}")
    print(f"  Output file:             {output_file}")
    print(f"="*50)
    print(f"\nYou can now run 'python qarm_sim.py' and press 'g'")
    print(f"to randomly select from these {len(targets)} reachable targets.\n")


if __name__ == '__main__':
    precompute_workspace_targets()
