#!/usr/bin/env python3
"""Test mathematical correctness of the robot arm simulator."""

import numpy as np
import math
from kinematics import (
    ArmConfig, ArmState, forward_kinematics, forward_kinematics_frames,
    compute_jacobian, solve_ik_analytical, solve_ik_numerical,
    dh_transform, solve_2r, ElbowConfig
)
from constraints import check_self_collision, check_singularity

def test_dh_transform():
    """Test DH transform correctness."""
    print("=== Testing DH Transform ===")
    
    # Test identity transform
    T = dh_transform(0, 0, 0, 0)
    expected = np.eye(4)
    print(f"Identity test: {np.allclose(T, expected)}")
    
    # Test rotation around Z
    T = dh_transform(0, 0, 0, math.pi/2)
    expected = np.array([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    print(f"Z rotation test: {np.allclose(T, expected, atol=1e-10)}")

def test_forward_kinematics_consistency():
    """Test consistency between different FK methods."""
    print("\n=== Testing FK Consistency ===")
    
    config = ArmConfig([3.0, 2.0], [(-2.967, 2.967)] * 2)
    state = ArmState(math.pi/4, [math.pi/6, math.pi/3])
    
    # Method 1: direct FK
    positions1 = forward_kinematics(config, state)
    
    # Method 2: via DH frames
    frames = forward_kinematics_frames(config, state)
    positions2 = [frame[:3, 3] for frame in frames[1:]]  # Skip world frame
    
    print(f"FK consistency: {np.allclose(positions1, positions2, atol=1e-10)}")
    
    # Print positions for verification
    print("Direct FK positions:")
    for i, pos in enumerate(positions1):
        print(f"  {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    print("DH frame positions:")
    for i, pos in enumerate(positions2):
        print(f"  {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

def test_jacobian_correctness():
    """Test Jacobian matrix correctness."""
    print("\n=== Testing Jacobian ===")
    
    config = ArmConfig([2.0, 1.5], [(-2.967, 2.967)] * 2)
    state = ArmState(0.0, [math.pi/4, math.pi/4])
    
    # Compute Jacobian
    J = compute_jacobian(config, state)
    print(f"Jacobian shape: {J.shape}")
    print(f"Jacobian:\n{J}")
    
    # Test Jacobian by finite differences
    delta = 1e-4
    positions = forward_kinematics(config, state)
    ee_pos = positions[-1]
    
    # Test base rotation column
    state_perturbed = state.copy()
    state_perturbed.base_angle += delta
    positions_perturbed = forward_kinematics(config, state_perturbed)
    ee_pos_perturbed = positions_perturbed[-1]
    
    numerical_derivative = (ee_pos_perturbed - ee_pos) / delta
    analytical_derivative = J[:, 0]
    
    print(f"Base rotation derivative - numerical: {numerical_derivative}")
    print(f"Base rotation derivative - analytical: {analytical_derivative}")
    print(f"Match: {np.allclose(numerical_derivative, analytical_derivative, atol=1e-3)}")

def test_2r_solver():
    """Test 2R closed-form solver."""
    print("\n=== Testing 2R Solver ===")
    
    # Test case 1: Simple reach
    L1, L2 = 3.0, 2.0
    r, z = 4.0, 1.0
    result = solve_2r(L1, L2, r, z, ElbowConfig.ELBOW_DOWN)
    
    if result:
        theta1, theta2 = result
        print(f"2R solution: θ1={math.degrees(theta1):.1f}°, θ2={math.degrees(theta2):.1f}°")
        
        # Verify solution
        x_calc = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
        z_calc = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
        
        print(f"Target: ({r}, {z})")
        print(f"Calculated: ({x_calc:.3f}, {z_calc:.3f})")
        print(f"Error: ({abs(r-x_calc):.6f}, {abs(z-z_calc):.6f})")
    else:
        print("2R solver failed")

def test_ik_accuracy():
    """Test IK solver accuracy."""
    print("\n=== Testing IK Accuracy ===")
    
    config = ArmConfig([3.0, 2.5, 2.0], [(-2.967, 2.967)] * 3)
    
    # Test various targets
    test_targets = [
        np.array([5.0, 0.0, 2.0]),    # Reachable
        np.array([0.0, 4.0, 1.0]),    # Reachable
        np.array([2.0, 2.0, 3.0]),    # Reachable
    ]
    
    for i, target in enumerate(test_targets):
        print(f"\nTarget {i+1}: {target}")
        
        # Analytical IK
        result = solve_ik_analytical(config, target)
        print(f"  Analytical: {result.success}, error: {result.error_distance:.6f}")
        
        if result.success:
            # Verify solution
            positions = forward_kinematics(config, result.state)
            ee_pos = positions[-1]
            actual_error = np.linalg.norm(ee_pos - target)
            print(f"  Actual error: {actual_error:.6f}")
            print(f"  Base angle: {math.degrees(result.state.base_angle):.1f}°")
            print(
                f"  Planar angles: "
                f"{[round(math.degrees(a), 1) for a in result.state.planar_angles]} deg"
            )

def test_singularity_detection():
    """Test singularity detection."""
    print("\n=== Testing Singularity Detection ===")
    
    config = ArmConfig([3.0, 2.0], [(-2.967, 2.967)] * 2)
    
    # Test full extension
    state_ext = ArmState(0.0, [0.0, 0.0])  # Both links straight up
    sing_ext = check_singularity(config, state_ext)
    print(f"Full extension singularity: {sing_ext['near_singularity']}, type: {sing_ext['type']}")
    
    # Test folded configuration
    state_fold = ArmState(0.0, [math.pi, 0.0])  # First link folded back
    sing_fold = check_singularity(config, state_fold)
    print(f"Folded configuration singularity: {sing_fold['near_singularity']}, type: {sing_fold['type']}")

def test_collision_detection():
    """Test self-collision detection."""
    print("\n=== Testing Collision Detection ===")
    
    config = ArmConfig([1.0, 1.0, 1.0], [(-2.967, 2.967)] * 3)
    
    # Test non-colliding configuration
    state_safe = ArmState(0.0, [math.pi/4, math.pi/4, math.pi/4])
    collision_safe = check_self_collision(config, state_safe)
    print(f"Safe configuration collision: {collision_safe}")
    
    # Test potentially colliding configuration
    state_collide = ArmState(0.0, [math.pi, math.pi, 0.0])
    collision_collide = check_self_collision(config, state_collide, margin=0.1)
    print(f"Colliding configuration collision: {collision_collide}")

if __name__ == "__main__":
    test_dh_transform()
    test_forward_kinematics_consistency()
    test_jacobian_correctness()
    test_2r_solver()
    test_ik_accuracy()
    test_singularity_detection()
    test_collision_detection()
    print("\n=== All mathematical correctness tests completed ===")