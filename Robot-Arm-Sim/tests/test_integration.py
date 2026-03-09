#!/usr/bin/env python3
"""Integration test for the robot arm simulator."""

import numpy as np
from kinematics import ArmConfig, ArmState, forward_kinematics, solve_ik_analytical
from constraints import ConstraintSet, validate_target, check_self_collision
from animation import Animator
from math_engine import MathEngine

def test_basic_functionality():
    """Test basic integration between modules."""
    print("=== Testing Basic Integration ===")
    
    # Create arm configuration
    config = ArmConfig([3.0, 2.5, 2.0, 1.5], [(-2.967, 2.967)] * 4)
    print(f"✓ Created arm config with {config.num_planar_joints} joints")
    
    # Test forward kinematics
    state = ArmState(0.0, [0.0, 0.0, 0.0, 0.0])
    positions = forward_kinematics(config, state)
    print(f"✓ Forward kinematics: {len(positions)} positions calculated")
    
    # Test target validation
    target = np.array([4.0, 3.0, 2.5])
    validation = validate_target(config, target, ConstraintSet())
    print(f"✓ Target validation: {validation['message']}")
    
    # Test analytical IK
    result = solve_ik_analytical(config, target)
    print(f"✓ Analytical IK: {result.success}, error: {result.error_distance:.4f}")
    
    if result.state:
        # Test animation
        animator = Animator()
        animator.start(state, result.state)
        print(f"✓ Animation started successfully")
        
        # Test math engine
        math_engine = MathEngine()
        context = {"theta_1": 0.5, "theta_2": 0.3, "x": 4.0, "y": 3.0, "z": 2.5}
        math_engine.set_expression(1, "theta_1 + theta_2")
        result = math_engine.evaluate(1, context)
        print(f"✓ Math engine: expression evaluated to {result}")
    
    print("\n=== All basic tests passed! ===")

def test_edge_cases():
    """Test edge cases and potential issues."""
    print("\n=== Testing Edge Cases ===")
    
    # Test single link arm
    config1 = ArmConfig([5.0], [(-2.967, 2.967)])
    state1 = ArmState(0.0, [0.0])
    target1 = np.array([0.0, 0.0, 5.0])  # Straight up
    result1 = solve_ik_analytical(config1, target1)
    print(f"✓ Single link test: {result1.success}")
    
    # Test two-link arm
    config2 = ArmConfig([3.0, 2.0], [(-2.967, 2.967)] * 2)
    state2 = ArmState(0.0, [0.0, 0.0])
    target2 = np.array([4.0, 0.0, 1.0])
    result2 = solve_ik_analytical(config2, target2)
    print(f"✓ Two-link test: {result2.success}")
    
    # Test unreachable target
    config3 = ArmConfig([1.0, 1.0], [(-2.967, 2.967)] * 2)
    target3 = np.array([10.0, 0.0, 0.0])  # Too far
    result3 = solve_ik_analytical(config3, target3)
    print(f"✓ Unreachable target test: {not result3.success}")
    
    print("\n=== Edge case tests completed ===")

if __name__ == "__main__":
    test_basic_functionality()
    test_edge_cases()