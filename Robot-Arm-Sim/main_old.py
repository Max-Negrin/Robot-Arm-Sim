"""
3D Robotic Arm Simulator with CCD Inverse Kinematics
Features:
- Full 3D kinematics and visualization
- Cyclic Coordinate Descent (CCD) algorithm
- Smooth animation with speed limits
- Self-collision detection
- Joint limits (-170, 170 degrees)
- Integrated GUI with input controls
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import List, Tuple
import math
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# ============================================================================
# ANIMATION PARAMETERS
# ============================================================================
max_rotation_speed = 1.5      # degrees per frame
fps = 50                  # frames per second

# Joint safety limits
joint_limit_min = -170.0      # degrees
joint_limit_max = 170.0       # degrees

# Self-collision margin (distance threshold)
collision_margin = 0.25


# ============================================================================
# ROTATION UTILITIES
# ============================================================================
def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    """Rotation matrix around X-axis."""
    angle_rad = math.radians(angle_deg)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """Rotation matrix around Y-axis."""
    angle_rad = math.radians(angle_deg)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """Rotation matrix around Z-axis."""
    angle_rad = math.radians(angle_deg)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to rotation matrix (ZYX convention)."""
    rz = rotation_matrix_z(yaw)
    ry = rotation_matrix_y(pitch)
    rx = rotation_matrix_x(roll)
    return rz @ ry @ rx

def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in radians."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1, 1)
    return math.acos(cos_angle)


# ============================================================================
# 3D ROBOTIC ARM CLASS
# ============================================================================
class PlanarArm3D:
    """Represents a 3D robotic arm with CCD inverse kinematics."""
    
    def __init__(self, link_lengths: List[float]):
        """
        Initialize a planar robotic arm (like an excavator).
        
        All joints operate in the same vertical plane.
        
        Args:
            link_lengths: List of link lengths from base to end-effector
        """
        self.link_lengths = link_lengths
        self.num_joints = len(link_lengths)
        
        # Store joint angles as pitch values (rotation in the vertical plane)
        # Each joint rotates around the Y-axis (pitch) within the XZ plane
        self.joint_angles = [0.0 for _ in range(self.num_joints)]
        
        # Base rotation angle (rotation around Z-axis)
        # This controls the horizontal direction the arm faces
        self.base_rotation = 0.0
        
        # Frame orientations for each joint
        self.frames = []
    
    def forward_kinematics(self) -> List[np.ndarray]:
        """
        Calculate joint positions using forward kinematics.
        Works in the XZ vertical plane, then applies base rotation.
        
        Returns:
            List of 3D positions for each joint plus end-effector
        """
        # Positions in the vertical XZ plane (Y=0 initially)
        positions_planar = [np.array([0.0, 0.0])]  # Start at origin in XZ plane
        
        # Cumulative angle in the vertical plane
        cumulative_angle = 0.0
        
        # Calculate positions in the vertical plane
        for i, length in enumerate(self.link_lengths):
            pitch = self.joint_angles[i]
            cumulative_angle += pitch
            
            # Convert cumulative angle to radians
            angle_rad = math.radians(cumulative_angle)
            
            # Link direction in the vertical plane (rotate from Z-axis)
            # Z points up, X points forward, so we rotate from Z toward X
            link_x = length * math.sin(angle_rad)
            link_z = length * math.cos(angle_rad)
            
            # Add to previous position
            new_pos_planar = positions_planar[-1] + np.array([link_x, link_z])
            positions_planar.append(new_pos_planar)
        
        # Apply base rotation and convert to 3D
        positions_3d = []
        for pos_planar in positions_planar:
            # Rotate around Z-axis (base rotation)
            base_angle_rad = math.radians(self.base_rotation)
            x = pos_planar[0] * math.cos(base_angle_rad) - 0 * math.sin(base_angle_rad)
            y = pos_planar[0] * math.sin(base_angle_rad) + 0 * math.cos(base_angle_rad)
            z = pos_planar[1]
            
            positions_3d.append(np.array([x, y, z]))
        
        return positions_3d
    
    def get_end_effector_pos(self) -> np.ndarray:
        """Get the position of the end-effector."""
        positions = self.forward_kinematics()
        return positions[-1]
    
    def distance_to_target(self, target: Tuple[float, float, float]) -> float:
        """Calculate distance from end-effector to target."""
        ee_pos = self.get_end_effector_pos()
        target_arr = np.array(target)
        return float(np.linalg.norm(ee_pos - target_arr))
    
    def check_self_collision(self) -> bool:
        """
        Check if any link intersects with another link or the base.
        
        Returns:
            True if collision detected, False otherwise
        """
        positions = self.forward_kinematics()
        
        # Check link-to-link collisions
        for i in range(len(positions) - 1):
            for j in range(i+2, len(positions) - 1):  # Skip adjacent links
                if self._segments_overlap_3d(
                    positions[i], positions[i+1],
                    positions[j], positions[j+1],
                    collision_margin
                ):
                    return True
        
        return False
    
    @staticmethod
    def _segments_overlap_3d(p1: np.ndarray, p2: np.ndarray,
                            p3: np.ndarray, p4: np.ndarray,
                            threshold: float = 0.25) -> bool:
        """
        Check if two 3D line segments are too close to each other.
        
        Args:
            p1, p2: First segment endpoints
            p3, p4: Second segment endpoints
            threshold: Minimum allowed distance
        """
        def point_to_segment_dist_3d(p, a, b):
            """Calculate minimum distance from point p to segment ab in 3D."""
            ab = b - a
            ab_sq = np.dot(ab, ab)
            
            if ab_sq < 1e-6:
                return float(np.linalg.norm(p - a))
            
            t = max(0, min(1, np.dot(p - a, ab) / ab_sq))
            closest = a + t * ab
            
            return float(np.linalg.norm(p - closest))
        
        d1 = point_to_segment_dist_3d(p1, p3, p4)
        d2 = point_to_segment_dist_3d(p2, p3, p4)
        d3 = point_to_segment_dist_3d(p3, p1, p2)
        d4 = point_to_segment_dist_3d(p4, p1, p2)
        
        min_dist = min(d1, d2, d3, d4)
        return min_dist < threshold
    
    def ccd_step(self, target: Tuple[float, float, float],
                 max_iterations: int = 5) -> bool:
        """
        Perform one step of Cyclic Coordinate Descent (planar).
        
        Args:
            target: Target position (x, y, z)
            max_iterations: Max iterations per step
            
        Returns:
            True if target is reachable, False if out of reach
        """
        target_arr = np.array(target)
        
        for iteration in range(max_iterations):
            positions = self.forward_kinematics()
            ee_pos = positions[-1]
            
            # Iterate from last joint to first
            for joint_idx in range(self.num_joints - 1, -1, -1):
                joint_pos = positions[joint_idx]
                
                # Vector from joint to end-effector (in XZ plane)
                ee_vec_xz = ee_pos[[0, 2]] - joint_pos[[0, 2]]
                
                # Vector from joint to target (in XZ plane)
                target_vec_xz = target_arr[[0, 2]] - joint_pos[[0, 2]]
                
                # Calculate angles in the vertical plane
                if np.linalg.norm(ee_vec_xz) < 1e-6 or np.linalg.norm(target_vec_xz) < 1e-6:
                    continue
                
                # Angle from joint to end-effector
                ee_angle = math.atan2(ee_vec_xz[0], ee_vec_xz[1])
                
                # Angle from joint to target
                target_angle = math.atan2(target_vec_xz[0], target_vec_xz[1])
                
                # Required rotation
                rotation_angle = target_angle - ee_angle
                
                # Normalize to [-pi, pi]
                while rotation_angle > math.pi:
                    rotation_angle -= 2 * math.pi
                while rotation_angle < -math.pi:
                    rotation_angle += 2 * math.pi
                
                # Limit rotation per step
                max_rotation_rad = math.radians(2.0)  # Max 2 degrees per joint per iteration
                rotation_angle = max(-max_rotation_rad, min(max_rotation_rad, rotation_angle))
                
                # Update joint angle
                old_angle = self.joint_angles[joint_idx]
                self.joint_angles[joint_idx] += math.degrees(rotation_angle)
                self.joint_angles[joint_idx] = max(joint_limit_min, min(joint_limit_max, self.joint_angles[joint_idx]))
                
                # Check for collisions
                if self.check_self_collision():
                    self.joint_angles[joint_idx] = old_angle
                
                positions = self.forward_kinematics()
                ee_pos = positions[-1]
        
        # Check if reachable
        max_reach = sum(self.link_lengths)
        target_dist_xz = math.sqrt(target_arr[0]**2 + target_arr[2]**2)
        
        return target_dist_xz <= max_reach
    
    def move_towards_ik_solution(self, target: Tuple[float, float, float],
                                 max_rotation_speed: float) -> None:
        """
        Smoothly move joints and base toward the IK solution.
        
        Args:
            target: Target position
            max_rotation_speed: Max rotation in degrees per frame
        """
        old_angles = [angle for angle in self.joint_angles]
        old_base_rotation = self.base_rotation
        
        # Compute one CCD step
        self.ccd_step(target, max_iterations=2)
        
        # Calculate target base angle from target position (XY plane)
        target_base_angle = math.degrees(math.atan2(target[1], target[0]))
        
        # Smoothly interpolate joint angles
        for i in range(self.num_joints):
            old_angle = old_angles[i]
            new_angle = self.joint_angles[i]
            
            # Calculate angle difference (shortest path)
            diff = self._shortest_angle_diff(old_angle, new_angle)
            
            # Limit speed
            diff = max(-max_rotation_speed, min(max_rotation_speed, diff))
            
            # Update angle
            final_angle = old_angle + diff
            final_angle = max(joint_limit_min, min(joint_limit_max, final_angle))
            
            self.joint_angles[i] = final_angle
        
        # Smoothly interpolate base rotation
        base_diff = self._shortest_angle_diff(old_base_rotation, target_base_angle)
        base_diff = max(-max_rotation_speed, min(max_rotation_speed, base_diff))
        self.base_rotation = old_base_rotation + base_diff
    
    @staticmethod
    def _shortest_angle_diff(angle1: float, angle2: float) -> float:
        """Calculate shortest angular difference."""
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff


# ============================================================================
# 3D VISUALIZER WITH INTEGRATED GUI
# ============================================================================
class Arm3DVisualizer:
    """Handles 3D visualization with integrated GUI controls."""
    
    def __init__(self, root: tk.Tk):
        """Initialize visualizer with main window."""
        self.root = root
        self.root.title("3D Robotic Arm Simulator")
        self.root.geometry("1400x900")
        
        self.arm = None
        self.target_pos = (4.0, 3.0, 2.5)
        self.frame_count = 0
        self.anim = None
        self.fig = None
        self.ax = None
        self.canvas = None
        
        # View sensitivity for mouse control (degrees per pixel)
        self.view_sensitivity = 5.0
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_frame, text="Configuration", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        
        # Number of links
        ttk.Label(left_panel, text="Number of Links:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))
        num_links_frame = ttk.Frame(left_panel)
        num_links_frame.pack(fill=tk.X, pady=5)
        self.num_links_var = tk.IntVar(value=4)
        ttk.Spinbox(num_links_frame, from_=1, to=10, textvariable=self.num_links_var, 
                   command=self.update_link_count, width=10).pack(side=tk.LEFT, padx=5)
        
        # Link lengths
        ttk.Label(left_panel, text="Link Lengths:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 5))
        
        # Scrollable frame for link inputs
        canvas_frame = tk.Canvas(left_panel, height=150, bg="white", highlightthickness=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=canvas_frame.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_frame.configure(yscrollcommand=scrollbar.set)
        
        self.links_frame = ttk.Frame(canvas_frame)
        canvas_frame.create_window((0, 0), window=self.links_frame, anchor="nw")
        
        # Target position
        ttk.Label(left_panel, text="Target Position (X, Y, Z):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 5))
        
        target_frame = ttk.Frame(left_panel)
        target_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_frame, text="X:").pack(side=tk.LEFT, padx=5)
        self.target_x = tk.DoubleVar(value=4.0)
        ttk.Entry(target_frame, textvariable=self.target_x, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(target_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        self.target_y = tk.DoubleVar(value=3.0)
        ttk.Entry(target_frame, textvariable=self.target_y, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(target_frame, text="Z:").pack(side=tk.LEFT, padx=5)
        self.target_z = tk.DoubleVar(value=2.5)
        ttk.Entry(target_frame, textvariable=self.target_z, width=8).pack(side=tk.LEFT, padx=5)
        
        # Apply button
        ttk.Button(left_panel, text="Apply & Start", command=self.apply_config).pack(fill=tk.X, pady=10)
        
        # Relative Angles Display
        ttk.LabelFrame(left_panel, text="Relative Angles (vs Z-axis)", padding=5).pack(fill=tk.X, pady=10)
        self.relative_angles_text = tk.Text(left_panel, height=6, width=30, state=tk.DISABLED)
        self.relative_angles_text.pack(fill=tk.X, pady=5)
        
        # Angle Changes Display
        ttk.LabelFrame(left_panel, text="Angle Changes (Δ)", padding=5).pack(fill=tk.X, pady=10)
        self.angle_changes_text = tk.Text(left_panel, height=6, width=30, state=tk.DISABLED)
        self.angle_changes_text.pack(fill=tk.X, pady=5)
        
        # Stats
        ttk.Label(left_panel, text="Statistics:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 5))
        self.stats_text = tk.Text(left_panel, height=15, width=30, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Right panel - Plot and controls
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Plot frame (will expand to fill available space)
        self.plot_frame = ttk.Frame(right_panel)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sensitivity control at bottom
        sens_control_frame = ttk.LabelFrame(right_panel, text="View Sensitivity", padding=5)
        sens_control_frame.pack(fill=tk.X, pady=5)
        
        self.sensitivity_var = tk.DoubleVar(value=5.0)
        sens_slider = ttk.Scale(sens_control_frame, from_=1, to=25, orient=tk.HORIZONTAL, 
                                variable=self.sensitivity_var, command=self._update_sensitivity)
        sens_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.sensitivity_label = ttk.Label(sens_control_frame, text="5.0x", width=5)
        self.sensitivity_label.pack(side=tk.LEFT, padx=5)
        
        # Initialize with default config
        self.update_link_count()
    
    def update_link_count(self):
        """Update the number of link input fields."""
        num_links = self.num_links_var.get()
        
        # Clear existing widgets
        for widget in self.links_frame.winfo_children():
            widget.destroy()
        
        # Create new input fields
        self.link_vars = []
        for i in range(num_links):
            frame = ttk.Frame(self.links_frame)
            frame.pack(fill=tk.X, pady=3, padx=5)
            
            ttk.Label(frame, text=f"Link {i}:").pack(side=tk.LEFT, padx=5)
            
            var = tk.DoubleVar(value=3.0 - i * 0.5)
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.pack(side=tk.LEFT, padx=5)
            
            self.link_vars.append(var)
    
    def apply_config(self):
        """Apply configuration and update target (without full restart)."""
        try:
            # Get link lengths and target
            link_lengths = [var.get() for var in self.link_vars]
            
            if any(x <= 0 for x in link_lengths):
                messagebox.showerror("Error", "All link lengths must be positive!")
                return
            
            # Get target position
            target_pos = (self.target_x.get(), self.target_y.get(), self.target_z.get())
            
            # If this is the first time or link count changed, create new arm and visualization
            if self.arm is None or len(self.arm.link_lengths) != len(link_lengths):
                self.arm = PlanarArm3D(link_lengths)
                self.create_visualization()
            else:
                # Just update link lengths and reset the "Before" values to current state
                self.arm.link_lengths = link_lengths
                self.frame_start_angles = [ang for ang in self.arm.joint_angles]
                self.frame_start_base = self.arm.base_rotation
            
            # Update target (animation will smoothly move toward it)
            self.target_pos = target_pos
            
        except ValueError:
            messagebox.showerror("Error", "Invalid input values!")
    
    def create_visualization(self):
        """Create the 3D visualization."""
        # Clean up old figure
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create new figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        max_reach = sum(self.arm.link_lengths)
        self.ax.set_xlim(-2, max_reach + 2)
        self.ax.set_ylim(-2, max_reach + 2)
        self.ax.set_zlim(-2, max_reach + 2)
        
        self.ax.set_xlabel('X (units)', fontsize=10)
        self.ax.set_ylabel('Y (units)', fontsize=10)
        self.ax.set_zlabel('Z (units)', fontsize=10)
        self.ax.set_title('3D Robotic Arm - CCD Inverse Kinematics', fontsize=12, fontweight='bold')
        
        # Initialize view angles
        self.view_elev = 20
        self.view_azim = 45
        self.ax.view_init(elev=self.view_elev, azim=self.view_azim)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Track previous angles for delta calculation
        self.previous_joint_angles = [0.0 for _ in range(self.arm.num_joints)]
        self.previous_base_rotation = 0.0
        
        # Store the true initial angles (0 degrees = vertical)
        self.initial_joint_angles = [0.0 for _ in range(self.arm.num_joints)]
        self.initial_base_rotation = 0.0
        self.frame_start_angles = [0.0 for _ in range(self.arm.num_joints)]
        self.frame_start_base = 0.0
        
        # Mouse tracking for view rotation
        self.mouse_press_pos = None
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        
        # Start animation
        self.frame_count = 0
        self.anim = animation.FuncAnimation(
            self.fig, self.update_plot, frames=600,
            interval=1000/fps, repeat=True, repeat_delay=2000, blit=False
        )
    
    def _on_mouse_press(self, event):
        """Handle mouse button press."""
        if event.inaxes == self.ax:
            self.mouse_press_pos = (event.xdata, event.ydata)
    
    def _update_sensitivity(self, value):
        """Update view sensitivity and label when slider changes."""
        self.view_sensitivity = float(value)
        self.sensitivity_label.config(text=f"{self.view_sensitivity:.1f}x")
    
    def _on_mouse_motion(self, event):
        """Handle mouse motion to rotate view."""
        if self.mouse_press_pos is None or event.inaxes != self.ax:
            return
        
        # Calculate mouse movement
        dx = event.xdata - self.mouse_press_pos[0] if event.xdata else 0
        dy = event.ydata - self.mouse_press_pos[1] if event.ydata else 0
        
        # Update view angles with adjustable sensitivity (inverted for natural controls)
        self.view_azim -= dx * self.view_sensitivity
        self.view_elev -= dy * self.view_sensitivity
        
        # Clamp elevation between -90 and 90 degrees
        self.view_elev = max(-90, min(90, self.view_elev))
        
        # Update axes view
        self.ax.view_init(elev=self.view_elev, azim=self.view_azim)
        
        # Redraw
        self.canvas.draw_idle()
    
    def _on_mouse_release(self, event):
        """Handle mouse button release."""
        self.mouse_press_pos = None
    
    def update_plot(self, frame):
        """Update the plot."""
        if self.arm is None:
            return
        
        self.frame_count += 1
        
        # Move arm toward target
        self.arm.move_towards_ik_solution(self.target_pos, max_rotation_speed)
        
        # Get positions
        positions = self.arm.forward_kinematics()
        positions_arr = np.array(positions)
        
        # Clear axes
        self.ax.clear()
        
        # Restore axis limits
        max_reach = sum(self.arm.link_lengths)
        self.ax.set_xlim(-2, max_reach + 2)
        self.ax.set_ylim(-2, max_reach + 2)
        self.ax.set_zlim(-2, max_reach + 2)
        
        self.ax.set_xlabel('X (units)', fontsize=10)
        self.ax.set_ylabel('Y (units)', fontsize=10)
        self.ax.set_zlabel('Z (units)', fontsize=10)
        self.ax.set_title('3D Robotic Arm - CCD Inverse Kinematics', fontsize=12, fontweight='bold')
        self.ax.view_init(elev=self.view_elev, azim=self.view_azim)
        
        # Draw the arm links
        if len(positions_arr) > 1:
            self.ax.plot(positions_arr[:, 0], positions_arr[:, 1], positions_arr[:, 2], 
                        'b-', linewidth=3, label='Arm Links')
        
        # Update joints
        if len(positions_arr) > 1:
            joint_positions = positions_arr[:-1]
            self.ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2],
                           c='blue', s=100, label='Joints')
        
        # Update end-effector
        ee_pos = positions[-1]
        self.ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], c='lime', s=150, marker='D',
                       label='End-Effector')
        
        # Target
        self.ax.scatter([self.target_pos[0]], [self.target_pos[1]], [self.target_pos[2]], 
                       c='red', s=200, marker='*', label='Target')
        
        # Base
        self.ax.scatter([0], [0], [0], c='black', s=200, marker='o', label='Base')
        
        self.ax.legend(loc='upper right', fontsize=9)
        
        # Calculate angle deltas
        angle_deltas = []
        for i in range(self.arm.num_joints):
            old_angle = self.previous_joint_angles[i]
            new_angle = self.arm.joint_angles[i]
            
            # Calculate magnitude of change
            delta_magnitude = abs(new_angle - old_angle)
            angle_deltas.append(delta_magnitude)
        
        # Store current as previous for next frame
        self.previous_joint_angles = [angle for angle in self.arm.joint_angles]
        self.previous_base_rotation = self.arm.base_rotation
        
        # Calculate relative angles (angles relative to upward Z direction)
        relative_angles = []
        for i in range(len(positions_arr) - 1):
            pos = positions_arr[i]
            next_pos = positions_arr[i + 1]
            
            # Vector from joint to next joint
            link_vec = next_pos - pos
            
            # Angle relative to Z-axis (upward)
            z_vec = np.array([0, 0, 1])
            if np.linalg.norm(link_vec) > 1e-6:
                cos_angle = np.dot(link_vec, z_vec) / np.linalg.norm(link_vec)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_from_z = math.degrees(math.acos(cos_angle))
            else:
                angle_from_z = 0.0
            
            relative_angles.append(angle_from_z)
        
        # Update stats display
        distance = self.arm.distance_to_target(self.target_pos)
        max_reach = sum(self.arm.link_lengths)
        target_dist = math.sqrt(self.target_pos[0]**2 + self.target_pos[1]**2 + self.target_pos[2]**2)
        
        # Update Relative Angles display (with table header)
        rel_angles_text = f"{'Link':<8} {'Angle from Z':<15}\n"
        rel_angles_text += f"{'-' * 25}\n"
        for i, angle in enumerate(relative_angles):
            rel_angles_text += f"Link {i:<3} {angle:>12.2f}°\n"
        
        self.relative_angles_text.config(state=tk.NORMAL)
        self.relative_angles_text.delete(1.0, tk.END)
        self.relative_angles_text.insert(tk.END, rel_angles_text if rel_angles_text else "No data")
        self.relative_angles_text.config(state=tk.DISABLED)
        
        # Update Angle Changes display with Before/Current/Change table
        changes_text = f"{'Item':<8} {'Before':<10} {'Current':<10} {'Change':<10}\n"
        changes_text += f"{'-' * 40}\n"
        
        # Base rotation first
        before_base = self.frame_start_base
        current_base = self.arm.base_rotation
        change_base = current_base - before_base
        # Normalize to shortest path
        while change_base > 180:
            change_base -= 360
        while change_base < -180:
            change_base += 360
        changes_text += f"Base  {before_base:>9.1f}° {current_base:>9.1f}° {change_base:>9.1f}°\n"
        
        # Then joint angles
        for i in range(self.arm.num_joints):
            before = self.frame_start_angles[i]
            current = self.arm.joint_angles[i]
            change = current - before
            changes_text += f"J{i:<7} {before:>9.1f}° {current:>9.1f}° {change:>9.1f}°\n"
        
        self.angle_changes_text.config(state=tk.NORMAL)
        self.angle_changes_text.delete(1.0, tk.END)
        self.angle_changes_text.insert(tk.END, changes_text if changes_text else "No data")
        self.angle_changes_text.config(state=tk.DISABLED)
        
        # Update Stats display
        stats = f"Frame: {self.frame_count}\n"
        stats += f"EE Distance: {distance:.4f}\n"
        stats += f"Max Reach: {max_reach:.2f}\n"
        stats += f"Target Dist: {target_dist:.2f}\n"
        stats += f"Reachable: {'Yes' if target_dist <= max_reach else 'No'}\n"
        stats += f"Collision: {'Yes' if self.arm.check_self_collision() else 'No'}\n\n"
        
        stats += f"=== Base Rotation ===\n"
        stats += f"Base: {self.arm.base_rotation:7.1f}°\n\n"
        
        stats += "=== Joint Angles (Planar) ===\n"
        for i, angle in enumerate(self.arm.joint_angles):
            stats += f"Joint {i}: {angle:7.1f}°\n"
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)
        
        # Redraw canvas
        self.canvas.draw()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    visualizer = Arm3DVisualizer(root)
    root.mainloop()
