"""OpenGL 3D viewport for the arm."""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ═══════════════════════════════════════════════════════════════════════════
# 3D Viewport
# ═══════════════════════════════════════════════════════════════════════════

class ArmViewport(gl.GLViewWidget):
    """OpenGL 3D viewport for rendering the robotic arm."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.opts['fov'] = 1  # Near-orthographic projection (no fish-eye)
        self.setCameraPosition(distance=200, elevation=20, azimuth=45)
        self.setBackgroundColor(30, 30, 36, 255)
        
        # Ground grid
        grid = gl.GLGridItem()
        grid.setSize(20, 20, 1)
        grid.setSpacing(1, 1, 1)
        grid.setColor((80, 80, 80, 100))
        self.addItem(grid)

        # Coordinate axes (RGB = XYZ)
        for axis_data, color in [
            (np.array([[0, 0, 0], [3, 0, 0]]), (255, 60, 60, 255)),
            (np.array([[0, 0, 0], [0, 3, 0]]), (60, 255, 60, 255)),
            (np.array([[0, 0, 0], [0, 0, 3]]), (60, 100, 255, 255)),
        ]:
            axis_line = gl.GLLinePlotItem(
                pos=axis_data.astype(np.float32),
                color=pg.mkColor(*color), width=2, antialias=True,
            )
            self.addItem(axis_line)
        
        # Arm links
        self.link_plot = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=pg.mkColor(40, 130, 255, 255), width=4, antialias=True,
        )
        self.addItem(self.link_plot)

        # Joint markers
        self.joint_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.2, 0.5, 1.0, 1.0), size=10,
        )
        self.addItem(self.joint_scatter)

        # End-effector marker
        self.ee_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.2, 1.0, 0.3, 1.0), size=14,
        )
        self.addItem(self.ee_scatter)

        # Target marker
        self.target_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.2, 0.2, 1.0), size=16,
        )
        self.addItem(self.target_scatter)

        # Base marker
        self.base_scatter = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]], dtype=np.float32),
            color=(0.9, 0.9, 0.9, 1.0), size=12,
        )
        self.addItem(self.base_scatter)

    def wheelEvent(self, event):
        """Zoom by scaling camera distance — boosted for near-orthographic (low FOV) mode."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # Scale factor per notch: larger when FOV is small so zoom feels consistent
        fov = self.opts.get('fov', 60)
        factor = 1.0 + (abs(delta) / 120.0) * max(0.1, fov / 60.0) * 0.15
        if delta > 0:
            self.opts['distance'] /= factor
        else:
            self.opts['distance'] *= factor
        self.opts['distance'] = max(1.0, self.opts['distance'])
        self.update()

    def update_arm(self, positions: list, target: np.ndarray) -> None:
        """Update all rendering data without recreating plot items."""
        if len(positions) < 2:
            return

        pos_arr = np.array(positions, dtype=np.float32)
        self.link_plot.setData(pos=pos_arr)
        # FK returns interleaved [eff0, nom0, eff1, nom1, …, nomN-1] (2N elements).
        # Joint dots at eff positions (even indices).
        self.joint_scatter.setData(pos=pos_arr[::2])
        self.ee_scatter.setData(pos=pos_arr[-1:])
        self.target_scatter.setData(pos=np.array([target], dtype=np.float32))

