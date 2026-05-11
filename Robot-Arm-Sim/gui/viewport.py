"""OpenGL 3D viewport for the arm."""
import os
import math
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

try:
    import trimesh as _trimesh
    _MESH_SUPPORT = True
except ImportError:
    _MESH_SUPPORT = False


def _load_mesh_file(path: str):
    """Load a mesh file. Returns (vertexes, faces) float32/int32 arrays or (None, None)."""
    if not _MESH_SUPPORT:
        return None, None
    try:
        obj = _trimesh.load(path, force='mesh')
        if hasattr(obj, 'vertices') and hasattr(obj, 'faces'):
            return np.array(obj.vertices, dtype=np.float32), np.array(obj.faces, dtype=np.int32)
        if hasattr(obj, 'geometry'):
            meshes = list(obj.geometry.values())
            if meshes:
                merged = _trimesh.util.concatenate(meshes)
                return np.array(merged.vertices, dtype=np.float32), np.array(merged.faces, dtype=np.int32)
    except Exception:
        pass
    return None, None


def _build_euler_rotation(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Build 4×4 rotation matrix from XYZ Euler angles (degrees). Order: Rz·Ry·Rx."""
    rx = math.radians(rx_deg); ry = math.radians(ry_deg); rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0,   0,  0],
                   [0, cx, -sx, 0],
                   [0, sx,  cx, 0],
                   [0, 0,   0,  1]], dtype=np.float64)
    Ry = np.array([[ cy, 0, sy, 0],
                   [  0, 1,  0, 0],
                   [-sy, 0, cy, 0],
                   [  0, 0,  0, 1]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0, 0],
                   [sz,  cz, 0, 0],
                   [ 0,   0, 1, 0],
                   [ 0,   0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _apply_transform(item, T: np.ndarray, scale: float = 1.0) -> None:
    """Apply a 4×4 world transform (and uniform scale) to a pyqtgraph GL item."""
    item.resetTransform()
    if abs(scale - 1.0) > 1e-9:
        item.scale(scale, scale, scale)
    R = T[:3, :3]
    tx, ty, tz = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
    # Axis-angle decomposition of rotation matrix
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    if angle_deg > 0.01:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        nrm = float(np.linalg.norm(axis))
        if nrm > 1e-9:
            axis /= nrm
            item.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
    item.translate(tx, ty, tz)


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

        # Arm links (skeleton lines — always visible as reference)
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

        # Mesh items — one per arm link, populated by load_meshes()
        self._mesh_items: list = []
        # Per-mesh local offset transforms (applied in joint frame before FK)
        self._mesh_local_transforms: list = []
        # Per-mesh hidden flags (True = don't render)
        self._mesh_hidden: list = []

    # ── Mesh management ────────────────────────────────────────────────────

    def load_meshes(self, paths: list, scale: float = 1.0) -> None:
        """Load/reload GLMeshItem objects for each arm link.

        paths: one file path per arm link ('' or None = no mesh for that link).
        scale: uniform scale applied to all meshes (e.g. 0.1 to convert mm→cm).
        """
        for item in self._mesh_items:
            if item is not None:
                self.removeItem(item)
        self._mesh_items = []

        # Preserve existing per-mesh local transforms; fill new slots with defaults.
        _default_lt = {'tx': 0.0, 'ty': 0.0, 'tz': 0.0,
                       'rx': 0.0, 'ry': 0.0, 'rz': 0.0, 'scale': 1.0}
        prev_lts = list(self._mesh_local_transforms)
        prev_hidden = list(self._mesh_hidden)
        self._mesh_local_transforms = []
        self._mesh_hidden = []
        for i in range(len(paths)):
            self._mesh_local_transforms.append(
                dict(prev_lts[i]) if i < len(prev_lts) else dict(_default_lt)
            )
            self._mesh_hidden.append(prev_hidden[i] if i < len(prev_hidden) else False)

        if not _MESH_SUPPORT:
            return

        for i, path in enumerate(paths):
            hidden = self._mesh_hidden[i]
            if path and os.path.exists(path):
                verts, faces = _load_mesh_file(path)
                if verts is not None:
                    if scale != 1.0:
                        verts = verts * np.float32(scale)
                    mesh_data = gl.MeshData(vertexes=verts, faces=faces)
                    item = gl.GLMeshItem(
                        meshdata=mesh_data,
                        smooth=True,
                        color=(0.55, 0.60, 0.70, 0.85),
                        glOptions='translucent',
                        drawEdges=False,
                    )
                    self.addItem(item)
                    item.setVisible(not hidden)
                    self._mesh_items.append(item)
                else:
                    self._mesh_items.append(None)
            else:
                self._mesh_items.append(None)

        # Dim the skeleton lines only when at least one mesh is loaded AND visible
        any_visible = any(
            m is not None and not h
            for m, h in zip(self._mesh_items, self._mesh_hidden)
        )
        alpha = 60 if any_visible else 255
        self.link_plot.setData(color=pg.mkColor(40, 130, 255, alpha))

    def reload_mesh(self, idx: int, path: str, scale: float = 1.0) -> None:
        """Reload a single mesh item at index idx without touching the others."""
        # Ensure lists are long enough
        while len(self._mesh_items) <= idx:
            self._mesh_items.append(None)
        _default_lt = {'tx': 0.0, 'ty': 0.0, 'tz': 0.0,
                       'rx': 0.0, 'ry': 0.0, 'rz': 0.0, 'scale': 1.0}
        while len(self._mesh_local_transforms) <= idx:
            self._mesh_local_transforms.append(dict(_default_lt))
        while len(self._mesh_hidden) <= idx:
            self._mesh_hidden.append(False)

        # Remove the old item for this slot
        old = self._mesh_items[idx]
        if old is not None:
            self.removeItem(old)
            self._mesh_items[idx] = None

        hidden = self._mesh_hidden[idx]
        if path and os.path.exists(path) and _MESH_SUPPORT:
            verts, faces = _load_mesh_file(path)
            if verts is not None:
                if scale != 1.0:
                    verts = verts * np.float32(scale)
                mesh_data = gl.MeshData(vertexes=verts, faces=faces)
                item = gl.GLMeshItem(
                    meshdata=mesh_data,
                    smooth=True,
                    color=(0.55, 0.60, 0.70, 0.85),
                    glOptions='translucent',
                    drawEdges=False,
                )
                self.addItem(item)
                item.setVisible(not hidden)
                self._mesh_items[idx] = item

        # Update skeleton line alpha
        any_visible = any(
            m is not None and not h
            for m, h in zip(self._mesh_items, self._mesh_hidden)
        )
        self.link_plot.setData(color=pg.mkColor(40, 130, 255, 60 if any_visible else 255))

    def set_mesh_local_transform(self, idx: int, tx: float = 0.0, ty: float = 0.0,
                                  tz: float = 0.0, rx_deg: float = 0.0,
                                  ry_deg: float = 0.0, rz_deg: float = 0.0,
                                  scale: float = 1.0) -> None:
        """Set per-mesh local offset transform (applied in joint frame before FK)."""
        _default = {'tx': 0.0, 'ty': 0.0, 'tz': 0.0,
                    'rx': 0.0, 'ry': 0.0, 'rz': 0.0, 'scale': 1.0}
        while len(self._mesh_local_transforms) <= idx:
            self._mesh_local_transforms.append(dict(_default))
        self._mesh_local_transforms[idx] = {
            'tx': tx, 'ty': ty, 'tz': tz,
            'rx': rx_deg, 'ry': ry_deg, 'rz': rz_deg,
            'scale': scale,
        }

    def set_mesh_hidden(self, idx: int, hidden: bool) -> None:
        """Show or hide the mesh for link idx without reloading it."""
        while len(self._mesh_hidden) <= idx:
            self._mesh_hidden.append(False)
        self._mesh_hidden[idx] = hidden
        if idx < len(self._mesh_items) and self._mesh_items[idx] is not None:
            self._mesh_items[idx].setVisible(not hidden)
        any_visible = any(
            m is not None and not h
            for m, h in zip(self._mesh_items, self._mesh_hidden)
        )
        self.link_plot.setData(color=pg.mkColor(40, 130, 255, 60 if any_visible else 255))

    # ── Rendering ──────────────────────────────────────────────────────────

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

    def update_arm(self, positions: list, target: np.ndarray,
                   frames: list | None = None) -> None:
        """Update all rendering data without recreating plot items.

        positions: list of N+1 world-space 3D points from forward_kinematics.
        frames:    optional list of 4×4 DH transform matrices from
                   forward_kinematics_frames; used to position mesh items.
        """
        if len(positions) < 2:
            return

        pos_arr = np.array(positions, dtype=np.float32)
        self.link_plot.setData(pos=pos_arr)
        self.joint_scatter.setData(pos=pos_arr[:-1])
        self.ee_scatter.setData(pos=pos_arr[-1:])
        self.target_scatter.setData(pos=np.array([target], dtype=np.float32))

        # Apply FK frames to mesh items.
        # Convention: STL 0,0,0 sits at the proximal joint (base of the link).
        # frames[i+1] = proximal joint position; frames[i+2] = distal orientation (includes joint rotation).
        # We build a frame with proximal POSITION + distal ORIENTATION so the mesh pivots correctly.
        if frames and self._mesh_items:
            for i, item in enumerate(self._mesh_items):
                if i < len(self._mesh_hidden) and self._mesh_hidden[i]:
                    continue  # hidden — skip transform math entirely
                prox_idx = i + 1   # proximal joint
                dist_idx = i + 2   # distal joint (carries the rotation for this link)
                if item is not None and dist_idx < len(frames):
                    # Combine: orientation from dist frame, position from prox frame
                    T_fk = frames[dist_idx].copy()
                    T_fk[:3, 3] = frames[prox_idx][:3, 3]
                    if i < len(self._mesh_local_transforms):
                        lt = self._mesh_local_transforms[i]
                        T_local = _build_euler_rotation(lt['rx'], lt['ry'], lt['rz'])
                        T_local[:3, 3] = [lt['tx'], lt['ty'], lt['tz']]
                        T_combined = T_fk @ T_local
                        _apply_transform(item, T_combined, scale=lt['scale'])
                    else:
                        _apply_transform(item, T_fk)
