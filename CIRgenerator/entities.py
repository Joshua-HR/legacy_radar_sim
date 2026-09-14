from dataclasses import dataclass, field
from typing import List, Tuple
from dataconfig import SceneConfig, as_xyz, vector_to_phi_theta_deg, point_to_phi_theta_deg, distance_3d
from numpy import array, ndarray, linspace, deg2rad, cos, sin, pi, linalg
@dataclass
class ReflectionPath:
    """
    Container for additional reflected path relative to the direct path
    """
    extra_distance_m: float
    attenuation: float
    phase_offset_rad: float = 0.0


@dataclass
class StaticClutterPath:
    """
    Container for Fixed environment reflector that stays in the same place for all frames.
    """

    name: str
    position_xy_m: Tuple[float, float]
    velocity_xy_m_per_frame: Tuple[float, float] = (0.0, 0.0)
    position_z_m: float = 0.0
    velocity_z_m_per_frame: float = 0.0
    amplitude: float = 1.0
    width_m: float = 1.0
    num_scatter_points: int = 1
    orientation_deg: float = 90.0
    material_type: str = "wall"
    material_factor: float = 1.0

    # Radar cross-section, in square metres, used by the round-trip radar
    # equation in CIRSimulator._radar_equation_amplitude_factor as
    # sqrt(rcs_m2). This is the only physically-dimensioned scattering
    # quantity on this class: `amplitude` and `material_factor` above remain
    # unitless empirical multipliers applied on top of it (they are set only
    # by the hardcoded scenario profiles and are NOT cross-sections).
    # The 1.0 default makes sigma = 1 m^2 the explicit engine-wide convention
    # and reproduces the pre-radar-equation amplitude exactly.
    rcs_m2: float = 1.0
    reflections: List[ReflectionPath] = field(default_factory=list)

    # --- 3D vertical extent -----------------------------------------------
    # Vertical (z) extent of the scatterer. When height_m <= 0 (default) the
    # scatter model collapses to the legacy horizontal-line behavior, so
    # existing callers stay bit-identical. When height_m > 0 AND
    # num_scatter_points_z > 1, scatter points form a rectangular
    # (width x height) grid in the local frame: horizontal offsets along
    # orientation_deg, vertical offsets from bottom_z_m to
    # bottom_z_m + height_m. Vertical offsets are computed relative to
    # the (micro-motion-adjusted for Target) center z, so a wall with
    # bottom_z_m=0 and height_m=room_height_z_m spans floor-to-ceiling
    # regardless of where its center sits.
    height_m: float = 0.0
    num_scatter_points_z: int = 1
    bottom_z_m: float = 0.0

    # --- 3D horizontal surface (floor/ceiling) ----------------------------
    # 2nd horizontal extent, perpendicular to orientation_deg. When depth_m <= 0
    # (default) the scatter model collapses to the 1D line along orientation_deg,
    # so existing callers stay bit-identical. When depth_m > 0 AND
    # num_scatter_points_depth > 1, scatter points form a 2D rectangular grid in
    # the xy-plane (width x depth), enabling floor/ceiling surfaces.
    depth_m: float = 0.0
    num_scatter_points_depth: int = 1

    def _get_center_position(self, frame_idx: int) -> ndarray:
        """Returns the target position at specific frame_idx

        Args:
            frame_idx (int): time frame for which the position is calculated

        Returns:
            np.ndarray: array shape [3] with target xyz position
        """
        x0, y0 = self.position_xy_m
        vx, vy = self.velocity_xy_m_per_frame
        z0 = self.position_z_m
        vz = self.velocity_z_m_per_frame

        return array(
            [x0 + frame_idx * vx, y0 + frame_idx * vy, z0 + frame_idx * vz],
            dtype=float,
        )
    def _build_scatter_grid(self, center: ndarray) -> List[ndarray]:
        """Build scatter points around an already-computed xyz center.

        Three modes, selected by height_m / depth_m flags:
        - height_m <= 0 AND depth_m <= 0 (default): legacy horizontal line,
          identical to pre-3D-extent behavior.
        - height_m > 0 AND num_scatter_points_z > 1: vertical wall surface grid
          (num_scatter_points x num_scatter_points_z). Vertical offsets are relative
          to center[2] so the grid spans [bottom_z_m, bottom_z_m + height_m] regardless of
          where the center sits.
        - depth_m > 0 AND num_scatter_points_depth > 1: horizontal surface grid
          (num_scatter_points x num_scatter_points_depth) in the xy-plane, for
          floor/ceiling. depth_dir is perpendicular to width_dir in xy-plane.

        Args:
            center (np.ndarray): xyz reference point, shape [3].

        Returns:
            (List[np.ndarray]): list of scatter points, each shape [3].
        """
        # Horizontal direction in the local frame (xy-plane yaw only;
        # orientation_deg does not tilt out of the horizontal plane).
        theta = deg2rad(self.orientation_deg)
        width_dir = array([cos(theta), sin(theta), 0.0], dtype=float)
        # Perpendicular horizontal direction for depth_m (floor/ceiling surfaces).
        depth_dir = array([-sin(theta), cos(theta), 0.0], dtype=float)

        # Horizontal offsets along orientation_deg: single point if width_m <= 0 or N <= 1.
        if self.width_m > 0.0 and self.num_scatter_points > 1:
            half_width = self.width_m / 2.0
            h_offsets = linspace(-half_width, half_width, int(self.num_scatter_points))
        else:
            h_offsets = [0.0]

        # Depth offsets (perpendicular to orientation_deg, in xy-plane). Only used
        # when depth_m > 0 AND num_scatter_points_depth > 1; otherwise the scatter is
        # a 1D line along orientation_deg (legacy behavior, bit-identical).
        if self.depth_m > 0.0 and self.num_scatter_points_depth > 1:
            half_depth = self.depth_m / 2.0
            d_offsets = linspace(-half_depth, half_depth, int(self.num_scatter_points_depth))
        else:
            d_offsets = [0.0]

        # Vertical offsets (relative to center.z). Only used when
        # height_m > 0 AND num_scatter_points_z > 1; otherwise the
        # scatter model collapses to a horizontal line at center.z.
        if self.height_m > 0.0 and self.num_scatter_points_z > 1:
            z_bottom_rel = self.bottom_z_m - center[2]
            z_top_rel = self.bottom_z_m + self.height_m - center[2]
            v_offsets = linspace(z_bottom_rel, z_top_rel, int(self.num_scatter_points_z))
        else:
            v_offsets = [0.0]

        return [
            center + h * width_dir + d * depth_dir + array([0.0, 0.0, v], dtype=float)
            for h in h_offsets
            for d in d_offsets
            for v in v_offsets
        ]

    def _get_scatter_points(self, frame_idx: int) -> list[ndarray]:
        """
        Build scatter points for a fixed object.

        Scatter points from a horizontal line by default (x-y plane
        offsets only, z taken from the center). When height_m > 0 and
        num_scatter_points_z > 1, they form a rectangular
        (width x height) surface grid; see _build_scatter_grid().

        Args:
            frame_idx (int): time frame for which the position is calculated

        Returns:
            (list[np.nadarray]): list of scatter points, each shape [3].
        """
        return self._build_scatter_grid(self._get_center_position(frame_idx))

@dataclass
class Target(StaticClutterPath):
    """
    Class for Target definition. Inherits from StaticClutter, but implements motion and micromotion
    """

    # Micro-motion / breathing model
    enable_micro_motion: bool = False
    micro_motion_amplitude_m: float = 0.0
    micro_motion_frequency_hz: float = 0.25
    micro_motion_phase_rad: float = 0.0
    micro_motion_axis: str = "radial"
    # Slow-time frame period in seconds. Default to 20 ms (matching RadarTestConfig.period)
    # CIRSimulator syncs this from cfg.radar.period during __init__.
    # Bugfix: human breathing [KMG, 260616]
    frame_period_s: float = 0.02

    def _get_frame_time_s(self, frame_idx: int) -> float:
        """Convert frame index to slow0time in seconds.

        Uses the target's frame_period_s attribute, which is synced from
        the simulator's cfg.radar.period during CIRSimulator.__init__

        Args:
            frame_idx (int): frame index

        Returns:
            float: time in seconds
        """
        return frame_idx * self.frame_period_s

    def _apply_target_micro_motion(self, frame_idx: int) -> ndarray:
        """
        Apply small periodic chest-like motion to the target center.

        Supported directions:
            - radial: along line from radar origin to target
            - x: along x axis
            - y: along y axis

        Args:
            frame_idx (int): time frame for which the position is calculated

        Returns:
            (np.ndarray): new target position
        """
        if not self.enable_micro_motion or self.micro_motion_amplitude_m <= 0.0:
            return self._get_center_position(frame_idx)

        t = self._get_frame_time_s(frame_idx)
        disp = self.micro_motion_amplitude_m * sin(
            2.0 * pi * self.micro_motion_frequency_hz * t + self.micro_motion_phase_rad
        )

        moved = self._get_center_position(frame_idx).copy()

        if self.micro_motion_axis == "x":
            moved[0] += disp
        
        elif self.micro_motion_axis == "y":
            moved[1] += disp

        elif self.micro_motion_axis == "z":
            moved[2] += disp

        else:
            # Default = radial, using the full 3D center vector (assumes
            # radar origin at [0, 0, 0]; radial motion relative to an
            # arbitrary radar origin would need that origin passed in).
            base = self._get_center_position(frame_idx)
            norm = linalg.norm(base)
            if norm > 1e-12:
                radial_dir = base / norm
                moved = base + disp * radial_dir
            else:
                moved[0] += disp

        return moved

    def _get_scatter_points(self, frame_idx: int) -> list[ndarray]:
        """
        Build scatter points for a target, using the micro-motion-adjusted
        center position so that breathing / chest displacement is reflected
        in the actual CIR-generating scatter points.

        Overrides StaticClutterPath._get_scatter_points which only uses the
        velocity-adjusted center and ignores micro-motion.

        Args:
            frame_idx (int): time frame for which the position is calculated

        Returns:
            (List[np.ndarray]): list of scatter points.
        """
        return self._build_scatter_grid(self._apply_target_micro_motion(frame_idx))

    def _get_ground_truth(
            self,
            frame_idx: int,
            target_center: "ndarray",
            phi_deg: float,
            theta_deg: float,
            distance_center_m: float = None,
    )->dict:
        """Build the per-frame ground-truth dict for this target.

        Args:
            frame_idx (int): frame index (0-based; stored as 1-based "frame").
            target_center (np.ndarray): xyz target center, shape [3] (or [2]).
            phi_deg (float): azimuth angle from radar origin, project convention.
            theta_deg (float): polar angle (from +z) from radar origin,
                project convention - 0=+z axis, 90=horizon, 180=-z axis.
            distance_center_m (Optional[float]): 3D distancce from radar origin
                to target_center, if the caller (CIRSimulator) has the radar
                origin available. If None, falls back to norm(target_center),
                which is only correct when radar origin is [0, 0, 0].
        
        Returns:
            dict: ground truth fields, including legacy angle_deg (azimuth)
            and the new theta/phi/elevation/azimuth fields. theta_deg/phi_deg
            are the raw polar-angle/azimuth pair used by the antenna pattern
            code; elevation_deg is a separate, more intuitive elevation-from-
            horizon field (elevation_deg = 90 - theta_deg) kept for
            downstream consumers/plots that expect 0=horizon.
        """
        center_xyz = as_xyz(target_center)

        if distance_center_m is None:
            # TODO: this assumes radar origin is [0, 0, 0]; prefer passing
            # distance_center_m computed from the actual radar origin.
            distance_center_m = float(linalg.norm(center_xyz))

        return {
            "frame": frame_idx + 1,
            "target_name": self.name,
            "position_x_m": float(center_xyz[0]),
            "position_y_m": float(center_xyz[1]),
            "position_z_m": float(center_xyz[2]),
            "angle_deg": float(phi_deg),    # backward-compatible: azimuth
            "theta_deg": float(theta_deg),
            "phi_deg": float(phi_deg),
            "elevation_deg": float(90.0 - theta_deg),
            "azimuth_deg": float(phi_deg),
            "distance_center_m": float(distance_center_m),
            "amplitude": float(self.amplitude),
            "width_m": float(self.width_m),
            "num_scatter_points": int(self.num_scatter_points),
            "orientation_deg": float(self.orientation_deg),
            "material_type": str(self.material_type),
            "material_factor": float(self.material_factor),
            "height_m": float(self.height_m),
            "num_scatter_points_z": int(self.num_scatter_points_z),
            "bottom_z_m": float(self.bottom_z_m),
            "enable_micro_motion": bool(self.enable_micro_motion),
            "micro_motion_amplitude_m": float(self.micro_motion_amplitude_m),
            "micro_motion_frequency_hz": float(self.micro_motion_frequency_hz),
            "micro_motion_phase_rad": float(self.micro_motion_phase_rad),
            "micro_motion_axis": str(self.micro_motion_axis),
        }
    
@dataclass
class Room:
    scene: SceneConfig
    material: str = "wall"
    amplitude: float = 0.2
    num_scatter_points: int = 25
    material_factor: float = 1.0

    def _place_walls(self):
        """Place the room enclosing surfaces as 3D scatters.

        In "3d" mode the room is fully enclosed with 6 surfaces:
            - 4 walls (left/right/front/back): each spans floor -> ceiling via
              height_m = scene.room_height_z_m and bottom_z_m = 0.0, with a
              sparse vertical scatter grid (num_scatter_points_z) so the total
              scatter count stays reasonable.
            - floor: horizontal surface at z = 0, spans the full room footprint
              (width=L along x, depth=W along y), modeled as a 2D xy-plane grid
              via depth_m=num_scatter_points_depth.
            - ceiling: same as floor but at z=room_height_z_m.
        
        Wall orientation convention:
            - left/right walls: orientation_deg = 0     (width along x)
            - front/back walls: orientation_deg = 90    (width along y)
            - floor/ceiling   : orientation_deg = 0     (width along x, depth along y)
        
        In "2d-legacy" mode only the 4 walls are returned (one-sided layout,
        x in [0, L], y in [-W/2, W/2]) to preserve bit-identical output for
        legacy callers.
        
        Returns:
            (List[StaticClutterPath]): 6 surfaces (3d) or 4 walls (2d_legacy).
        """
        L = self.scene.room_length_x_m
        W = self.scene.room_width_y_m
        H = self.scene.room_height_z_m

        # Sparse vertical scatter: at least 3 points, otherwise ~1/5 of 
        # the horizontal count. Keeps total per-wall scatter count
        # manageable (e.g. 25 horizontal x 5 vertical = 125 points).
        num_scatter_points_z = max(self.num_scatter_points // 5, 3)

        # Sparse depth grid for floor/ceiling: ~1/2 of horizontal count,
        # at least 5. Keeps total per-surface scatter count manageable
        # (e.g. 25 x 12 = 300 points per floor/ceiling).
        num_scatter_points_depth = max(self.num_scatter_points // 2, 5)

        # (position_xy_m, orientation_deg, width_along_direction)
        # Named "wall_back" is new - pre-3D-extent code only emitted 3 walls.
        # In "3d" mode the room is origin-centered: x ∈ [-L/2, L/2], y ∈ [-W/2, W/2].
        # In "2d_legacy" mode the room is one-sided: x ∈ [0, L], y ∈ [-W/2, W/2].
        # (preserves bit-identical output for legacy callers).
        geometry_mode = getattr(self.scene, "geometry_mode", "2d_legacy")
        if geometry_mode == "3d":
            wall_specs = [
                ((0.0,      W / 2.0),   0.0,  L),   # wall_left     (width along x)
                ((0.0,      -W / 2.0),  0.0,  L),   # wall_right    (width along x)
                (( L / 2.0, 0.0),       90.0, W),   # wall_front    (width along y)
                ((-L / 2.0, 0.0),       90.0, W),   # wall_back     (width along y)
            ]
            # Floor/ceiling: horizontal surfaces at z=0 and z=H, spanning full footprint.
            # (position_xy_m, orientation_deg, width_m, depth_m, bottom_z_m)
            surface_specs = [
                ((0.0, 0.0), 0.0, L, W, 0.0),   # floor   (z=0)
                ((0.0, 0.0), 0.0, L, W, H),     # ceiling (z=H)
            ]
        else:
            wall_specs = [
                ((L / 2.0,  W / 2.0),   0.0,  L), # wall_left (width along x)
                ((L / 2.0, -W / 2.0),   0.0,  L), # wall_right (width along x)
                ((L,        0.0),       90.0, W), # wall_front (width along y)
                ((0.0,      0.0),       90.0, W), # wall_back (width along y)
            ]
            surface_specs = []  # 2d_legacy: no floor/ceiling (bit-identical legacy)
        
        surfaces = []
        for idx, (xy, orientation_deg, width) in enumerate(wall_specs):
            name = ["wall_left", "wall_right", "wall_front", "wall_back"][idx]
            surfaces.append(
                StaticClutterPath(
                    name=name,
                    position_xy_m=xy,
                    amplitude=self.amplitude,
                    material_type=self.material,
                    num_scatter_points=self.num_scatter_points,
                    width_m=width,
                    orientation_deg=orientation_deg,
                    material_factor=1.0,
                    height_m=H,
                    num_scatter_points_z=num_scatter_points_z,
                    bottom_z_m=0.0,
                )
            )

        # Floor/ceiling (3d mode only): horizontal 2D surfaces.
        # position_z_m=bottom_z is required because height_m=0 collapse v_offsets
        # to [0.0], so center[2] (=position_z_m) becaomes the scatter z.
        for idx, (xy, orientation_deg, width, depth, bottom_z) in enumerate(surface_specs):
            name = ["floor", "ceiling"][idx]
            surfaces.append(
                StaticClutterPath(
                    name=name,
                    position_xy_m=xy,
                    position_z_m=bottom_z,
                    amplitude=self.amplitude,
                    material_type=self.material,
                    num_scatter_points=self.num_scatter_points,
                    width_m=width,
                    orientation_deg=orientation_deg,
                    material_factor=1.0,
                    height_m=0.0,
                    num_scatter_points_z=1,
                    bottom_z_m=bottom_z,
                    depth_m=depth,
                    num_scatter_points_depth=num_scatter_points_depth,
                )
            )

        return surfaces