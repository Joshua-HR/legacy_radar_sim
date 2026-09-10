import os
import csv
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence
from datetime import datetime
from scipy.constants import speed_of_light
import configparser
from datetime import datetime

from dataconfig import *
from entities import *
from exporter import *
from impairment_params import apply_case_params_to_cfg
import ego_motion

db_to_linear_amplitude = lambda gain_db: 10.0 ** (gain_db / 20.0)
@dataclass
class SimulationConfig:
    """
    Container for main simulation configuration.
    """
    num_antennas: int
    level: int = 1
    """Legacy numeric tag kept mainly for labeling plots/files. The actual simulation scenario is now controlled by:
        - hardware_profile
        - room_profile
        - target_profile
        - direct cfg.targets / cfg.static_clutter overrides
    """
    scenario_name: str = ""
    hardware_profile: str = "ua200_realistic_v1"
    room_profile: str = "small_room_center"
    target_profile: str = "static_object"
    random_seed: int = 42
    output_root: str = "output_cases"
    hardware_params_json: str = ""
    """Path to a case*_params.json fit, consumed only when
    hardware_profile == "fitted". Empty otherwise. Set from the [profiles]
    section by create_default_config()."""

    cir: CIRConfig = field(default_factory=CIRConfig)
    optional: OptionalImpairments = field(default_factory=OptionalImpairments)
    scene: SceneConfig = field(default_factory=SceneConfig)
    radar_motion: RadarMotionConfig = field(default_factory=RadarMotionConfig)
    """Radar ego motion (slow-time radar pose). Default type="static" is inert
    and takes a bit-identical legacy fast path. Set from the optional
    [radar_motion] INI section by create_default_config()."""
    targets: List[Target] = field(default_factory=list)
    static_clutter: List[StaticClutterPath] = field(default_factory=list)

    #metadata
    radar: RadarTestConfig = field(default_factory=RadarTestConfig)
    dut: DUTConfig = field(default_factory=DUTConfig)
    build: BuildInfo = field(default_factory=BuildInfo)

    def __post_init__(self):
        self.leakage = LeakageConfig(self.num_antennas)
        self.antenna = AntennaConfig(self.num_antennas)
        self.frontend = FrontendConfig(self.num_antennas)

import configparser
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


def _get_float_list(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    expected_len: Optional[int] = None,
) -> list[float]:
    raw = parser.get(section, option)
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]

    if expected_len is not None and len(values) != expected_len:
        raise ValueError(
            f"[{section}] {option} must have {expected_len} values, got {len(values)}: {values}"
        )

    return values


def _require_sections(
    parser: configparser.ConfigParser,
    sections: Sequence[str],
) -> None:
    missing = [
        section
        for section in sections
        if not parser.has_section(section)
    ]

    if missing:
        raise ValueError(
            f"Missing required section(s) in synthetic default config: {missing}"
        )


def _default_synthetic_config_path() -> Path:
    return Path(__file__).resolve().parent / "synthetic_default.ini"


def create_default_config(
    config_path: Optional[str | Path] = None,
) -> SimulationConfig:
    """Create SimulationConfig from synthetic_default.ini.

    Args:
        config_path:
            Optional path to an INI file. If omitted, this function loads
            synthetic_default.ini located in the same directory as this module.

    Returns:
        SimulationConfig: configuration object populated from INI.
    """
    ini_path = (
        Path(config_path)
        if config_path is not None
        else _default_synthetic_config_path()
    )

    if not ini_path.exists():
        raise FileNotFoundError(
            f"Synthetic default config not found: {ini_path}"
        )

    parser = configparser.ConfigParser()
    read_files = parser.read(ini_path, encoding="utf-8")

    if not read_files:
        raise RuntimeError(
            f"Failed to read synthetic default config: {ini_path}"
        )

    _require_sections(
        parser,
        [
            "simulation",
            "antenna",
            "cir",
            "frontend",
            "radar",
            "build",
            "profiles",
        ],
    )

    cfg = SimulationConfig(
        num_antennas=parser.getint("simulation", "num_antennas"),
    )

    # Hardware defaults
    cfg.antenna.antenna_spacing_m = parser.getfloat(
        "antenna",
        "antenna_spacing_m",
    )
    cfg.antenna.carrier_freq_hz = parser.getfloat(
        "antenna",
        "carrier_freq_hz",
    )
    cfg.antenna.pattern_mode = parser.get(
        "antenna",
        "pattern_mode",
    )
    cfg.antenna.boresight_deg = parser.getfloat(
        "antenna",
        "boresight_deg",
    )
    # 3D-mode boresight offsets. Optional keys: fall back to safe defaults
    # so older INI files without them keep working.
    cfg.antenna.boresight_phi_deg = parser.getfloat(
        "antenna",
        "boresight_phi_deg",
        fallback=cfg.antenna.boresight_deg,
    )
    cfg.antenna.boresight_theta_deg = parser.getfloat(
        "antenna",
        "boresight_theta_deg",
        fallback=0.0,
    )
    cfg.antenna.min_gain = parser.getfloat(
        "antenna",
        "min_gain",
    )

    # Free-space (Friis) path-loss model. fallback= values reproduce the
    # previous hardcoded exponent values (2.0/2.3/2.5) for INI files that
    # don't have these keys yet.
    cfg.antenna.target_path_loss_exp = parser.getfloat(
        "antenna",
        "target_path_loss_exp",
        fallback=2.0,
    )
    cfg.antenna.clutter_path_loss_exp = parser.getfloat(
        "antenna",
        "clutter_path_loss_exp",
        fallback=2.3,
    )
    cfg.antenna.reflection_path_loss_exp = parser.getfloat(
        "antenna",
        "reflection_path_loss_exp",
        fallback=2.5,
    )
    cfg.antenna.path_loss_ref_distance_m = parser.getfloat(
        "antenna",
        "path_loss_ref_distance_m",
        fallback=1.0,
    )
    cfg.antenna.path_loss_floor_m = parser.getfloat(
        "antenna",
        "path_loss_floor_m",
        fallback=0.1,
    )
    cfg.antenna.path_loss_ref_gain = parser.getfloat(
        "antenna",
        "path_loss_ref_gain",
        fallback=112090.1475,
    )

    # pattern_phi_deg / pattern_theta_deg / pattern_gains_dbi_2d default to
    # the AntennaConfig code defaults (all-zero-dBi isotropic 3D table)
    # unless [antenna] pattern_csv_ant<N> keys are present, in which case a
    # measured G(phi, theta) CSV is loaded per antenna (see
    # load_pattern_gains_dbi_2d_csv()/load_pattern_gains_dbi_2d_per_ant_csv()
    # in dataconfig.py). Only pattern_mode == "table_3d" consumes these -
    # other modes ignore pattern_gains_dbi_2d(_per_ant) entirely.
    pattern_csv_paths = []
    ant_idx = 0
    while parser.has_option("antenna", f"pattern_csv_ant{ant_idx}"):
        pattern_csv_paths.append(parser.get("antenna", f"pattern_csv_ant{ant_idx}"))
        ant_idx += 1

    if pattern_csv_paths:
        (
            cfg.antenna.pattern_phi_deg,
            cfg.antenna.pattern_theta_deg,
            cfg.antenna.pattern_gains_dbi_2d_per_ant,
        ) = load_pattern_gains_dbi_2d_per_ant_csv(pattern_csv_paths)
        # pattern_gains_dbi_2d (the shared/fallback table) mirrors antenna 0's
        # table, so any antenna index beyond len(pattern_csv_paths) - 1 (not
        # given its own CSV) still gets a real measured pattern rather than
        # silently falling back to the all-zero-dBi isotropic default.
        cfg.antenna.pattern_gains_dbi_2d = cfg.antenna.pattern_gains_dbi_2d_per_ant[0]
        
    # num_antennas count, reused below for per-antenna list validation.
    # 
    # Note: the legacy [feedthrough] INI section is intentionally not read.
    # Feedthrough / early-leakage are driven entirely by the per-antenna
    # scalars set in _update_derived_hardware_params() (built-in profiles) or
    # injected from a fit JSON (hardware_profile = "fitted"); the old
    # [feedthrough] keys were dead config and have been removed.
    n_ant = cfg.num_antennas

    # CIR defaults
    cfg.cir.num_bins = parser.getint(
        "cir",
        "num_bins",
    )
    cfg.cir.bin_time_s = parser.getfloat(
        "cir",
        "bin_time_s",
    )
    cfg.cir.num_frames = parser.getint(
        "cir",
        "num_frames",
    )
    cfg.cir.noise_std = parser.getfloat(
        "cir",
        "noise_std",
    )
    if parser.has_option("cir", "noise_std_per_ant"):
        cfg.cir.noise_std_per_ant = _get_float_list(
            parser, "cir", "noise_std_per_ant", expected_len=n_ant,
        )

    # Frontend defaults
    cfg.frontend.enable_quantization = parser.getboolean(
        "frontend",
        "enable_quantization",
    )
    cfg.frontend.enable_adc_clipping = parser.getboolean(
        "frontend",
        "enable_adc_clipping",
    )
    cfg.frontend.enable_frame_drift = parser.getboolean(
        "frontend",
        "enable_frame_drift",
    )

    # Metadata
    cfg.radar.test_name = parser.get(
        "radar",
        "test_name",
    )
    cfg.radar.cir_taps = parser.getint(
        "radar",
        "cir_taps",
    )
    cfg.radar.period = parser.getint(
        "radar",
        "period",
        fallback=20,
    )

    build_date = parser.get(
        "build",
        "build_date",
        fallback="now",
    ).strip()

    if build_date.lower() == "now":
        cfg.build.build_date = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    else:
        cfg.build.build_date = build_date

    # Default profiles
    cfg.hardware_profile = parser.get(
        "profiles",
        "hardware_profile",
    )
    cfg.room_profile = parser.get(
        "profiles",
        "room_profile",
    )
    cfg.target_profile = parser.get(
        "profiles",
        "target_profile",
    )
    # Optional: path to a fitted-impairment JSON, consumed only when
    # hardware_profile == "fitted" (see _inject_params_from_json).
    cfg.hardware_params_json = parser.get(
        "profiles", "hardware_params_json", fallback=""
    )
    # Optional named radar ego-motion preset. Empty (default) defers to the
    # [radar_motion] section below. The wrapper's [synthetic] section can
    # override this by name, the same way it overrides the other three profiles.
    cfg.radar_motion.profile = parser.get(
        "profiles", "radar_motion_profile", fallback=""
    ).strip()

    # Scene / geometry_mode. [scene] is optional - INI files without it
    # keep the SceneConfig dataclass defaults (2d_legacy, room 4x4x2.6m,
    # radar at origin height 1.2m), same fallback pattern as
    # boresight_phi_deg/boresight_theta_deg above.
    if parser.has_section("scene"):
        cfg.scene.geometry_mode = parser.get(
            "scene", "geometry_mode", fallback=cfg.scene.geometry_mode,
        )
        cfg.scene.room_length_x_m = parser.getfloat(
            "scene", "room_length_x_m", fallback=cfg.scene.room_length_x_m,
        )
        cfg.scene.room_width_y_m = parser.getfloat(
            "scene", "room_width_y_m", fallback=cfg.scene.room_width_y_m,
        )
        cfg.scene.room_height_z_m = parser.getfloat(
            "scene", "room_height_z_m", fallback=cfg.scene.room_height_z_m,
        )
        cfg.scene.radar_position_x_m = parser.getfloat(
            "scene", "radar_position_x_m", fallback=cfg.scene.radar_position_x_m,
        )
        cfg.scene.radar_position_y_m = parser.getfloat(
            "scene", "radar_position_y_m", fallback=cfg.scene.radar_position_y_m,
        )
        cfg.scene.radar_height_z_m = parser.getfloat(
            "scene", "radar_height_z_m", fallback=cfg.scene.radar_height_z_m,
        )
        # radar_placement_mode / enable_default_room_clutter are not read
        # here: _apply_room_profile() unconditionally overwrites both from
        # the room_profile name, so an INI value here would be silently
        # clobbered.
        
    # Radar ego motion. [radar_motion] is optional (omitted == static ==
    # bit-identical legacy behavior), but WITHIN the section the surface is
    # strict: ego_motion.validate_ini_keys() rejects any unknown key and any
    # key belonging to a different `type`, so a typo cannot silently disable
    # the motion. Note this is deliberately stricter than the rest of this
    # function, which uses tolerant `fallback=` reads throughout.
    if parser.has_section(ego_motion.INI_SECTION):
        section = ego_motion.INI_SECTION
        cfg.radar_motion.type = parser.get(
            section, "type", fallback=cfg.radar_motion.type,
        ).strip().lower()

        # parser.options() folds in [DEFAULT] keys, which are not this
        # section's own and must not trip the strict-key check.
        section_keys = [
            key for key in parser.options(section) if key not in parser.defaults()
        ]
        ego_motion.validate_ini_keys(cfg.radar_motion.type, section_keys)

        if cfg.radar_motion.type == "sampled_pose":
            cfg.radar_motion.file = parser.get(section, "file", fallback="")
            cfg.radar_motion.reference = ego_motion.validate_required_literal(
                "reference",
                parser.get(section, "reference", fallback=cfg.radar_motion.reference),
            )
            cfg.radar_motion.extrapolation = ego_motion.validate_required_literal(
                "extrapolation",
                parser.get(section, "extrapolation", fallback=cfg.radar_motion.extrapolation),
            )
        
        elif cfg.radar_motion.type == "handheld_jitter":
            if parser.has_option(section, "translation_rms_local_m"):
                cfg.radar_motion.translation_rms_local_m = tuple(
                    _get_float_list(parser, section, "translation_rms_local_m", expected_len=3)
                )
            if parser.has_option(section, "rotation_rms_local_rad"):
                cfg.radar_motion.rotation_rms_local_rad = tuple(
                    _get_float_list(parser, section, "rotation_rms_local_rad", expected_len=3)
                )
            cfg.radar_motion.min_frequency_hz = parser.getfloat(
                section, "min_frequency_hz", fallback=cfg.radar_motion.min_frequency_hz,
            )
            cfg.radar_motion.max_frequency_hz = parser.getfloat(
                section, "max_frequency_hz", fallback=cfg.radar_motion.max_frequency_hz,
            )
            cfg.radar_motion.num_spectral_components = parser.getint(
                section,
                "num_spectral_components",
                fallback=cfg.radar_motion.num_spectral_components,
            )

    # Needed to resolve a relative sampled_pose trace path against the file
    # that declared it. Recorded unconditionally so the resolution base is
    # visible in the saved config even for static runs.
    cfg.radar_motion.scenario_dir = str(ini_path.resolve().parent)

    return cfg



class CIRSimulator:
    """Main simulator class. Generate the CIR based of the configuration data.
    """
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        """SimulationConfig object that sets basic configuration values."""
        self.rng = np.random.default_rng(config.random_seed)
        # Apply realism profiles first
        self.antenna_positions = self._build_antenna_positions()

        self._apply_hardware_profile()
        self._apply_room_profile()
        self._apply_target_profile()
        self._sync_target_frame_period()
        self._update_derived_hardware_params()

        self.lambda_m = self.cfg.antenna.c / self.cfg.antenna.carrier_freq_hz

        self._debug_check_3d_geometry()

        # Radar ego motion. Built LAST and eagerly: CSV parsing, tremor
        # synthesis, grid-coverage and Nyquist checks all happen here, so a bad
        # motion configuration fails during construction rather than part-way
        # through path generation. No profile touches cfg.scene.radar_position_*
        # or radar_height_z_m, so the base pose is already final at this point.
        self._setup_radar_motion()

    def _debug_check_3d_geometry(self) -> None:
        """Lightweight sanity check (not a test suite) that the 3D geometry
        migration didn't change array shapes CIR generation depends on."""
        assert self.antenna_positions.shape[1] == 3, (
            f"antenna_positions must be [N, 3], got {self.antenna_positions.shape}"
        )

        if self.cfg.targets:
            first_center = self.cfg.targets[0]._get_center_position(0)
            assert len(first_center) == 3, (
                f"target center must be length 3, got {len(first_center)}"
            )

    def _radar_origin(self) -> np.ndarray:
        """Radar origin position, shape [3].

        Used as the reference point for target phi/theta (azimuth/polar
        angle) and distance_center_m in ground truth. z follows the same
        geometry_mode gating as _build_antenna_positions() so that
        "2d_legacy" naturally yields theta_deg=90.0 (horizon) for
        x-y-plane targets (antenna z=0, target z defaults to 0.0, radar
        origin z=0.0), while "3d" uses the real radar height, so theta_deg
        can differ from 90.0.
        """
        geometry_mode = getattr(self.cfg.scene, "geometry_mode", "2d_legacy")
        rz = self.cfg.scene.radar_height_z_m if geometry_mode == "3d" else 0.0

        return np.array([
            self.cfg.scene.radar_position_x_m,
            self.cfg.scene.radar_position_y_m,
            rz,
        ], dtype=float)
    
    def _build_antenna_positions(self) -> np.ndarray:
        """
        Build 3D antenna geometry.

        Coordinate meaning:
            x = forward/range direction
            y = left/right direction
            z = height/vertical direction

        Antennas are placed side by side on the y-axis, centered around the
        radar scene position. z is 0.0 in "2d_legacy" geometry_mode, and
        cfg.scene.radar_height_z_m in "3d" geometry_mode.

        Returns: numpy array of antenna positions, shape [num_antennas, 3].
        """
        n = self.cfg.num_antennas
        d = self.cfg.antenna.antenna_spacing_m

        rx = self.cfg.scene.radar_position_x_m
        ry = self.cfg.scene.radar_position_y_m

        geometry_mode = getattr(self.cfg.scene, "geometry_mode", "2d_legacy")
        rz = self.cfg.scene.radar_height_z_m if geometry_mode == "3d" else 0.0

        x_positions = np.full(n, rx, dtype=float)
        y_offsets = np.linspace(-(n - 1) * d / 2, (n - 1) * d / 2, n)
        y_positions = ry + y_offsets
        z_positions = np.full(n, rz, dtype=float)

        return np.stack([x_positions, y_positions, z_positions], axis=1)

    # -----------------------------------------------------------------------
    # Radar ego motion (slow-time radar pose)
    #
    # Pipeline boundary, enforced here and nowhere else:
    #
    #   slow_time + configured base pose
    #       -> pose trajectory (one pose per slow-time sample)
    #       -> Tx/Rx antenna phase-centre world coordinates
    #       -> path geometry -> delay -> carrier phase -> Doppler
    #
    # The motion model never writes a CIR phase or a tap. It only moves the
    # antenna phase centres; _generate_one_frame() then recomputes distance ->
    # delay -> phase from that geometry as it always did, so Doppler falls out
    # of the time-varying path length as a derived quantity and is never
    # multiplied in a second time.
    #
    # Impairments are strictly downstream: _apply_tx_rx_feedthrough*,
    # _apply_radiation_leakage, _apply_pcb_leakage, _apply_early_leakage,
    # _apply_rx_gain, _apply_adc_clipping, _apply_quantization and the noise
    # block all run after the geometry loop and take no pose argument. Tx->Rx
    # coupling is body-fixed, so that is physically correct as well as
    # structurally convenient.
    # -----------------------------------------------------------------------

    def _build_ego_base_pose(self) -> "ego_motion.Pose":
        """The configured static radar pose that ego motion is relative to.

        Position is the existing scene radar position (via _radar_origin(), so
        the geometry_mode z gating is shared and cannot drift). Orientation is
        IDENTITY: this engine has no radar-orientation configuration field, and
        AntennaConfig.boresight_* is an antenna-pattern lookup offset rather
        than a radar body rotation (subtracting boresight_theta_deg is not an
        SO(3) action off the boresight meridian), so it is deliberately not
        reinterpreted as a base attitude here.

        Returns:
            ego_motion.Pose: base pose, identity attitude.
        """
        return ego_motion.Pose.from_parts(
            self._radar_origin(), ego_motion.IDENTITY_QUAT_WXYZ
        )

    def _build_slow_time_grid_s(self) -> np.ndarray:
        """Slow-time sample instants for the whole run, shape [num_frames].

        Derived from _get_frame_time_s() so the motion model and the target
        micro-motion model share exactly one time base (frame_idx * period).

        Returns:
            np.ndarray shape [num_frames], seconds.
        """
        return np.array(
            [self._get_frame_time_s(f) for f in range(self.cfg.cir.num_frames)],
            dtype=float,
        )

    def _setup_radar_motion(self):
        """Build the ego-motion trajectory and the per-frame antenna geometry.

        Sets:
            self._ego_base_pose: configured static pose.
            self._antenna_offsets_body_m: [N, 3] rigid body-frame antenna
                offsets, i.e. the nominal array expressed relative to the base
                position. Antennas are rigidly attached, so these never change.
            self._slow_time_grid_s: [F] slow-time instants.
            self.radar_motion: the ego_motion.RadarMotion model.
            self.ego_pose_log: None when static; otherwise a list of F Poses.
            self._antenna_positions_by_frame: None when static; otherwise
                [F, N, 3] world antenna positions.
            self.ego_motion_metadata: reproducibility metadata (also attached
                to cfg.radar_motion.metadata so it reaches cir_metadata.json
                through CIRMetadata.raw_config without any writer change).

        NOTE self.antenna_positions keeps its pre-existing shape [N, 3] and its
        pre-existing value (the nominal/base-pose array). Every existing
        consumer -- the plotters in exporter.py, save_all_figures, plot_total,
        the validation metadata dict -- reads that attribute and would silently
        mis-plot rather than raise if it were reshaped to [F, N, 3]. Per-frame
        geometry is therefore exposed only through _antenna_positions_at_frame().
        """
        self._ego_base_pose = self._build_ego_base_pose()
        self._antenna_offsets_body_m = (
            self.antenna_positions - self._ego_base_pose.position_m[None, :]
        )
        self._slow_time_grid_s = self._build_slow_time_grid_s()

        # Two mutually exclusive ways in: a named preset, or the [radar_motion]
        # section's `type` + parameters. Refusing the ambiguous combination is
        # deliberate -- silently letting one shadow the other is exactly how a
        # scenario ends up not running the motion its author thought it did.
        profile = (self.cfg.radar_motion.profile or "").strip()
        section_type = (self.cfg.radar_motion.type or "static").strip().lower()

        if profile and section_type != "static":
            raise ValueError(
                f"ambiguous radar ego-motion configuration: radar_motion_profile = "
                f"{profile!r} was selected while [radar_motion] also declares "
                f"type = {section_type!r}. Pick one -- either the named preset or the "
                f"explicit section, not both."
            )

        if profile:
            self.radar_motion = ego_motion.build_radar_motion_from_profile(
                profile,
                targets=self.cfg.targets,
                radar_origin_m=self._ego_base_pose.position_m,
                slow_time_s=self._slow_time_grid_s,
                random_seed=self.cfg.random_seed,
            )
        else:
            self.radar_motion = ego_motion.build_radar_motion(
                self.cfg.radar_motion,
                slow_time_s=self._slow_time_grid_s,
                random_seed=self.cfg.random_seed,
                scenario_dir=self.cfg.radar_motion.scenario_dir,
            )

        self.ego_motion_metadata = dict(self.radar_motion.metadata())
        self.ego_motion_metadata["radar_motion_profile"] = profile or None
        self.cfg.radar_motion.metadata = self.ego_motion_metadata

        if self.radar_motion.is_static:
            # Legacy fast path: nothing precomputed, nothing to look up.
            self.ego_pose_log = None
            self._antenna_positions_by_frame = None
            return

        self.ego_pose_log = [
            self.radar_motion.evaluate(float(t), self._ego_base_pose)
            for t in self._slow_time_grid_s
        ]
        self._antenna_positions_by_frame = np.stack(
            [pose.transform_body_points(self._antenna_offsets_body_m) for pose in self.ego_pose_log],
            axis=0,
        )

    def _ego_pose_at_frame(self, frame_idx: int) -> "ego_motion.Pose":
        """Radar pose at one slow-time sample.

        Args:
            frame_idx (int): frame index, 0-based.

        Returns:
            ego_motion.Pose: the base pose when static, else the trajectory
            sample. Always exactly one finite pose.
        """
        if self.ego_pose_log is None:
            return self._ego_base_pose
        
        return self.ego_pose_log[frame_idx]

    def _radar_origin_at_frame(self, frame_idx: int) -> np.ndarray:
        """Radar body origin at one slow-time sample, shape [3].

        Reference point for target phi/theta and distance_center_m in the
        ground truth. Returns the exact same array as _radar_origin() when
        motion is static.

        Args:
            frame_idx (int): frame index, 0-based.

        Returns:
            np.ndarray shape [3].
        """
        if self.ego_pose_log is None:
            return self._radar_origin()

        return self.ego_pose_log[frame_idx].position_m

    def _antenna_positions_at_frame(self, frame_idx: int) -> np.ndarray:
        """World antenna phase-centre positions at one slow-time sample.

        When motion is static this returns the pre-computed
        self.antenna_positions object ITSELF (not a copy, not a recomputation),
        so the legacy path generators run over the legacy array in the legacy
        order and the resulting CIR is bit-identical to the pre-ego-motion
        engine.

        Args:
            frame_idx (int): frame index, 0-based.

        Returns:
            np.ndarray shape [num_antennas, 3].
        """
        if self._antenna_positions_by_frame is None:
            return self.antenna_positions

        return self._antenna_positions_by_frame[frame_idx]

    def _apply_hardware_profile(self):
        """Function to apply the hardware profile

        Raises:
            ValueError: "Unsupported hardware_profile."
        """
        profile = (self.cfg.hardware_profile or "").strip().lower()

        if profile == "ideal_lab":
            self.cfg.leakage.enable_radiation_leakage = False
            self.cfg.leakage.enable_pcb_leakage = False
            self.cfg.leakage.enable_tx_rx_feedthrough = False
            self.cfg.leakage.enable_early_leakage = False

            self.cfg.frontend.enable_adc_clipping = False
            self.cfg.frontend.enable_quantization = False
            self.cfg.frontend.enable_frame_drift = False

        elif profile == "ua200_realistic_v1":
            self.cfg.leakage.enable_radiation_leakage = True
            self.cfg.leakage.enable_pcb_leakage = True
            self.cfg.leakage.enable_tx_rx_feedthrough = True
            self.cfg.leakage.enable_early_leakage = True

            self.cfg.frontend.enable_adc_clipping = True
            self.cfg.frontend.enable_quantization = True
            self.cfg.frontend.enable_frame_drift = True

        elif profile == "strong_coupling_board":
            self.cfg.leakage.enable_radiation_leakage = True
            self.cfg.leakage.enable_pcb_leakage = False
            self.cfg.leakage.enable_tx_rx_feedthrough = True
            self.cfg.leakage.enable_early_leakage = True

            self.cfg.frontend.enable_adc_clipping = True
            self.cfg.frontend.enable_quantization = True
            self.cfg.frontend.enable_frame_drift = True

        elif profile == "measured_case0":
            # Measurement-fitted preset from case0_params.json (data/cir/260701/case0.log).
            # Numeric values are injected in _update_derived_hardware_params().
            self.cfg.leakage.enable_tx_rx_feedthrough = True
            self.cfg.leakage.enable_tx_rx_feedthrough_ringing = True
            self.cfg.leakage.enable_radiation_leakage = False
            self.cfg.leakage.enable_pcb_leakage = False
            self.cfg.leakage.enable_early_leakage = False

            self.cfg.frontend.enable_adc_clipping = False
            self.cfg.frontend.enable_quantization = True
            self.cfg.frontend.enable_frame_drift = True

            # Match the measured UA200 RX gains used in the fit (gain_deembedded domain)
            self.cfg.radar.rx1_gain = 29
            self.cfg.radar.rx2_gain = 74

        elif profile == "fitted":
            # Config-driven measurement-fitted profile: numeric values are loaded
            # from cfg.hardware_params_json and injected in
            # _update_derived_hardware_params() via _inject_params_from_json().
            # Effective realization = noise + tx_rx_feedthrough + ringing ONLY
            # (matches validation/cir_validation_adapter.build_injected_simulator);
            # quantization / frame_drift / adc_clipping stay OFF.
            self.cfg.leakage.enable_tx_rx_feedthrough = True
            self.cfg.leakage.enable_tx_rx_feedthrough_ringing = True
            self.cfg.leakage.enable_radiation_leakage = False
            self.cfg.leakage.enable_pcb_leakage = False
            self.cfg.leakage.enable_early_leakage = False

            self.cfg.frontend.enable_adc_clipping = False
            self.cfg.frontend.enable_quantization = False
            self.cfg.frontend.enable_frame_drift = False

        else:
            raise ValueError("Unsupported hardware_profile.")

    
    def _update_derived_hardware_params(self):
        """
        Compute adaptive hardware/leakage parameters from the main config.
        This is the only place where derived leakage values are generated.
        """
        lam = self.cfg.antenna.c / self.cfg.antenna.carrier_freq_hz
        d = self.cfg.antenna.antenna_spacing_m
        spacing_ratio = d / max(lam, 1e-12)

        avg_rx_gain = 0.5 * (self.cfg.radar.rx1_gain + self.cfg.radar.rx2_gain)

        profile = (self.cfg.hardware_profile or "").strip().lower()

        if profile == "measured_case0":
            # Measurement-fitted constants from case0_params.json.
            # These bypass formula derivation entirely.
            self._inject_measured_case0_params()
            return

        if profile == "fitted":
            # Config-driven fitted profile: inject values from the JSON fit and
            # bypass formula derivation (same overwrite-trap avoidance as
            # measured_case0 above).
            self._inject_params_from_json(self.cfg.hardware_params_json)
            return

        if self.cfg.leakage.enable_radiation_leakage:
            coupling_strength = 0.08 * np.exp(-2.2 * spacing_ratio)
            gain_factor = 1.0 + max(avg_rx_gain - 30.0, 0.0) / 60.0

            rad = coupling_strength * gain_factor
            rad = float(np.clip(rad, 0.005, 0.12))

            tx_idx = 1  # antenna 1 in code = RX2 + TX

            if tx_idx == 1:
                # stronger coupling from antenna 1 -> antenna 0
                self.cfg.leakage.radiation_leakage_factor_12 = rad * 1.35
                self.cfg.leakage.radiation_leakage_factor_21 = rad * 0.65
            else:
                self.cfg.leakage.radiation_leakage_factor_12 = rad * 0.65
                self.cfg.leakage.radiation_leakage_factor_21 = rad * 1.35

            self.cfg.leakage.radiation_delay_bins = 0 if d < 0.025 else 1
            self.cfg.leakage.radiation_phase_offset_12_rad = 0.05
            self.cfg.leakage.radiation_phase_offset_21_rad = -0.05

        if self.cfg.leakage.enable_pcb_leakage:
            pcb_base = 0.035
            if spacing_ratio < 0.60:
                pcb_base *= 1.20

            gain_factor = 1.0 + max(avg_rx_gain - 30.0, 0.0) / 70.0
            pcb = pcb_base * gain_factor
            pcb = float(np.clip(pcb, 0.01, 0.08))

            tx_idx = 1  # antenna 1 in code = RX2 + TX

            if tx_idx == 1:
                self.cfg.leakage.pcb_leakage_factor_12 = pcb * 1.40
                self.cfg.leakage.pcb_leakage_factor_21 = pcb * 0.60
            else:
                self.cfg.leakage.pcb_leakage_factor_12 = pcb * 0.60
                self.cfg.leakage.pcb_leakage_factor_21 = pcb * 1.40
            
            self.cfg.leakage.pcb_phase_offset_12_rad = 0.03
            self.cfg.leakage.pcb_phase_offset_21_rad = -0.03
            self.cfg.leakage.pcb_delay_bins = 0

        if self.cfg.leakage.enable_tx_rx_feedthrough:
            profile = (self.cfg.hardware_profile or "").strip().lower()

            sic_factor = 0.50 if self.cfg.radar.sic_enable == 1 else 1.0
            gain_factor = 1.0 + max(avg_rx_gain - 30.0, 0.0) / 50.0

            base_tx_leak = 0.42 * sic_factor * gain_factor

            if profile == "strong_coupling_board":
                base_tx_leak *= 1.5

            base_tx_leak = float(np.clip(base_tx_leak, 0.10, 0.90))

            self.cfg.leakage.tx_antenna_index = 1

            # RX1 only: receives coupled TX leakage from antenna 2
            self.cfg.leakage.tx_rx_feedthrough_amp_ant0 = base_tx_leak * 0.85

            # RX2 + TX: strongest self-leakage should be here
            self.cfg.leakage.tx_rx_feedthrough_amp_ant1 = base_tx_leak * 1.35

            self.cfg.leakage.tx_rx_feedthrough_start_tap = 0
            self.cfg.leakage.tx_rx_feedthrough_num_taps = 6 if profile != "strong_coupling_board" else 8
            self.cfg.leakage.tx_rx_feedthrough_decay_ant0 = 0.80
            self.cfg.leakage.tx_rx_feedthrough_decay_ant1 = 0.78
            self.cfg.leakage.tx_rx_feedthrough_phase_ant0_rad = -0.10
            self.cfg.leakage.tx_rx_feedthrough_phase_ant1_rad = 0.18
            self.cfg.leakage.tx_rx_feedthrough_ripple_amp_ant0 = 0.08
            self.cfg.leakage.tx_rx_feedthrough_ripple_amp_ant1 = 0.10
            self.cfg.leakage.tx_rx_feedthrough_ripple_freq_ant0 = 1.0
            self.cfg.leakage.tx_rx_feedthrough_ripple_freq_ant1 = 1.2
            self.cfg.leakage.tx_rx_feedthrough_random_std_ant0 = 0.01
            self.cfg.leakage.tx_rx_feedthrough_random_std_ant1 = 0.01


        if self.cfg.leakage.enable_early_leakage:
            sic_factor = 0.55 if self.cfg.radar.sic_enable == 1 else 1.0
            quant_factor = 1.05 if self.cfg.frontend.enable_quantization else 1.0
            drift_factor = 1.05 if self.cfg.frontend.enable_frame_drift else 1.0

            base_early = 0.18 * sic_factor * quant_factor * drift_factor

            profile = (self.cfg.hardware_profile or "").strip().lower()
            if profile == "strong_coupling_board":
                base_early *= 1.6

            base_early = float(np.clip(base_early, 0.05, 0.45))

            self.cfg.leakage.early_leakage_amplitude_ant0 = base_early * 0.80
            self.cfg.leakage.early_leakage_amplitude_ant1 = base_early * 1.25

            self.cfg.leakage.early_leakage_num_taps = 8 if profile == "strong_coupling_board" else 6
            self.cfg.leakage.early_leakage_decay_ant0 = 0.78
            self.cfg.leakage.early_leakage_decay_ant1 = 0.76
            self.cfg.leakage.early_leakage_phase_ant0_rad = -0.20
            self.cfg.leakage.early_leakage_phase_ant1_rad = 0.12
            self.cfg.leakage.early_leakage_ripple_amp_ant0 = 0.10
            self.cfg.leakage.early_leakage_ripple_amp_ant1 = 0.10
            self.cfg.leakage.early_leakage_ripple_freq_ant0 = 1.0
            self.cfg.leakage.early_leakage_ripple_freq_ant1 = 1.15
            self.cfg.leakage.early_leakage_random_std_ant0 = 0.01
            self.cfg.leakage.early_leakage_random_std_ant1 = 0.01


    def _inject_measured_case0_params(self):
        """Inject measurement-fitted impairment constants from case0_params.json.

        Fitted from data/cir/260701/case0.log (UA200, gain_deembedded domain).
        Values are taken directly from the least-squares fit - no formula
        derivation.  See CIRgenerator/notebooks/impairments_fitting.ipynb.
        """
        L = self.cfg.leakage

        # -- TX-RX feedthrough (tap 0-4) --
        L.tx_antenna_index = 1
        L.tx_rx_feedthrough_start_tap = 0
        L.tx_rx_feedthrough_num_taps = 5
        L.tx_rx_feedthrough_amp_ant0 = 308.7549760958734
        L.tx_rx_feedthrough_amp_ant1 = 0.7200434109868733
        L.tx_rx_feedthrough_decay_ant0 = 0.5200851229459805
        L.tx_rx_feedthrough_decay_ant1 = 0.7089544239802014
        L.tx_rx_feedthrough_phase_ant0_rad = -0.24051105213243196
        L.tx_rx_feedthrough_phase_ant1_rad = 0.12929720552638344
        L.tx_rx_feedthrough_ripple_amp_ant0 = 0.025346598017481604
        L.tx_rx_feedthrough_ripple_amp_ant1 = 0.60
        L.tx_rx_feedthrough_ripple_freq_ant0 = 0.8571859616672961
        L.tx_rx_feedthrough_ripple_freq_ant1 = 0.8712055072841399
        L.tx_rx_feedthrough_random_std_ant0 = 0.0
        L.tx_rx_feedthrough_random_std_ant1 = 0.0

        # -- Tx-Rx feedthrough ringing tail (tap 5-15) --
        L.tx_rx_feedthrough_ringing_start_tap = 5
        L.tx_rx_feedthrough_ringing_num_taps = 11
        L.tx_rx_feedthrough_ringing_offset_amp_ant0 = 3.5088542773064533
        L.tx_rx_feedthrough_ringing_offset_amp_ant1 = 0.006636360102148943
        L.tx_rx_feedthrough_ringing_offset_phase_ant0_rad = -2.9313885808115003
        L.tx_rx_feedthrough_ringing_offset_phase_ant1_rad = -2.839619247363284
        L.tx_rx_feedthrough_ringing_amp_ant0 = 2.765879047851826
        L.tx_rx_feedthrough_ringing_amp_ant1 = 0.0006210878436960386
        L.tx_rx_feedthrough_ringing_decay_ant0 = 0.33189529393814926
        L.tx_rx_feedthrough_ringing_decay_ant1 = 0.9989999999999999
        L.tx_rx_feedthrough_ringing_phase_ant0_rad = -0.5263005518908146
        L.tx_rx_feedthrough_ringing_phase_ant1_rad = -1.3414529571380909
        L.tx_rx_feedthrough_ringing_freq_ant0 = 0.22377580920273918
        L.tx_rx_feedthrough_ringing_freq_ant1 = 0.7737726508671994

        # -- Radiation leakage (gated unreliable by fit, factors = 0) --
        L.radiation_delay_bins = 5
        L.radiation_leakage_factor_12 = 0.0
        L.radiation_leakage_factor_21 = 0.0
        L.radiation_phase_offset_12_rad = -0.1632967487580505
        L.radiation_phase_offset_21_rad = 0.16329674875805048


        # -- Noise (per-antenna, gain-deembedded domain) --
        self.cfg.cir.noise_std_per_ant = [0.010242580289304417, 5.759826173255991e-05]
        self.cfg.cir.noise_std = 0.005150089275518488

        # -- Frame drift --
        self.cfg.frontend.frame_amplitude_drift_std = 0.01060406502213973
        self.cfg.frontend.frame_phase_drift_std = [0.001476754146717064, 0.0021325951570907754]
        self.cfg.frontend.frame_timing_jitter_std_bins = 0.0

        # -- Quantization (gain_deembedded domain) --
        self.cfg.frontend.quantization_dc_offset_i = -15.164589709756994
        self.cfg.frontend.quantization_dc_offset_q = -9.62612113856849
        # Derive bits/full_scale from step and full_scale_est in the fit
        q_step = 0.0001989999999999978
        q_full_scale_est = 318.1890529906708
        q_max_level = q_full_scale_est / q_step
        import math as _math
        self.cfg.frontend.quantization_bits = int(np.clip(np.ceil(np.log2(q_max_level + 1)) + 1, 4, 16))
        self.cfg.frontend.quantization_full_scale = q_full_scale_est

        # -- Optional impairments (off for measured preset) --
        self.cfg.optional.enable_gain_mismatch = False
        self.cfg.optional.enable_phase_mismatch = False

    def _inject_params_from_json(self, path: str):
        """Load a case*_params.json fit and inject it via the shared injector.

        Used by the config-driven ``hardware_profile == "fitted"``. Reuses the
        exact field injection of 
        ``validation/cir_validation_adapter.build_injected_simulator`` through
        ``impairment_params.apply_case_params_to_cfg``.

        A relative ``path`` is resolved against the project root (the parent of
        this CIRgenerator package) so it works regardless of CWD. Guards:
            * fit ``num_antennas`` must match ``cfg.num_antennas`` (hard error);
            * a ``num_taps`` mismatch vs ``cfg.cir.num_bins`` is warned but
              ``cfg.cir.num_bins`` stays authoritative (the tap-indexed
              feedthrough/ringing params assume the fit's tap grid).
        """
        if not path:
            raise ValueError(
                'hardware_profile = "fitted" requires [profiles] '
                "hardware_params_json to point at a case*_params.json fit."
            )

        json_path = Path(path)
        if not json_path.is_absolute():
            json_path = Path(__file__).resolve().parents[1] / json_path
        if not json_path.exists():
            raise FileNotFoundError(f"hardware_params_json not found: {json_path}")

        with open(json_path) as f:
            case_params = json.load(f)

        meta = case_params.get("meta", {})
        fit_antennas = meta.get("num_antennas")
        if fit_antennas is not None and int(fit_antennas) != self.cfg.num_antennas:
            raise ValueError(
                f"hardware_params_json num_antennas ({fit_antennas}) != "
                f"cfg.mnum_antennas ({self.cfg.num_antennas}): {json_path}"
            )

        fit_taps = meta.get("num_taps")
        if fit_taps is not None and int(fit_taps) != self.cfg.cir.num_bins:
            print(
                f"[CIRSimulator] WARNING: hardware_params_json was fit at "
                f"num_taps={fit_taps} but cfg.cir.num_bins={self.cfg.cir.num_bins}; "
                f"keeping cfg.cir.num_bins. Tap-indexed impairments "
                f"(feedthrough/ringing start_tap/num_taps) assume the fit's tap grid."
            )

        apply_case_params_to_cfg(self.cfg, case_params)

    def _apply_room_profile(self):
        """
        Apply room/environment-only settings.
        """
        profile = (self.cfg.room_profile or "").strip().lower()

        self.cfg.static_clutter = []

        the_room = Room(scene=self.cfg.scene)
        self.cfg.scene.radar_placement_mode = "center"
        self.cfg.scene.enable_default_room_clutter = False
        self.cfg.enable_baseline_subtraction = False
        
        if profile == "anechoic_like":
            pass

        elif profile == "small_room":
            self.cfg.static_clutter = the_room._place_walls()

        elif profile == "small_room_center":
            self.cfg.static_clutter = the_room._place_walls()
            self.cfg.scene.enable_default_room_clutter = True

            wooden_table = StaticClutterPath(
                    name="table_front",
                    position_xy_m=(1.35, 0.0),
                    amplitude=0.12,
                    width_m=0.5,
                    num_scatter_points=5,
                    orientation_deg=0.0,
                    material_type="wood",
                    material_factor=0.9
                )

            self.cfg.static_clutter.append(wooden_table)

        elif profile == "cluttered_room":
            self.cfg.static_clutter = the_room._place_walls()
            self.cfg.scene.enable_default_room_clutter = True
            self.cfg.enable_baseline_subtraction = True
            wooden_table_front = StaticClutterPath("table_front", (1.7, 0.0), amplitude=0.14, width_m=0.5,
                                                                num_scatter_points=5, orientation_deg=0.0,
                                                                material_type="wood", material_factor=0.9)
                                                            
            plastic_chair = StaticClutterPath("chair_left", (1.1, -1.55), amplitude=0.10, width_m=0.25,
                                                                num_scatter_points=3, orientation_deg=90.0,
                                                                material_type="plastic", material_factor=0.8)
            
            metal_shelf = StaticClutterPath("metal_shelf", (2.9, 0.75), amplitude=0.22, width_m=0.35,
                                                                num_scatter_points=5, orientation_deg=90.0,
                                                                material_type="metal", material_factor=1.4)
            self.cfg.static_clutter.append(wooden_table_front)
            self.cfg.static_clutter.append(plastic_chair)
            self.cfg.static_clutter.append(metal_shelf)

        else:
            raise ValueError("Unsupported room_profile.")

    def _apply_target_profile(self):
        """
        Apply target-only settings.

        Raises:
            ValueError: "Unsupported target_profile."
        """
        profile = (self.cfg.target_profile or "").strip().lower()

        self.cfg.targets = []
        #self.cfg.cir.noise_std = 0.02      # no hard-coded noise[KMG, 260616]

        if profile == "empty":
            self.cfg.targets = []

        elif profile == "static_object":
            self.cfg.targets = [
                Target(
                    name="object_1",
                    position_xy_m=(1.2, 0.25),
                    amplitude=0.9,
                    width_m=0.50,
                    num_scatter_points=0,
                    orientation_deg=90.0,
                    material_type="metal",
                    material_factor=1.4,
                    enable_micro_motion=False,
                )
            ]
            self.cfg.enable_baseline_subtraction = False

        elif profile == "static_human":
            self.cfg.targets = [
                Target(
                    name="human_1",
                    position_xy_m=(1.0, 0.2),
                    amplitude=1.0,
                    width_m=0.25,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="human",
                    material_factor=1.0,
                    enable_micro_motion=False,
                )
            ]

        elif profile == "breathing_human":
            self.cfg.targets = [
                Target(
                    name="human_1",
                    position_xy_m=(1.0, 0.2),
                    amplitude=1.0,
                    width_m=0.25,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="human",
                    material_factor=1.0,
                    enable_micro_motion=True,
                    micro_motion_amplitude_m=0.006,
                    micro_motion_frequency_hz=0.30,
                    micro_motion_phase_rad=0.0,
                    micro_motion_axis="radial",
                )
            ]

        elif profile == "breathing_human_front":
            # Same as "breathing_human" but at (1.0, 0.0) to match the
            # measured static human in data/cir/260701/case1.log (see
            # data/cir/260701/case_geometry.md: "case1 : (1m,0) static
            # human"). position_z_m stays at the Target default (0.0,
            # floor), matching every other built-in target_profile.
            self.cfg.targets = [
                Target(
                    name="human_1",
                    position_xy_m=(1.0, 0.0),
                    amplitude=1.0,
                    width_m=0.25,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="human",
                    material_factor=1.0,
                    enable_micro_motion=True,
                    micro_motion_amplitude_m=0.006,
                    micro_motion_frequency_hz=0.30,
                    micro_motion_phase_rad=0.0,
                    micro_motion_axis="radial",
                )
            ]

        elif profile == "breathing_human_torso":
            # Same as "breathing_human_front" but position_z_m=1.2, i.e.
            # the scattering center is placed at the radar height instead
            # of the floor. Used to test whether a standing human's main
            # scatter center (torso, near radar height) reproduces the
            # measured target range/tap better than the floor-level
            # convention every other target_profile branch uses.
            self.cfg.targets = [
                Target(
                    name="human_1",
                    position_xy_m=(1.0, 0.0),
                    position_z_m=1.2,
                    amplitude=1.0,
                    width_m=0.25,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="human",
                    material_factor=1.0,
                    enable_micro_motion=True,
                    micro_motion_amplitude_m=0.006,
                    micro_motion_frequency_hz=0.30,
                    micro_motion_phase_rad=0.0,
                    micro_motion_axis="radial",
                )
            ]

        elif profile == "human_plus_object":
            self.cfg.targets = [
                Target(
                    name="human_1",
                    position_xy_m=(1.0, -0.2),
                    amplitude=1.0,
                    width_m=0.25,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="human",
                    material_factor=1.0,
                    enable_micro_motion=True,
                    micro_motion_amplitude_m=0.006,
                    micro_motion_frequency_hz=0.30,
                    micro_motion_phase_rad=0.0,
                    micro_motion_axis="radial",
                ),
                Target(
                    name="object_1",
                    position_xy_m=(1.5, 0.25),
                    amplitude=0.95,
                    width_m=0.18,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="metal",
                    material_factor=1.5,
                    enable_micro_motion=False,
                )
            ]
            # self.cfg.cir.noise_std = 0.03

        elif profile == "moving_human":
            self.cfg.targets = [
                Target(
                    name="human_1",
                    position_xy_m=(1.0, 0.2),
                    amplitude=1.0,
                    velocity_xy_m_per_frame=(0.005, 0.0), # unit: m/second for per-frame
                    width_m=0.25,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="human",
                    material_factor=1.0,
                    enable_micro_motion=False,
                )
            ]


        elif profile == "multi_human":
            self.cfg.targets = [
                Target(
                    name="human_1",
                    position_xy_m=(1.0, -0.35),
                    amplitude=1.0,
                    width_m=0.25,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="human",
                    material_factor=1.0,
                    enable_micro_motion=False,
                ),
                Target(
                    name="human_2",
                    position_xy_m=(1.3, 0.0),
                    amplitude=1.0,
                    width_m=0.25,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="human",
                    material_factor=1.0,
                    enable_micro_motion=False,
                ),
                Target(
                    name="human_3",
                    position_xy_m=(1.6, 0.35),
                    amplitude=1.0,
                    width_m=0.25,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="human",
                    material_factor=1.0,
                    enable_micro_motion=False,
                ),
            ]

        elif profile == "moving_object":
            self.cfg.targets = [
                Target(
                    name="object_1",
                    position_xy_m=(1.2, 0.25),
                    amplitude=0.9,
                    velocity_xy_m_per_frame=(0.008, 0.0),
                    width_m=0.18,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="metal",
                    material_factor=1.4,
                    enable_micro_motion=False,
                )
            ]

        elif profile == "metal_false_reflector":
            self.cfg.targets = [
                Target(
                    name="false_reflector_1",
                    position_xy_m=(1.7, -0.3),
                    amplitude=1.2,
                    width_m=0.12,
                    num_scatter_points=3,
                    orientation_deg=90.0,
                    material_type="metal",
                    material_factor=1.8,
                    enable_micro_motion=False,
                )
            ]

        else:
            raise ValueError("Unsupported target_profile.")
    
    def _sync_target_frame_period(self):
        """
        Sync the slow-time frame period from the radar config to all targets.

        Target._apply_target_micro_motion() uses self.frame_period_s to convert
        frame indices into seconds. This method ensures that value matches the
        simulator's cfg.radar.period (in ms).
        """
        period_s = self.cfg.radar.period / 1000.0
        for t in self.cfg.targets:
            if isinstance(t, Target):
                t.frame_period_s = period_s

    def _get_frame_time_s(self, frame_idx: int) -> float:

        """Convert frame index to time in seconds using radar period in ms.

        Args:
            frame_idx (int): frame index

        Returns:
            (float): seconds
        """
        #

        period_ms = float(self.cfg.radar.period)
        return frame_idx * (period_ms / 1000.0)


    def _distance(self, target_pos: np.ndarray, antenna_pos: np.ndarray) -> float:
        """Calculates the 3D distance between target_pos and antenna_pos.

        Uses the common distance_3d() helper, which upgrades legacy 2D
        (x, y) inputs to 3D (z=0.0) automatically.

        Args:
            target_pos (np.ndarray): target's position from which to calculate the norm
            antenna_pos (np.ndarray): antennas position from which to calculate the norm

        Returns:
            float: value of the distance
        """
        return distance_3d(target_pos, antenna_pos)

    def _radar_equation_amplitude_factor(
        self,
        d_tx_m: float,
        d_rx_m: float,
        rcs_m2: float,
        path_loss_exp: float,
    ) -> float:
        """Round-trip radar-equation amplitude factor for a scattered echo.

        This is the TX -> scatterer -> RX (two-way) law, written in the general
        bistatic form:
        
            amplitude ~ lambda_m * sqrt(sigma) / ((4*pi)**1.5 * Rt * Rr)
                        * (sqrt(Rt*Rr) / d0)**-(n - 2.0)

        Derivation: the radar equation gives received POWER
        Pr/Pt ~ sigma / (Rt**2 * Rr**2); the CIR tap is a complex VOLTAGE
        transfer coefficient (|h|**2 == power), so |h| ~ sqrt(Pr/Pt)
        ~ sqrt(sigma) / (Rt*Rr), with the free-space aperture term
        contributing lambda_m**1 (NOT lambda_m**2) and the normalisation
        (4*pi)**1.5.

        Monostatic (collocated TX/RX, Rt == Rr == d) reduces exactly to
        lambda_m*sqrt(sigma) / ((4*pi)**1.5 * d**2), i.e. amplitude ~ d**-2.
        TX and RX are collocated by construction in this engine (both derive
        from self.antenna_positions[ant_idx]), so both call sites currently
        pass Rt == Rr; the bistatic signature is kept so a future distinct TX
        position needs no change here.

        NOTE: this replaced a one-way Friis POWER term
        (lambda_m/(4*pi*d))**2, which had the right d exponent for the wrong
        reason (a squared one-way term rather than a product of two one-way
        voltage terms) and the wrong wavelength exponent (lambda**2). At the
        default carrier frequency with sigma = 1 m^2 the new law reproduces the
        old absolute amplitude, because path_loss_ref_gain was recalibrated
        alongside it - see docs/friis_path_loss_model.md.

        The excess-loss exponent is unchanged in meaning: n == 2.0 is the exact
        radar equation, n > 2.0 adds empirical log-distance excess loss beyond
        d0. It uses the geometric mean sqrt(Rt*Rr) so the monostatic case is
        identical to the previous (d_eff/d0) form.

        Uses the same self.lambda_m as the phase term (no second frequency
        knob). The max(d, path_loss_floor_m) clamp is a non-physical
        singularity guard inherited from the legacy inline max(d, 0.1); it
        bounds the d -> 0 divergence and is NOT a near-field model.

        Args:
            d_tx_m (float): TX-to-scatterer distance Rt, metres.
            d_rx_m (float): scatterer-to-RX distance Rr, metres.
            rcs_m2 (float): radar cross-section sigma, square metres.
            path_loss_exp (float): exponent n (2.0 = exact radar equation;
                >2.0 = extra excess loss beyond path_loss_ref_distance_m).

        Returns:
            float: unitless linear amplitude multiplier.
        """
        d0 = self.cfg.antenna.path_loss_ref_distance_m
        floor_m = self.cfg.antenna.path_loss_floor_m
        dt_eff = max(d_tx_m, floor_m)
        dr_eff = max(d_rx_m, floor_m)

        radar_term = (
            self.lambda_m * np.sqrt(max(rcs_m2, 0.0))
            / ((4.0 * np.pi) ** 1.5 * dt_eff * dr_eff)
        )
        excess_term = (np.sqrt(dt_eff * dr_eff) / d0) ** (-(path_loss_exp - 2.0))

        return self.cfg.antenna.path_loss_ref_gain * radar_term * excess_term

    def _distance_to_bin(self, round_trip_distance_m: float) -> float:
        delay_s = round_trip_distance_m / speed_of_light
        return delay_s / self.cfg.cir.bin_time_s

    def _deposit_fractional_bin(self, cir_vector: np.ndarray, bin_idx: float, value: complex):
        """
        Deposit a complex sample into CIR bins.

        First perform fractional interpolation between the 2 nearest bins.
        Then optionally apply a small pulse-spreading kernel so the energy

        Args:
            cir_vector (np.ndarray): CIR array.
            bin_idx (float): BIN index.
            value (complex): value to deposit in the bins.

        Returns:
            None
        """
        i0 = int(np.floor(bin_idx))
        frac = bin_idx - i0
        i1 = i0 + 1

        if not self.cfg.frontend.enable_pulse_spreading:
            if 0 <= i0 < len(cir_vector):
                cir_vector[i0] += value * (1.0 - frac)
            if 0 <= i1 < len(cir_vector):
                cir_vector[i1] += value * frac
            return
        
        kernel = np.array(self.cfg.frontend.pulse_spread_kernel, dtype=float)

        # Normalize kernel so energy stays controlled
        kernel_sum = np.sum(kernel)
        if kernel_sum <= 0:
            kernel = np.array([1.0], dtype=float)
            kernel_sum = 1.0
        kernel = kernel / kernel_sum

        center = len(kernel) // 2

        # Split between i0 and i1 first, then spread each part
        for base_idx, base_weight in [(i0, 1.0 - frac), (i1, frac)]:
            base_value = value * base_weight

            for k, kw in enumerate(kernel):
                idx = base_idx + (k - center)
                if 0 <= idx < len(cir_vector):
                    cir_vector[idx] += base_value * kw

    def _apply_antenna_impairments(self, antenna_idx: int, sample: complex) -> complex:
        """
        Apply optional per-antenna gain and phase mismatch.

        Args:
            antenna_idx (int): antenna index.
            sample (complex): sample value.

        Returns:
            (complex): impairment value.
        """
        if self.cfg.optional.enable_gain_mismatch and self.cfg.optional.antenna_gain_mismatch is not None:
            sample *= self.cfg.optional.antenna_gain_mismatch[antenna_idx]

        if self.cfg.optional.enable_phase_mismatch and self.cfg.optional.antenna_phase_mismatch_rad is not None:
            sample *= np.exp(1j * self.cfg.optional.antenna_phase_mismatch_rad[antenna_idx])

        return sample

    
    def _apply_rx_gain(self, cir_frame: np.ndarray) -> np.ndarray:
        """
        Apply simplified effective RX gain.

        Use 30 dB as a reference point so gain differences remain realistic
        and do not explode numerically.

        Args:
            cir_frame (np.ndarray): CIR array.

        Returns:
            (np.ndarray): rx gain
        """
        out = cir_frame.copy()

        for ot, rx_gain in zip(out, self.cfg.radar.rx_gains):
            ot *= db_to_linear_amplitude(rx_gain - np.mean(self.cfg.radar.rx_gains))
        return out

    def _get_frame_drift_params(self):
        """
        Generate small frame-level drift parameters.

        Returns:
            amp_scale (float): common amplitude scaling for this frame.
            phase_ant (float): small phase drift for antenna
            timing_jitter_bins (float): small common delay jitter in bins.
        """
        if not self.cfg.frontend.enable_frame_drift:
            return 1.0, [0.0] * self.cfg.num_antennas, 0.0 # bug fix: parameter unpack error [KMG, 260616]

        amp_scale = 1.0 + self.rng.normal(0.0, self.cfg.frontend.frame_amplitude_drift_std)
        timing_jitter_bins = self.rng.normal(0.0, self.cfg.frontend.frame_timing_jitter_std_bins)
        phase_ant = []
        for value in self.cfg.frontend.frame_phase_drift_std:
            phase_ant.append(self.rng.normal(0.0, value))


        return amp_scale, phase_ant, timing_jitter_bins

    def _apply_adc_clipping(self, cir_frame: np.ndarray) -> np.ndarray:
        """
        Simulate ADC compression + clipping.

        Soft limit:
            gradual compression starts

        Hard limit:
            absolute clipping threshold

        Args:
            cir_frame (np.ndarray): CIR array.

        Returns:
            (np.ndarray): clipped value of CIR.
        """
        if not self.cfg.frontend.enable_adc_clipping:
            return cir_frame

        out = cir_frame.copy()

        soft = float(self.cfg.frontend.adc_soft_limit)
        hard = float(self.cfg.frontend.adc_hard_limit)

        for ant in range(out.shape[0]):
            for i in range(out.shape[1]):
                sample = out[ant, i]

                mag = np.abs(sample)
                phase = np.angle(sample)

                # soft compression region
                if mag > soft:
                    mag = soft + (mag - soft) * 0.35

                # hard clipping region
                if mag > hard:
                    mag = hard

                out[ant, i] = mag * np.exp(1j * phase)

        return out

    def _apply_quantization(self, cir_frame: np.ndarray) -> np.ndarray:
        """
        Apply simple fixed-point quantization to the complex CIR.

        Real and imaginary parts are quantized separately using a signed
        uniform quantizer with configurable bit depth and full-scale range.

        Args:
            cir_frame (np.ndarray): CIR array.

        Returns:
            (np.ndarray): quantizited CIR.
        """
        if not self.cfg.frontend.enable_quantization:
            return cir_frame

        out = cir_frame.copy()

        bits = int(self.cfg.frontend.quantization_bits)
        full_scale = float(self.cfg.frontend.quantization_full_scale)

        if bits < 2:
            return out

        qmax = (2 ** (bits - 1)) - 1
        qmin = -(2 ** (bits - 1))
        
        # Quantization step
        step = full_scale / max(qmax, 1)

        for ant in range(out.shape[0]):
            for i in range(out.shape[1]):
                sample = out[ant, i]

                real_val = np.real(sample) + self.cfg.frontend.quantization_dc_offset_i
                imag_val = np.imag(sample) + self.cfg.frontend.quantization_dc_offset_q

                # Clip to quantizer range
                real_val = np.clip(real_val, -full_scale, full_scale)
                imag_val = np.clip(imag_val, -full_scale, full_scale)

                # Quantize to integer grid then map back to float grid
                real_q = np.round(real_val / step)
                imag_q = np.round(imag_val / step)

                real_q = np.clip(real_q, qmin, qmax)
                imag_q = np.clip(imag_q, qmin, qmax)

                out[ant, i] = (real_q * step) + 1j * (imag_q * step)

        return out

    def _generate_baseline_frame(self):
        """
        Generate empty-room baseline frame.

        This includes:
        - static clutter
        - wall reflections
        - table reflections
        - leakage
        - hardware effects

        This excludes:
        - human targets

        Radar ego motion: the baseline is generated at frame_idx=0, so it is
        frozen at the radar pose of the FIRST slow-time sample and does not
        track the tremor. That is deliberate and physical -- a real device's
        clutter/baseline estimate cannot follow the user's hand either, and the
        resulting mismatch is exactly the artifact this feature exists to
        reproduce. For handheld_jitter that first pose is the configured base
        pose exactly (the per-component anchoring makes delta(t_0) = 0); for
        sampled_pose it is whatever the trace holds at the first grid instant.
        """

        #save targets lists
        original_targets = self.cfg.targets
        self.cfg.targets = []
        #empty rrom
        baseline_frame, _ = self._generate_one_frame(frame_idx=0)
        #restore targets
        self.cfg.targets = original_targets

        return baseline_frame

    def _apply_radiation_leakage(self, cir_frame: np.ndarray) -> np.ndarray:
        """
        Simulate leakage through air between two antennas.

        Typical factors:
            0.02 to 0.05 = mild
            0.05 to 0.10 = moderate
            0.10+        = strong

        Args:
            cir_frame (np.ndarray): CIR array.
        Returns:
            (np.ndarray): radiation leakage.
        """
        if not self.cfg.leakage.enable_radiation_leakage:
            return cir_frame

        if cir_frame.shape[0] != 2:
            return cir_frame

        out = cir_frame.copy()
        src = cir_frame.copy()
        delay = int(self.cfg.leakage.radiation_delay_bins)

        ant1_from_ant2 = src[1].copy()
        ant2_from_ant1 = src[0].copy()

        if delay > 0:
            delayed_12 = np.zeros_like(ant1_from_ant2)
            delayed_21 = np.zeros_like(ant2_from_ant1)
            delayed_12[delay:] = ant1_from_ant2[:-delay]
            delayed_21[delay:] = ant2_from_ant1[:-delay]
            ant1_from_ant2 = delayed_12
            ant2_from_ant1 = delayed_21

        ant1_from_ant2 *= (
            self.cfg.leakage.radiation_leakage_factor_12
            * np.exp(1j * self.cfg.leakage.radiation_phase_offset_12_rad)
        )
        ant2_from_ant1 *= (
            self.cfg.leakage.radiation_leakage_factor_21
            * np.exp(1j * self.cfg.leakage.radiation_phase_offset_21_rad)
        )

        out[0] += ant1_from_ant2
        out[1] += ant2_from_ant1
        return out

    def _apply_pcb_leakage(self, cir_frame: np.ndarray) -> np.ndarray:
        """
        Simulate leakage through PCB/routing/ground path.

        Typical factors:
            0.02 to 0.08 = small
            0.10 to 0.20 = suspicious
            0.20+        = strong / bad board test

        Args:
            cir_frame (np.ndarray): CIR array.
        Returns:
            (np.ndarray): PCB leakage.
        """
        if not self.cfg.leakage.enable_pcb_leakage:
            return cir_frame

        if cir_frame.shape[0] != 2:
            return cir_frame

        out = cir_frame.copy()
        src = cir_frame.copy()
        delay = int(self.cfg.leakage.pcb_delay_bins)

        ant1_from_ant2 = src[1].copy()
        ant2_from_ant1 = src[0].copy()

        if delay > 0:
            delayed_12 = np.zeros_like(ant1_from_ant2)
            delayed_21 = np.zeros_like(ant2_from_ant1)
            delayed_12[delay:] = ant1_from_ant2[:-delay]
            delayed_21[delay:] = ant2_from_ant1[:-delay]
            ant1_from_ant2 = delayed_12
            ant2_from_ant1 = delayed_21

        ant1_from_ant2 *= (
            self.cfg.leakage.pcb_leakage_factor_12
            * np.exp(1j * self.cfg.leakage.pcb_phase_offset_12_rad)
        )
        ant2_from_ant1 *= (
            self.cfg.leakage.pcb_leakage_factor_21
            * np.exp(1j * self.cfg.leakage.pcb_phase_offset_21_rad)
        )

        out[0] += ant1_from_ant2
        out[1] += ant2_from_ant1
        return out

    def _apply_tx_rx_feedthrough(self, cir_frame: np.ndarray) -> np.ndarray:
        """
        Simulate direct TX -> RX feedthrough / near-field antenna coupling.

        In this project:
        - antenna 0 = RX1 only
        - antenna 1 = RX2 + TX

        So TX-originated leakage should be strongest on the TX-own RX chain,
        and somewhat smaller on the other RX chain.
        Args:
            cir_frame (np.ndarray): CIR array.
        Returns:
            (np.ndarray): tx-rx feedthrough.
        """
        if not self.cfg.leakage.enable_tx_rx_feedthrough or cir_frame.shape[0] < 2:
            return cir_frame

        out = cir_frame.copy()
        num_ant, num_bins = out.shape

        start_tap = max(0, int(self.cfg.leakage.tx_rx_feedthrough_start_tap))
        n_taps = min(int(self.cfg.leakage.tx_rx_feedthrough_num_taps), num_bins - start_tap)

        if n_taps <= 0:
            return out

        tx_idx = int(np.clip(self.cfg.leakage.tx_antenna_index, 0, num_ant - 1))
        other_idx = 1 - tx_idx

        # Per-tap phase slope of the feedthrough (structured-schema fit). Defaults
        # to 0.0 so every non-structured caller is byte-identical to before.
        phase_slope0 = getattr(self.cfg.leakage, "tx_rx_feedthrough_phase_slope_ant0", 0.0)
        phase_slope1 = getattr(self.cfg.leakage, "tx_rx_feedthrough_phase_slope_ant1", 0.0)

        for k in range(n_taps):
            tap = start_tap + k
            
            # Leakage into RX1-only chain
            weight0 = self.cfg.leakage.tx_rx_feedthrough_decay_ant0 ** k
            ripple0 = 1.0 + self.cfg.leakage.tx_rx_feedthrough_ripple_amp_ant0 * np.cos(
                self.cfg.leakage.tx_rx_feedthrough_ripple_freq_ant0 * k
            )
            rand0 = 1.0 + self.rng.normal(0.0, getattr(self.cfg.leakage, "tx_rx_feedthrough_random_std_ant0", 0.0))

            leak_to_ant0 = (
                self.cfg.leakage.tx_rx_feedthrough_amp_ant0 * weight0 * ripple0 * rand0 * np.exp(1j * (self.cfg.leakage.tx_rx_feedthrough_phase_ant0_rad + phase_slope0 * k))
            )

            # Leakage into RX2+TX chain
            weight1 = self.cfg.leakage.tx_rx_feedthrough_decay_ant1 ** k
            ripple1 = 1.0 + self.cfg.leakage.tx_rx_feedthrough_ripple_amp_ant1 * np.cos(
                self.cfg.leakage.tx_rx_feedthrough_ripple_freq_ant1 * k
            )
            rand1 = 1.0 + self.rng.normal(0.0, getattr(self.cfg.leakage, "tx_rx_feedthrough_random_std_ant1", 0.0))

            leak_to_ant1 = (
                self.cfg.leakage.tx_rx_feedthrough_amp_ant1 * weight1 * ripple1 * rand1 * np.exp(1j * (self.cfg.leakage.tx_rx_feedthrough_phase_ant1_rad + phase_slope1 * k))
            )

            out[0, tap] += leak_to_ant0
            out[1, tap] += leak_to_ant1

        return out

    def _apply_tx_rx_feedthrough_ringing(self, cir_frame: np.ndarray) -> np.ndarray:
        """
        Simulate the damped ringing tail that follows the main TX-RX
        feedthrough peak (matched-filter / pulse-shaping ringing), independent
        of target reflection or antenna-to-antenna radiation coupling.

        Model per antenna:
            S[a,k] = offset_amp_a * exp(j*offset_phase_a)
                   + ring_amp_a * decay_a**k * exp(j*(ring_phase_a + freq_a*k))
        where k is measured from tx_rx_feedthrough_ringing_start_tap.

        Args:
            cir_frame (np.ndarray): CIR array.
        Returns:
            (np.ndarray): tx-rx feedthrough ringing tail.
        """
        if not self.cfg.leakage.enable_tx_rx_feedthrough_ringing or cir_frame.shape[0] < 2:
            return cir_frame
        
        out = cir_frame.copy()
        num_ant, num_bins = out.shape

        start_tap = max(0, int(self.cfg.leakage.tx_rx_feedthrough_ringing_start_tap))
        n_taps = min(int(self.cfg.leakage.tx_rx_feedthrough_ringing_num_taps), num_bins - start_tap)

        if n_taps <= 0:
            return out

        for k in range(n_taps):
            tap = start_tap + k

            ring0 = (
                self.cfg.leakage.tx_rx_feedthrough_ringing_offset_amp_ant0
                * np.exp(1j * self.cfg.leakage.tx_rx_feedthrough_ringing_offset_phase_ant0_rad)
                + self.cfg.leakage.tx_rx_feedthrough_ringing_amp_ant0
                * (self.cfg.leakage.tx_rx_feedthrough_ringing_decay_ant0 ** k)
                * np.exp(1j * (
                    self.cfg.leakage.tx_rx_feedthrough_ringing_phase_ant0_rad
                    + self.cfg.leakage.tx_rx_feedthrough_ringing_freq_ant0 * k
                ))
            )

            ring1 = (
                self.cfg.leakage.tx_rx_feedthrough_ringing_offset_amp_ant1
                * np.exp(1j * self.cfg.leakage.tx_rx_feedthrough_ringing_offset_phase_ant1_rad)
                + self.cfg.leakage.tx_rx_feedthrough_ringing_amp_ant1
                * (self.cfg.leakage.tx_rx_feedthrough_ringing_decay_ant1 ** k)
                * np.exp(1j * (
                    self.cfg.leakage.tx_rx_feedthrough_ringing_phase_ant1_rad
                    + self.cfg.leakage.tx_rx_feedthrough_ringing_freq_ant1 * k
                ))
            )

            out[0, tap] += ring0
            out[1, tap] += ring1

        return out


    def _apply_early_leakage(self, cir_frame: np.ndarray) -> np.ndarray:
        """
        Simulate structured early-tap leakage caused by antenna coupling.
        front-end leakage, PCB/internal ringing, or direct TX-RX feedthrough.

        This effect appears at the beginning of the CIR and does NOT
        depend on target distance.

        Compared with the simple version, this model adds:
            - antenna-specific decay
            - small ripple over taps
            - small random per-tap variation

        Args:
            cir_frame (np.ndarray): CIR array.

        Returns:
            (np.ndarray): early leakage.
        """

        num_ant, num_bins = cir_frame.shape

        if not self.cfg.leakage.enable_early_leakage or num_ant < 2:
            return cir_frame

        out = cir_frame.copy()

        n_taps = min(int(self.cfg.leakage.early_leakage_num_taps), num_bins)
        for tap in range(n_taps):
            weight0 = self.cfg.leakage.early_leakage_decay_ant0 ** tap
            ripple0 = 1.0 + self.cfg.leakage.early_leakage_ripple_amp_ant0 * np.cos(
                self.cfg.leakage.early_leakage_ripple_freq_ant0 * tap
            )
            rand0 = 1.0 + self.rng.normal(0.0, self.cfg.leakage.early_leakage_random_std_ant0)

            leak0 = (
                self.cfg.leakage.early_leakage_amplitude_ant0 * weight0 * ripple0 * rand0 * np.exp(1j * self.cfg.leakage.early_leakage_phase_ant0_rad)
            )

            weight1 = self.cfg.leakage.early_leakage_decay_ant1 ** tap
            ripple1 = 1.0 + self.cfg.leakage.early_leakage_ripple_amp_ant1 * np.cos(
                self.cfg.leakage.early_leakage_ripple_freq_ant1 * tap
            )
            rand1 = 1.0 + self.rng.normal(0.0, self.cfg.leakage.early_leakage_random_std_ant1)

            leak1 = (
                self.cfg.leakage.early_leakage_amplitude_ant1 * weight1 * ripple1 * rand1 * np.exp(1j * self.cfg.leakage.early_leakage_phase_ant1_rad)
            )

            out[0, tap] += leak0
            out[1, tap] += leak1
    
        return out

    


    def _generate_one_frame(self, frame_idx: int):
        """Main simulation function. Will calculate target distance relative for each antenna, calculate time delay, amplitude and phase of the reflected signal.
        Will apply impairments, clutter, and leakage if present in configuration.

        Args:
            frame_idx (int): frame index.

        Returns:
            Tuple: CIR for the frame and ground_truth dictionary.
        """
        num_bins = self.cfg.cir.num_bins

        cir_frame = np.zeros((self.cfg.num_antennas, self.cfg.cir.num_bins), dtype=np.complex128)
        ground_truth = []

        frame_amp_scale, frame_phase_ant, frame_timing_jitter = self._get_frame_drift_params()

        colliders = self.cfg.targets + self.cfg.static_clutter
        
        # Radar ego motion: geometry for THIS slow-time sample. Both calls
        # short-circuit to the legacy static values (the same array object, in
        # the case of the antenna array) when no motion is configured.
        radar_origin = self._radar_origin_at_frame(frame_idx)
        antenna_positions_frame = self._antenna_positions_at_frame(frame_idx)

        for collider in colliders:
            if isinstance(collider, Target):
                collider_center = collider._apply_target_micro_motion(frame_idx)
                path_loss_exp = self.cfg.antenna.target_path_loss_exp

                # Project angle convention: phi=azimuth, theta=polar angle
                # from +z (matches HFSS/standard spherical convention).
                # Preferred order everywhere is (phi, theta).
                phi_deg, theta_deg = point_to_phi_theta_deg(radar_origin, collider_center)
                distance_center_m = distance_3d(radar_origin, collider_center)

                ground_truth.append(
                    collider._get_ground_truth(
                        frame_idx,
                        collider_center,
                        phi_deg,
                        theta_deg,
                        distance_center_m=distance_center_m,
                    )
                )
            else:
                collider_center = collider._get_center_position(frame_idx)
                path_loss_exp = self.cfg.antenna.clutter_path_loss_exp

            reflection_path_loss_exp = self.cfg.antenna.reflection_path_loss_exp

            scatter_points = collider._get_scatter_points(frame_idx)
            # The collider's total RCS is split equally over its scatter points:
            # each point scatters sigma/N, so its voltage contribution carries
            # sqrt(sigma/N) via _radar_equation_amplitude_factor. This keeps the
            # incoherent (energy) sum equal to sigma regardless of N -- RCS is an
            # energy-like quantity. The previous 1/N AMPLITUDE normalisation made
            # total energy fall as 1/N, so a finely-discertised target was
            # silently dimmer than a coarse one.
            rcs_share_m2 = collider.rcs_m2 / max(len(scatter_points), 1)

            for ant_idx, ant_pos in enumerate(antenna_positions_frame):
                for scatter_pt in scatter_points:

                    d_direct = self._distance(scatter_pt, ant_pos)
                    # TODO: replace monostatic approximation with bistatic
                    # TX->scatter->RX path in a later step.
                    round_trip_direct = 2.0 * d_direct

                    pattern_gain = self.cfg.antenna._antenna_pattern_gain(ant_pos, scatter_pt, ant_idx=ant_idx)
                    # Two-way antenna gain. The radar equation needs
                    # sqrt(G_tx * G_rx) in the voltage domain, where G_* are
                    # POWER gains. _antenna_pattern_gain returns an AMPLITUDE
                    # gain 10**(dBi/20) == sqrt(G_power), and TX/RX are
                    # collocated with the same pattern here, so
                    # sqrt(G_tx*G_rx) == pattern_gain**2. Applying it once (the
                    # previous behaviour) modelled a single aperture and
                    # under-weighted angular roll-off by exactly 2x in dB.
                    two_way_pattern_gain = pattern_gain ** 2
                    # TX/RX are collocated here, so Rt == Rr == d_direct (the
                    # monostatic special case of the bistatic radar equation).
                    direct_amp = ( collider.amplitude * collider.material_factor * two_way_pattern_gain) * self._radar_equation_amplitude_factor(d_direct, d_direct, rcs_share_m2, path_loss_exp)

                    direct_phase = -2.0 * np.pi * round_trip_direct / self.lambda_m
                    direct_phase += frame_phase_ant[ant_idx]

                    direct_sample = (direct_amp * frame_amp_scale) * np.exp(1j * direct_phase)
                    direct_sample = self._apply_antenna_impairments(ant_idx, direct_sample)

                    direct_bin = self._distance_to_bin(round_trip_direct) + frame_timing_jitter
                    self._deposit_fractional_bin(cir_frame[ant_idx], direct_bin, direct_sample)

                    for refl in collider.reflections:
                        round_trip_refl = round_trip_direct + refl.extra_distance_m

                        pattern_gain = self.cfg.antenna._antenna_pattern_gain(ant_pos, scatter_pt, ant_idx=ant_idx)
                        # Same two-way gain reasoning as the direct path above.
                        two_way_pattern_gain = pattern_gain ** 2
                        # One-way-equivalent leg length for the bounced path;
                        # Rt == Rr keeps the monostatic-equivalent convention
                        # already used for this path's delay and phase.
                        refl_leg_m = round_trip_refl / 2.0
                        refl_amp = (
                            collider.amplitude * collider.material_factor * refl.attenuation * two_way_pattern_gain
                        ) * self._radar_equation_amplitude_factor(refl_leg_m, refl_leg_m, rcs_share_m2, reflection_path_loss_exp)

                        refl_phase = (
                            -2.0 * np.pi * round_trip_refl / self.lambda_m
                            + refl.phase_offset_rad
                        )

                        refl_phase += frame_phase_ant[ant_idx]

                        refl_sample = (refl_amp * frame_amp_scale) * np.exp(1j * refl_phase)
                        refl_sample = self._apply_antenna_impairments(ant_idx, refl_sample)

                        refl_bin = self._distance_to_bin(round_trip_refl) + frame_timing_jitter
                        self._deposit_fractional_bin(cir_frame[ant_idx], refl_bin, refl_sample)
        

        cir_frame = self._apply_tx_rx_feedthrough(cir_frame)
        cir_frame = self._apply_tx_rx_feedthrough_ringing(cir_frame)
        cir_frame = self._apply_radiation_leakage(cir_frame)
        cir_frame = self._apply_pcb_leakage(cir_frame)
        cir_frame = self._apply_early_leakage(cir_frame)

        # Apply RX gain after propagation/leakage effects
        cir_frame = self._apply_rx_gain(cir_frame)
        cir_frame = self._apply_adc_clipping(cir_frame)

        noise_std_per_ant = self.cfg.cir.noise_std_per_ant
        amp_proportional_factor = self.cfg.cir.noise_amp_proportional_factor

        if noise_std_per_ant is not None:
            if len(noise_std_per_ant) != self.cfg.num_antennas:
                raise ValueError(
                    f"cfg.cir.noise_std_per_ant has {len(noise_std_per_ant)} values, "
                    f"expected {self.cfg.num_antennas} (cfg.num_antennas)."
                )
            if amp_proportional_factor is not None and len(amp_proportional_factor) != self.cfg.num_antennas:
                raise ValueError(
                    f"cfg.cir.noise_amp_proportional_factor has {len(amp_proportional_factor)} values, "
                    f"expected {self.cfg.num_antennas} (cfg.num_antennas)."
                )
            noise_real = np.empty_like(cir_frame, dtype=float)
            noise_imag = np.empty_like(cir_frame, dtype=float)
            for ant, std in enumerate(noise_std_per_ant):
                std = max(std, 1e-6)    # same tiny-noise floor as the scalar-off branch below
                if amp_proportional_factor is not None:
                    # sigma[k] = sqrt(flat_std**2 + (factor * |cir_frame[ant,k]|)**2):
                    # models frame-to-frame fluctuation that scales with tap
                    # amplitude (e.g. TX-RX feedthrough/ringing tail), on top of
                    # the existing flat noise floor.
                    sigma_map = np.sqrt(std ** 2 + (amp_proportional_factor[ant] * np.abs(cir_frame[ant])) ** 2)
                    noise_real[ant] = self.rng.normal(0, sigma_map)
                    noise_imag[ant] = self.rng.normal(0, sigma_map)
                else:
                    noise_real[ant] = self.rng.normal(0, std, size=cir_frame.shape[1:])
                    noise_imag[ant] = self.rng.normal(0, std, size=cir_frame.shape[1:])
            # No RX-gain rescale here: fitted per-antenna values are already the
            # target per-antenna noise; applying the rescale on top would double-scale.

        elif self.cfg.cir.noise_std > 0:
            noise_real = self.rng.normal(0, self.cfg.cir.noise_std, size=cir_frame.shape)
            noise_imag = self.rng.normal(0, self.cfg.cir.noise_std, size=cir_frame.shape)

            # Optional simple per-antenna noise scaling with effective RX gain
            ref_gain_db = 0.5 * (self.cfg.radar.rx1_gain + self.cfg.radar.rx2_gain)

            if cir_frame.shape[0] >= 1:
                n0 = db_to_linear_amplitude(self.cfg.radar.rx1_gain - ref_gain_db)
                noise_real[0] *= n0
                noise_imag[0] *= n0

            if cir_frame.shape[0] >= 2:
                n1 = db_to_linear_amplitude(self.cfg.radar.rx2_gain - ref_gain_db)
                noise_real[1] *= n1
                noise_imag[1] *= n1
        else:
            noise_real = self.rng.normal(0, 1e-6, size=cir_frame.shape)
            noise_imag = self.rng.normal(0, 1e-6, size=cir_frame.shape)
        for ant in range(self.cfg.num_antennas):    #robustness for various number of antennas
            cir_frame[ant] += noise_real[ant] + 1j * noise_imag[ant]

        #this could hide some information so is disabled
        # cir_frame = self._apply_quantization(cir_frame)

        return cir_frame, ground_truth

    def generate(self, plot=True):
        """
        Generate full CIR dataset.

        Args:
            plot (bool): If True, call plot_total() after generation. Set False
                         to skip plotting when generate() is called in a loop
                         (e.g. velocity sweep) to avoid redundant figure overhead.

        Returns:
            cir_data (shape [num_frames, num_antennas, num_bins]): all calculated CIRs.
            truth (List[dict]): list of per-frame target metadata.
        """
        from exporter import plot_total

        num_frames = self.cfg.cir.num_frames
        M = 2.5
        cir_data = np.zeros((num_frames, self.cfg.num_antennas, self.cfg.cir.num_bins), dtype=np.complex128)
        truth = []

        baseline_frame = None

        if self.cfg.enable_baseline_subtraction:
            baseline_frame = self._generate_baseline_frame()

        for frame_idx in range(num_frames):
            cir_frame, frame_truth = self._generate_one_frame(frame_idx)

            #this will substract the environment from the detection
            if self.cfg.enable_baseline_subtraction and baseline_frame is not None:
                cond = (cir_frame - baseline_frame)/baseline_frame > M
                cir_frame =  np.where(cond, cir_frame, 0)

            cir_data[frame_idx] = cir_frame
            truth.append(frame_truth)
        
        if plot:
            plot_total(
                cir_data=cir_data,
                cfg=self.cfg,
                antennas_position=self.antenna_positions,
                truth=truth,
                selected_level=self.cfg.level,
                save_dir="outputs/CIR_figures",
                show_plots=False
            )

        return cir_data, truth


if __name__ == "__main__":
    SELECTED_LEVEL = 2
    cfg = create_default_config()

    # Optional named scenario overrides level defaults
    # Examples:
    # "static_human"
    # "breathing_human"
    # "moving_human"
    # "multi_human"
    # "static_object"
    # "moving_object"
    # "metal_false_reflector"
    # "human_vs_object" mixed
    # "empty"

    cfg.scenario_name = "moving_object_real_room"
    cfg.hardware_profile = "ua200_realistic_v1"
    cfg.room_profile = "cluttered_room"
    cfg.target_profile = "moving_object"
    cfg.output_root = "output_cases"
    # Build simulator AFTER selecting profiles
    simulator = CIRSimulator(cfg)

    # Print the final config actually used in this run
    # print_selected_run_config(cfg)

    # Run simulation
    cir_data, truth = simulator.generate()
    print_strongest_taps(cir_data, frame_idx=0, taps_to_check=64)

    print(f"Generated CIR data shape: {cir_data.shape}")
    print_truth_summary(truth, max_frames=5)

    exporter = CIRLogExporter(cfg)
    paths = exporter.save_case(cir_data, truth)

    fig_paths = save_all_figures(paths["case_dir"], cfg, simulator.antenna_positions, cir_data, truth, selected_level=SELECTED_LEVEL)
    copied_code_path = copy_current_script(paths["case_dir"])

    print("Saved files:")
    print(f"  Case folder     : {paths['case_dir']}")
    print(f"   Radar log       : {paths['log_path']}")
    print(f"   Ground truth CSV: {paths['truth_path']}")
    print(f"   Config JSON     : {paths['config_path']}")
    print(f"   Raw CIR array   : {paths['cir_npy_path']}")
    print(f"   Figures folder  : {os.path.join(paths['case_dir'], 'figures')}")

    if copied_code_path is not None:
        print(f"   Code copy       : {copied_code_path}")
    else:
        print("   Code copy.      : could not determine current script path")

    # Optional display on screen
    plot_cir_lin(
        cir_data,
        frame_idx=0,
        title=f"Level {SELECTED_LEVEL} - CIR Magnitude (Frame 1)",
        show=True,
    )
    plot_cir_magnitude(
        cir_data,
        frame_idx=0,
        title=f"Level {SELECTED_LEVEL} - CIR Magnitude (Frame 1)",
        show=True,
    )

    plot_cir_taps_magnitude(
        cir_data,
        taps=cfg.radar.cir_taps,
        frame_idx=0,
        title=f"Level {SELECTED_LEVEL} - CIR Taps Magnitude",
        show=True,
    )

    plot_antenna_pattern_polar(
        cfg,
        title="Antenna Radiation Pattern (Polar)",
        show=False,
    )

    plot_dual_antenna_coverage_cartesian(
        cfg,
        simulator.antenna_positions,
        radius_m=3.0,
        title="Dual Antenna Coverage (Cartesian)",
        show=False,
    )

    plot_cir_taps_overlay_db(
        cir_data,
        taps=cfg.radar.cir_taps,
        antenna_idx=0,
        max_frames=50,
        title="Overlay CIR Taps in dB - Antenna 0",
        show=False,
    )

    plot_cir_taps_overlay_db(
        cir_data,
        taps=cfg.radar.cir_taps,
        antenna_idx=1,
        max_frames=50,
        title="Overlay CIR Taps in dB - Antenna 1",
        show=False,
    )

    plot_phase_difference_vs_angle(
        cfg,
        angle_min_deg=-90,
        angle_max_deg=90,
        title="Phase Difference vs AoA (19 mm spacing)",
        show=False,
    )

    plot_cir_heatmap(
        cir_data,
        antenna_idx=1,
        title=f"Level {SELECTED_LEVEL} - CIR Heatmap (Antenna 1)",
        show=False,
    )

    plot_target_trajectories(
        truth,
        title=f"Level {SELECTED_LEVEL} - Target Trajectories",
        show=False,
    )

    plot_target_width_snapshot(
        cfg,
        simulator.antenna_positions,
        frame_idx=0,
        title=f"Level {SELECTED_LEVEL} - Target Width / Scatter Points",
        show=True,
    )

    plot_tap_over_time(
        cir_data,
        antenna_idx=1,
        tap_idx=14,
        title="Antenna 1 - Tap 14 Over Time",
        show=False,
    )