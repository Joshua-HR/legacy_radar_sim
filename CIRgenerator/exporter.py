import csv
import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from typing import List, Optional
from datetime import datetime
from dataconfig import wrap_angle_deg


class CIRLogExporter:
    def __init__(self, config):
        """Class to export the resulting data in excel, json etc.

        Args:
            config (SimulationConfig): configuration object.
        """
        self.cfg = config

    def _create_case_folder(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg.scenario_name:
            safe_scenario = self.cfg.scenario_name.replace(" ", "_")
            case_name = f"{self.cfg.radar.test_name}_{safe_scenario}_{now}"
        else:
            case_name = f"{self.cfg.radar.test_name}_case_level{self.cfg.level}_{now}"
        case_dir = os.path.join(self.cfg.output_root, case_name)
        os.makedirs(case_dir, exist_ok=True)
        return case_dir

    def _format_config_ini_block(self) -> str:
        """Format the config ini block

        Returns:
            str: config. ini description
        """
        dut = self.cfg.dut
        radar = self.cfg.radar
        test_name = radar.test_name

        lines = [
            f"{test_name} test",
            f"##################################          config.ini for {test_name} test          ##################################",
            "",
            "[DUT]",
            f"rawlog = {dut.rawlog}",
            f"device_type = {dut.device_type}",
            f"instrument = {dut.instrument}",
            f"reset_timeout = {dut.reset_timeout}",
            f"response_timeout = {dut.response_timeout}",
            f"baudrate = {dut.baudrate}",
            f"dump_cir = {dut.dump_cir}",
            "",
            f"[{test_name}]",
            f"com = {radar.com}",
            f"preamble_code = {radar.preamble_code}",
            f"tpc = {radar.tpc}",
            f"rx1_gain = {radar.rx1_gain}",
            f"rx2_gain = {radar.rx2_gain}",
            f"channel = {radar.channel}",
            f"tx_psf = {radar.tx_psf}",
            f"rframe = {radar.rframe}",
            f"prf = {radar.prf}",
            f"sic_enable = {radar.sic_enable}",
            f"calibrate_gain = {radar.calibrate_gain}",
            f"calibrate_sic = {radar.calibrate_sic}",
            f"rx_accumulation = {radar.rx_accumulation}",
            f"packets = {radar.packets}",
            f"cir_taps = {radar.cir_taps}",
            f"offset = {radar.offset}",
            f"period = {radar.period}",
            f"ant = {radar.ant}",
            "",
            f"##################################          end of config.ini for {test_name} test          ##################################"
        ]
        return "\n".join(lines)

    def _complex_to_hex_word(self, sample: complex) -> str:
        """
        Convert one complex sample into one 8-char hex word:
        real(int16) + imag(int16), stored as unsigned two's complement.

        Args:
            sample (complex): sample to convert
        Returns:
            (str): hex word.
        """
        scale = 256
        real_i16 = int(np.clip(np.round(np.real(sample) * scale), -32768, 32767))
        imag_i16 = int(np.clip(np.round(np.imag(sample) * scale), -32768, 32767))

        real_hex = f"{(real_i16 & 0xFFFF):04x}"
        imag_hex = f"{(imag_i16 & 0xFFFF):04x}"
        return f"{real_hex}{imag_hex}"

    def _cir_line_hex(self, cir_vector: np.ndarray, taps: int) -> str:
        """Make a list of hex words from CIR array

        Args:
            cir_vector (np.ndarray): CIR array.
            taps (int): CIR number of taps.
        
        Returns:
            str: word in hex format for CIR.
        """
        words = [self._complex_to_hex_word(cir_vector[i]) for i in range(min(taps, len(cir_vector)))]
        return " ".join(words)

    def save_case(self, cir_data: np.ndarray, truth: List[List[dict]]):
        """Function for saving the results.

        Args:
            cir_data (np.ndarray): CIR data
            truth (List[List[dict]]): ground truth list.

        Returns:
            (dict): Dictionary with all the save path for logging purpouses.
        """
        case_dir = self._create_case_folder()

        log_path = os.path.join(case_dir, "radar_log.txt")
        truth_path = os.path.join(case_dir, "ground_truth.csv")
        config_path = os.path.join(case_dir, "run_config.json")
        cir_npy_path = os.path.join(case_dir, "cir_data.npy")
        configuration_path = os.path.join(case_dir, "configuration.json")
        np.save(cir_npy_path, cir_data)

        config_dump = {
            "level": self.cfg.level,
            "scenario_name": self.cfg.scenario_name,
            "antenna": asdict(self.cfg.antenna),
            "cir": asdict(self.cfg.cir),
            "dut": asdict(self.cfg.dut),
            "radar": asdict(self.cfg.radar),
            "scene": asdict(self.cfg.scene),
            "build": asdict(self.cfg.build),
            "random_seed": self.cfg.random_seed,
            "enable_gain_mismatch": self.cfg.optional.enable_gain_mismatch,
            "antenna_gain_mismatch": self.cfg.optional.antenna_gain_mismatch,
            "enable_phase_mismatch": self.cfg.optional.enable_phase_mismatch,
            "antenna_phase_mismatch_rad": self.cfg.optional.antenna_phase_mismatch_rad,
            "leakage": asdict(self.cfg.leakage),
            "frontend": asdict(self.cfg.frontend),

            "targets": [
                {
                    "name": t.name,
                    "position_xy_m": list(t.position_xy_m),
                    "position_z_m": t.position_z_m,
                    "amplitude": t.amplitude,
                    "velocity_xy_m_per_frame": list(t.velocity_xy_m_per_frame),
                    "velocity_z_m_per_frame": t.velocity_z_m_per_frame,
                    "width_m": t.width_m,
                    "num_scatter_points": t.num_scatter_points,
                    "orientation_deg": t.orientation_deg,
                    "material_type": t.material_type,
                    "material_factor": t.material_factor,
                    "enable_micro_motion": t.enable_micro_motion,
                    "micro_motion_amplitude_m": t.micro_motion_amplitude_m,
                    "micro_motion_frequency_hz": t.micro_motion_frequency_hz,
                    "micro_motion_phase_rad": t.micro_motion_phase_rad,
                    "micro_motion_axis": t.micro_motion_axis,
                    "reflections": [
                        {
                            "extra_distance_m": r.extra_distance_m,
                            "attenuation": r.attenuation,
                            "phase_offset_rad": r.phase_offset_rad,
                        }
                        for r in t.reflections
                    ],
                }
                for t in self.cfg.targets
            ],
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dump, f, indent=2)

        with open(truth_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "target_name",
                "position_x_m",
                "position_y_m",
                "position_z_m",
                "angle_deg",
                "theta_deg",
                "phi_deg",
                "elevation_deg",
                "azimuth_deg",
                "distance_center_m",
                "amplitude",
                "width_m",
                "num_scatter_points",
                "orientation_deg",
                "material_type",
                "material_factor",
                "enable_micro_motion",
                "micro_motion_amplitude_m",
                "micro_motion_frequency_hz",
                "micro_motion_phase_rad",
                "micro_motion_axis",
            ])
            for frame_truth in truth:
                for item in frame_truth:
                    writer.writerow([
                        item["frame"],
                        item["target_name"],
                        item["position_x_m"],
                        item["position_y_m"],
                        item.get("position_z_m", 0.0),
                        item["angle_deg"],
                        item.get("theta_deg", 90.0),
                        item.get("phi_deg", item["angle_deg"]),
                        item.get("elevation_deg", 0.0),
                        item.get("azimuth_deg", item["angle_deg"]),
                        item["distance_center_m"],
                        item["amplitude"],
                        item["width_m"],
                        item["num_scatter_points"],
                        item["orientation_deg"],
                        item["material_type"],
                        item["material_factor"],
                        item["enable_micro_motion"],
                        item["micro_motion_amplitude_m"],
                        item["micro_motion_frequency_hz"],
                        item["micro_motion_phase_rad"],
                        item["micro_motion_axis"],
                    ])

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(self._format_config_ini_block() + "\n")
            f.write(
                f"name: {self.cfg.build.name} "
                f"chip_id: {self.cfg.build.chip_id} "
                f"git_tag: {self.cfg.build.git_tag} "
                f"build_date: {self.cfg.build.build_date}\n"
            )

            taps = self.cfg.radar.cir_taps
            num_frames = cir_data.shape[0]
            num_ant = cir_data.shape[1]

            ant_bitmaps = ["0101", "0201"]
            rx_gains = [self.cfg.radar.rx1_gain, self.cfg.radar.rx2_gain]

            for seq in range(num_frames):
                for ant_idx in range(num_ant):
                    ant_bitmap = ant_bitmaps[ant_idx] if ant_idx < len(ant_bitmaps) else f"{ant_idx + 1:04d}"
                    rx_gain = rx_gains[ant_idx] if ant_idx < len(rx_gains) else self.cfg.radar.rx1_gain
                    cir_hex = self._cir_line_hex(cir_data[seq, ant_idx], taps)

                    f.write(
                        f"status: OK, "
                        f"sequence: {seq + 1}, "
                        f"ant_bitmap: {ant_bitmap}, "
                        f"rx_gain_dB: {rx_gain}, "
                        f"CIR: {cir_hex}\n"
                    )
        with open(configuration_path, "w") as file:
            json.dump(asdict(self.cfg), file, indent = 4)

        return {
            "case_dir": case_dir,
            "log_path": log_path,
            "truth_path": truth_path,
            "config_path": config_path,
            "cir_npy_path": cir_npy_path,
        }





def plot_cir_lin(
    cir_data: np.ndarray, frame_idx: int = 0, title: str = "",
    save_path: Optional[str] = None, show: bool = True,
):
    """Plotter for linear CIR

    Args:
        cir_data (np.ndarray): CIR data.
        frame_idx (int, optional): frame index. Defaults to 0.
        title (str, optional): plot title. Defaults to "".
        save_path (Optional[str], optional): where the plot will be saved. Defaults to None.
        show (bool, optional): if show the plot or not. Defaults to True.
    """
    plt.figure(figsize=(10, 5))
    x = np.linspace(0, cir_data.shape[-1], cir_data.shape[-1])

    for ant_idx in range(cir_data.shape[1]):
        y = cir_data[frame_idx, ant_idx]
        plt.plot(x, y, label=f"Antenna {ant_idx}")

    plt.xlim(0, cir_data.shape[-1])
    plt.xticks(np.arange(0, cir_data.shape[-1], 2))
    plt.xlabel("CIR Bin")
    plt.ylabel("Amplitude")
    plt.title(title if title else f"CIR Amplitude - Frame {frame_idx}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_cir_magnitude(
    cir_data: np.ndarray,
    frame_idx: int = 0,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plotting function for CIR in dB

    Args:
        cir_data (np.ndarray): CIR data.
        frame_idx (int, optional): frame index. Defaults to 0.
        title (str, optional): plot title. Defaults to "".
        save_path (Optional[str], optional): where the plot will be saved. Defaults to None.
        show (bool, optional): if show the plot or not. Defaults to True.
    """
    plt.figure(figsize=(10, 5))
    for ant_idx in range(cir_data.shape[1]):
        with np.errstate(divide='ignore'):
            y = amplitude_to_db(cir_data[frame_idx, ant_idx])
        plt.plot(y, label=f"Antenna {ant_idx}")
    plt.xlim(0, cir_data.shape[-1])
    plt.xticks(np.arange(0, cir_data.shape[-1], 2))
    plt.xlabel("CIR Bin")
    plt.ylabel("Magnitude (dB)")
    plt.title(title if title else f"CIR Magnitude - Frame {frame_idx}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_cir_taps_magnitude(
    cir_data: np.ndarray,
    taps: int,
    frame_idx: int = 0,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot magnitude vs CIR taps (NOT full bins).


    Args:
        cir_data (np.ndarray): CIR data.
        taps (int): number of taps to use.
        frame_idx (int, optional): frame index. Defaults to 0.
        title (str, optional): plot title. Defaults to "".
        save_path (Optional[str], optional): where to save the plot. Defaults to None.
        show (bool, optional): if show plot or not. Defaults to True.
    """

    plt.figure(figsize=(10,5))

    for ant_idx in range(cir_data.shape[1]):
        with np.errstate(divide='ignore'):
            eps = 1e-12
            taps_data = 10 * np.log10(np.maximum(np.abs(cir_data[frame_idx, ant_idx, :taps]), eps))
        plt.plot(taps_data, marker='o', label=f"Antenna {ant_idx}")
    
    plt.xlabel("CIR Tap Index")
    plt.ylabel("Magnitude (dB)")
    plt.title(title if title else f"CIR Magnitude (First {taps} Taps) - Frame {frame_idx}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


def _antenna_gain_dbi_for_polar_plot(cfg, phi_deg: float) -> float:
    """Return gain in dBi for an azimuth cut at theta=90 (horizon), used by
    plot_antenna_pattern_polar() for every pattern_mode.

    - "isotropic": 0 dBi everywhere.
    - "table"/"table_azimuth" (legacy): azimuth-only lookup, as before.
    - "table_3d"/"table_3d_separable": G(phi, theta=90) azimuth cut.
    - anything else (e.g. "cosine"): fall back to the legacy azimuth table
      shape so the plot still renders something meaningful.
    """
    antenna = cfg.antenna
    pattern_mode = antenna.pattern_mode

    if pattern_mode == "isotropic":
        return 0.0

    if pattern_mode == "table_3d":
        rel_phi = wrap_angle_deg(phi_deg - antenna.boresight_phi_deg)
        rel_theta = 90.0 - antenna.boresight_theta_deg
        return antenna._interp_2d_gain_dbi(rel_phi, rel_theta)

    if pattern_mode == "table_3d_separable":
        rel_phi = wrap_angle_deg(phi_deg - antenna.boresight_phi_deg)
        rel_theta = 90.0 - antenna.boresight_theta_deg
        phi_gain_dbi = antenna._interp_1d_dbi(rel_phi, antenna.pattern_phi_deg, antenna.pattern_phi_gains_dbi)
        theta_gain_dbi = antenna._interp_1d_dbi(rel_theta, antenna.pattern_theta_deg, antenna.pattern_theta_gains_dbi)
        return phi_gain_dbi + theta_gain_dbi

    # Legacy azimuth-only table (and cosine fallback): as before.
    rel_angle = wrap_angle_deg(phi_deg - antenna.boresight_deg)
    return antenna._interpolate_pattern_gain_dbi(rel_angle)


def plot_antenna_pattern_polar(
    cfg,
    title: str = "Antenna Radiation Pattern (Polar, normalized dB)",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plotting function for antenna polar radiation pattern.

    Always plots an azimuth (phi) cut at theta=90 (horizon), i.e. G(phi, 90).
    For pattern_mode == "isotropic" this is a flat 0 dB circle; for legacy
    "table"/"table_azimuth" it reproduces the previous azimuth-only plot;
    for "table_3d"/"table_3d_separable" it evaluates the full G(phi, theta)
    pattern at theta=90 (the horizontal plane, matching the legacy
    azimuth-only plot's implicit assumption).

    Args:
        cfg (SimulationConfig): simulation configuration file containing antenna information.
        title (str, optional): plot title. Defaults to "Antenna Radiation Pattern (Polar, normalized dB)".
        save_path (Optional[str], optional): where to save the plot. Defaults to None.
        show (bool, optional): it to show the plot or not. Defaults to True.
    """
    angles_deg = np.linspace(-180, 180, 721)

    gains_dbi = np.array([
        _antenna_gain_dbi_for_polar_plot(cfg, float(phi_deg))
        for phi_deg in angles_deg
    ])

    # Normalizeƒ to peak = 0 dB for visual comparison with datasheet-style plots
    gains_db_norm = gains_dbi - np.max(gains_dbi)

    # Clamp for cleaner polar display
    gains_db_norm = np.clip(gains_db_norm, -15.0, 0.0)

    # Shift to positive radius for matplotlib polar plotting
    radii = gains_db_norm + 15.0

    angles_rad = np.deg2rad(angles_deg)

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, projection="polar")
    ax.plot(angles_rad, radii, linewidth=2)

    ax.set_title(title, va="bottom")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    # Show labels as normalized dB
    ax.set_rticks([0, 3, 6, 9, 12, 15])
    ax.set_yticklabels(["-15", "-12", "-9", "-6", "-3", "0 dB"])
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_antenna_pattern_phi_theta_heatmap(
    cfg,
    title: str = "Antenna Pattern G(phi, theta) Heatmap (dBi)",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Optional 2D heatmap of the full G(phi, theta) pattern (dBi).

    x-axis = phi/azimuth, y-axis = theta/polar angle (0=+z, 90=horizon,
    180=-z), color = gain dBi. Only meaningful for pattern_mode in
    ("table_3d", "table_3d_separable"); for other modes this still renders
    (isotropic -> flat 0 dBi plane).

    Args:
        cfg (SimulationConfig): configuration object containing antenna information.
        title (str, optional): plot title.
        save_path (Optional[str], optional): where to save the plot.
        show (bool, optional): whether to display the plot.
    """
    phi_deg = np.linspace(-180, 180, 73)
    theta_deg = np.linspace(0, 180, 37)

    gain_dbi = np.zeros((len(theta_deg), len(phi_deg)), dtype=float)

    antenna = cfg.antenna
    for ti, th in enumerate(theta_deg):
        for pi, ph in enumerate(phi_deg):
            if antenna.pattern_mode == "table_3d":
                rel_phi = wrap_angle_deg(float(ph) - antenna.boresight_phi_deg)
                rel_theta = float(th) - antenna.boresight_theta_deg
                gain_dbi[ti, pi] = antenna._interp_2d_gain_dbi(rel_phi, rel_theta)
            elif antenna.pattern_mode == "table_3d_separable":
                rel_phi = wrap_angle_deg(float(ph) - antenna.boresight_phi_deg)
                rel_theta = float(th) - antenna.boresight_theta_deg
                phi_gain = antenna._interp_1d_dbi(rel_phi, antenna.pattern_phi_deg, antenna.pattern_phi_gains_dbi)
                theta_gain = antenna._interp_1d_dbi(rel_theta, antenna.pattern_theta_deg, antenna.pattern_theta_gains_dbi)
                gain_dbi[ti, pi] = phi_gain + theta_gain
            else:
                gain_dbi[ti, pi] = _antenna_gain_dbi_for_polar_plot(cfg, float(ph))

    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(phi_deg, theta_deg, gain_dbi, cmap="viridis", shading="auto")
    ax.set_xlabel("Phi / Azimuth (deg)")
    ax.set_ylabel("Theta / Polar angle from +z (deg)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Gain (dBi)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    

def plot_dual_antenna_coverage_cartesian(
    cfg,
    antennas_position,
    radius_m: float = 3.0,
    num_angles: int = 721,
    title: str = "Dual Antenna Coverage (Cartesian)",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plotting function for show the dual antenna coverage in Cartesian coordinates.

    Args:
        cfg (SimulationConfig): configuration object containing the antenna information.
        radius_m (float, optional): radius in meters. Defaults to 3.0.
        num_angles (int, optional): number of angles. Defaults to 721.
        title (str, optional): plot title. Defaults to "Dual Antenna Coverage (Cartesian)".
        save_path (Optional[str], optional): where to save the plot. Defaults to None.
        show (bool, optional): it show the plot or not. Defaults to True.
    """

    plt.figure(figsize=(9, 8))

    colors = ["blue", "green"]

    for ant_idx, ap in enumerate(antennas_position):
        angles_deg = np.linspace(-180, 180, num_angles)
        xs = []
        ys = []

        for ang in angles_deg:
            # antennas_position may be shape [N, 3]; keep the horizontal
            # (x, y) sweep at the antenna's own height (z unchanged).
            far_point = ap + radius_m * np.array([
                np.cos(np.deg2rad(ang)),
                np.sin(np.deg2rad(ang)),
                0.0,
            ])[: len(ap)]

            gain = cfg.antenna._antenna_pattern_gain(ap, far_point, ant_idx=ant_idx)

            # scale radius by gain
            rr = radius_m * gain

            x = ap[0] + rr * np.cos(np.deg2rad(ang))
            y = ap[1] + rr * np.sin(np.deg2rad(ang))

            xs.append(x)
            ys.append(y)

        plt.plot(xs, ys, color=colors[ant_idx % len(colors)], linewidth=2, label=f"RX{ant_idx+1} pattern")
        plt.scatter(ap[0], ap[1], color=colors[ant_idx % len(colors)], marker="^", s=120)

    rx = cfg.scene.radar_position_x_m
    ry = cfg.scene.radar_position_y_m

    plt.scatter(rx, ry, color="red", marker="s", s=100, label="Radar origin")
    plt.arrow(rx, ry, 0.8, 0, head_width=0.05, head_length=0.08, length_includes_head=True, color="red")
    plt.text(rx + 0.85, ry + 0.03, "Forward (+x)", color="red")

    if len(antennas_position) >= 2:
        plt.plot(antennas_position[:, 0], antennas_position[:, 1], "k--", linewidth=1.5, label="Antenna baseline")

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_cir_taps_overlay_db(
    cir_data: np.ndarray,
    taps: int,
    antenna_idx: int,
    max_frames: Optional[int] = None,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plotting function for overlay many frames for one antenna in dB, similar to the real-data CIR plot.

    Args:
        cir_data (np.ndarray): CIR data.
        taps (int): number of taps.
        antenna_idx (int): antenna index.
        max_frames (Optional[int], optional): max number of frames to take. Defaults to None.
        title (str, optional): plot title. Defaults to "".
        save_path (Optional[str], optional): where to save the plot. Defaults to None.
        show (bool, optional): if show the plot or not. Defaults to True.
    """
    if max_frames is None:
        max_frames = cir_data.shape[0]

    plt.figure(figsize=(10, 5))

    for frame_idx in range(min(max_frames, cir_data.shape[0])):
        with np.errstate(divide='ignore'):
            eps = 1e-12
            y = 10 * np.log10(np.maximum(np.abs(cir_data[frame_idx, antenna_idx, :taps]), eps))
        plt.plot(y, alpha=0.35)

    plt.xlabel("CIR Tap Index")
    plt.ylabel("Magnitude (dB)")
    plt.title(title if title else f"Overlay CIR Taps in dB - Antenna {antenna_idx}")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_phase_difference_vs_angle(
    cfg,
    angle_min_deg: float = -90,
    angle_max_deg: float = 90,
    num_points: int = 1000,
    title: str = "Phase Difference vs AoA",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot theoretical phase difference between RX1 and RX2 versus Angle of Arrival (AoA). This is the key AoA relation for 2-antenna interferometric radar.

    Args:
        cfg (SimulationConfig): configuration object containing the phase values
        angle_min_deg (float, optional): lower limit for angle in degrees. Defaults to -90.
        angle_max_deg (float, optional): upper limit for angles in degrees. Defaults to 90.
        num_points (int, optional): number of point to display. Defaults to 1000.
        title (str, optional): plot title. Defaults to "Phase Difference vs AoA".
        save_path (Optional[str], optional): where to save the plot. Defaults to None.
        show (bool, optional): if show the plot or not. Defaults to True.
    """

    d = cfg.antenna.antenna_spacing_m
    f = cfg.antenna.carrier_freq_hz
    c = cfg.antenna.c

    wavelength = c / f

    angles_deg = np.linspace(angle_min_deg, angle_max_deg, num_points)
    angles_rad = np.deg2rad(angles_deg)

    # Core phase difference equation
    phase_diff_rad = (2 * np.pi * d / wavelength) * np.sin(angles_rad)

    # Optional wrapped phase (-pi to pi)
    wrapped_phase = np.angle(np.exp(1j * phase_diff_rad))

    plt.figure(figsize=(10, 6))

    plt.plot(
        angles_deg,
        phase_diff_rad,
        linewidth=2,
        label="Unwrapped Phase Difference"
    )

    plt.plot(
        angles_deg,
        wrapped_phase,
        linestyle="--",
        linewidth=2,
        label="Wrapped Phase (-π to π)"
    )

    plt.axhline(np.pi, linestyle=":", linewidth=1)
    plt.axhline(-np.pi, linestyle=":", linewidth=1)

    plt.xlabel("AoA Angle (degrees)")
    plt.ylabel("Phase Difference (radians)")
    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_cir_heatmap(
    cir_data: np.ndarray,
    antenna_idx: int = 0,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plotting function for CIR heatmap

    Args:
        cir_data (np.ndarray): CIR data.
        antenna_idx (int, optional): antenna index. Defaults to 0.
        title (str, optional): plot title. Defaults to "".
        save_path (Optional[str], optional): where to save the plot. Defaults to None.
        show (bool, optional): if show the plot or not. Defaults to True.
    """
    data = np.abs(cir_data[:, antenna_idx, :])
    fig, ax = plt.subplots(figsize=(10,6))
    mesh = ax.pcolormesh(data, cmap='viridis', edgecolors='none')
    ax.set_xlabel("CIR Bin")
    ax.set_ylabel("Frame")
    ax.set_title(title if title else f"CIR Heatmap - Antenna {antenna_idx}")
    fig.colorbar(mesh, ax = ax, label="Magnitude")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_tap_over_time(
    cir_data: np.ndarray,
    antenna_idx: int,
    tap_idx: int,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plotting function for taps over time.

    Args:
        cir_data (np.ndarray): CIR data.
        antenna_idx (int): antenna index.
        tap_idx (int): tap index.
        title (str, optional): plot title. Defaults to "".
        save_path (Optional[str], optional): where to save the plot. Defaults to None.
        show (bool, optional): if show the plot or not. Defaults to True.
    """
    with np.errstate(divide='ignore'):
        eps = 1e-12
        data = 10 * np.log10(np.maximum(np.abs(cir_data[:, antenna_idx, tap_idx]), eps))

    plt.figure(figsize=(10, 4))
    plt.plot(data, marker="o")
    plt.xlabel("Frame")
    plt.ylabel("Magnitude (dB)")
    plt.title(title if title else f"Antenna {antenna_idx} - Tap {tap_idx} Over Time")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_target_trajectories(
    truth: list[list[dict]],
    title: str = "Target Trajectories",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plotting function for targets trajectories

    Args:
        truth (List[List[dict]]): the dictionary containing the ground truth values.
        title (str, optional): plot title. Defaults to "".
        save_path (Optional[str], optional): where to save the plot. Defaults to None.
        show (bool, optional): if show the plot or not. Defaults to True.
    """
    target_positions = {}

    for frame_truth in truth:
        for item in frame_truth:
            name = item["target_name"]
            target_positions.setdefault(name, {"x": [], "y": []})
            target_positions[name]["x"].append(item["position_x_m"])
            target_positions[name]["y"].append(item["position_y_m"])

    plt.figure(figsize=(7, 7))
    for name, pos in target_positions.items():
        plt.plot(pos["x"], pos["y"], marker="o", label=name)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_target_width_snapshot(
    cfg,
    antennas_position,
    frame_idx: int = 0,
    title: str = "",
    save_path: str | None = None,
    show: bool = True,
):
    """_summary_

    Args:
        cfg (SimulationConfig): configuration object containing antenna information.
        frame_idx (int, optional): frame index. Defaults to 0.
        title (str, optional): plot title. Defaults to "".
        save_path (Optional[str], optional): where to save the plot. Defaults to None.
        show (bool, optional): if show the plot or not. Defaults to True.
    """

    plt.figure(figsize=(8, 8))
    rx = cfg.scene.radar_position_x_m
    ry = cfg.scene.radar_position_y_m

    plt.scatter(antennas_position[:, 0], antennas_position[:, 1], marker="^", s=140, label="Antennas")
    plt.scatter(rx, ry, marker="s", s=120, label="Radar origin")

    if len(antennas_position) >= 2:
        plt.plot(antennas_position[:, 0], antennas_position[:, 1], linestyle="--", linewidth=1.5, label="Antenna baseline")

    plt.arrow(rx, ry, 0.8, 0, head_width=0.05, head_length=0.08, length_includes_head=True)
    plt.text(rx + 0.85, ry + 0.03, "Radar forward (+x)", fontsize=10)

    for target in cfg.targets + cfg.static_clutter:#sim.cfg.targets:
        center = target._get_center_position(frame_idx)
        scatter_pts = np.array(target._get_scatter_points(frame_idx))

        plt.scatter(center[0], center[1], marker="o", s=100, label=f"{target.name} center")
        plt.scatter(scatter_pts[:, 0], scatter_pts[:, 1], marker="x", s=80, label=f"{target.name} scatter pts")

        if len(scatter_pts) > 1:
            plt.plot(scatter_pts[:, 0], scatter_pts[:, 1], linestyle="--")
        
        plt.text(center[0] + 0.03, center[1] + 0.03, target.name, fontsize=10)

    plt.xlabel("X Position (m)   [Forward / Range]")
    plt.ylabel("Y Position (m)   [Left / Right]")
    plt.title(title if title else f"Azimuth Geometry - Frame {frame_idx + 1}")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()




def amplitude_to_db(x: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Converts complex amplitude to dB with controlled display floor.

    Since CIR is complex amplitude (not power),
    we use 20*log10(), not 10*log10().

    Args:
        x (np.ndarray): array to convert
        floor_db (float, optional): fallback value for undersized input . Defaults to -60.0

    Returns:
        np.ndarray: amplitude in dB.
    """

    mag = np.abs(x)

    # Convert display floor from dB to linear amplitude
    floor_linear = 10 ** (floor_db / 20.0)

    return 20.0 * np.log10(np.maximum(mag, floor_linear))


def print_strongest_taps(cir_data, frame_idx=0, taps_to_check=64):
    """Prints in stout the strongest tap for debugging purposes

    Args:
        cir_data (List[np.ndarray]): CIR frame
        frame_idx (int, optional): frame index. Defaults to 0.
        taps_to_check (int, optional): window of taps for the lookout. Defaults to 64.
    """
    print("\n===== STRONGEST TAPS ======")
    for ant_idx in range(cir_data.shape[1]):
        mag = np.abs(cir_data[frame_idx, ant_idx, :taps_to_check])
        idx = int(np.argmax(mag))
        print(f"Antenna {ant_idx}: strongest tap = {idx}, magnitude = {mag[idx]:.6f}")
    print("==========================\n")


def print_truth_summary(truth: List[List[dict]], max_frames: int = 5):
    """Prints in stout the ground truth summary. For debugging purposes.

    Args:
        truth (List[List[dict]]): ground truth
        max_frames (int, optional): max number of frames to take into consideration. Defaults to 5.
    """
    print("\n=============== GROUND TRUTH SUMMARY ===============")
    for frame_idx, frame_truth in enumerate(truth[:max_frames]):
        print(f"\nFrame {frame_idx + 1}:")
        for item in frame_truth:
            print(
                f"   {item['target_name']}: "
                f"pos=({item['position_x_m']:.3f}, {item['position_y_m']:.3f}) m, "
                f"angle={item['angle_deg']:.2f} deg, "
                f"dist={item['distance_center_m']:.3f} m, "
                f"width={item['width_m']:.3f} m, "
                f"scatter_pts={item['num_scatter_points']}"
            )
        print("======================================================\n")


def save_all_figures(case_dir: str, cfg, antennas_position, cir_data: np.ndarray, truth: List[List[dict]], selected_level:str):
    """Helper function for save all plots.

    Args:
        case_dir (str): target directory
        cfg (SimulationConfig): configuration object.
        cir_data (np.ndarray): CIR array.
        truth (LIST[LIST[dict]]): ground truth dictionary
    
    Returns:
        (str): where the plots have been saved.
    """
    figures_dir = os.path.join(case_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig_paths = {}
    fig_paths["cir_amplitude_frame1"] = os.path.join(figures_dir, "cir_amplitude_frame1.png")
    plot_cir_lin(
        cir_data,
        frame_idx=0,
        title=f"Level {selected_level} - CIR Amplitude (Frame 1)",
        show=False,
    )
    fig_paths["cir_magnitude_frame1"] = os.path.join(figures_dir, "cir_magnitude_frame1.png")
    plot_cir_magnitude(
        cir_data,
        frame_idx=0,
        title=f"Level {cfg.level} - CIR Magnitude (Frame 1)",
        save_path=fig_paths["cir_magnitude_frame1"],
        show=False,
    )

    fig_paths["antenna_pattern_polar"] = os.path.join(figures_dir, "antenna_pattern_polar.png")
    plot_antenna_pattern_polar(
        cfg,
        title="Antenna Radiation Pattern (Polar)",
        save_path=fig_paths["antenna_pattern_polar"],
        show=False,
    )

    fig_paths["dual_antenna_coverage"] = os.path.join(figures_dir, "dual_antenna_coverage.png")
    plot_dual_antenna_coverage_cartesian(
        cfg,
        antennas_position,
        radius_m=3.0,
        title="Dual Antenna Coverage (Cartesian)",
        save_path=fig_paths["dual_antenna_coverage"],
        show=False,
    )


    fig_paths["phase_difference_vs_angle"] = os.path.join(figures_dir, "phase_difference_vs_angle.png")
    plot_phase_difference_vs_angle(
        cfg,
        angle_min_deg=-90,
        angle_max_deg=90,
        title="Phase Difference vs AoA (19 mm spacing)",
        save_path=fig_paths["phase_difference_vs_angle"],
        show=False,
    )

    fig_paths["cir_taps_magnitude"] = os.path.join(figures_dir, "cir_taps_magnitude.png")
    plot_cir_taps_magnitude(
        cir_data,
        taps=cfg.radar.cir_taps,
        frame_idx=0,
        title=f"Level {cfg.level} - CIR Taps Magnitude",
        save_path=fig_paths["cir_taps_magnitude"],
        show=False,
    )

    fig_paths["cir_heatmap_ant0"] = os.path.join(figures_dir, "cir_heatmap_ant0.png")
    plot_cir_heatmap(
        cir_data,
        antenna_idx=0,
        title=f"Level {cfg.level} - CIR Heatmap (Antenna 0)",
        save_path=fig_paths["cir_heatmap_ant0"],
        show=False,
    )

    fig_paths["target_trajectories"] = os.path.join(figures_dir, "target_trajectories.png")
    plot_target_trajectories(
        truth,
        title=f"Level {cfg.level} - Target Trajectories",
        save_path=fig_paths["target_trajectories"],
        show=False,
    )

    fig_paths["target_width_snapshot_frame1"] = os.path.join(figures_dir, "target_width_snapshot_frame1.png")
    plot_target_width_snapshot(
        cfg,
        antennas_position,
        frame_idx=0,
        title=f"Level {cfg.level} - Target Width / Scatter Points",
        save_path=fig_paths["target_width_snapshot_frame1"],
        show=False,
    )

    return fig_paths


def copy_current_script(case_dir: str):
    """Helper function to copy the current script and save in the target folder.

    Args:
        case_dir (str): target folder path.

    Returns:
        (str): path where the script has been saved.
    """
    try:
        script_path = os.path.abspath(__file__)
    except NameError:
        script_path = None

    if script_path is None or not os.path.isfile(script_path):
        return None

    dst_path = os.path.join(case_dir, os.path.basename(script_path))
    shutil.copy2(script_path, dst_path)
    return dst_path


def plot_total(
    cir_data: np.ndarray,
    cfg,
    antennas_position: np.ndarray,
    truth: List[List[dict]],
    selected_level: int = 2,
    save_dir: Optional[str] = None,
    show_plots: bool = False,
):
    """
    Wrapper function that calls all plot functions to display CIR simulation results.

    Args:
        cir_data (np.ndarray): CIR data array with shape [num_frames, num_antennas, num_bins].
        cfg (SimulationConfig): Simulation configuration object.
        antennas_position (np.ndarray): Antenna positions array.
        truth (List[List[dict]]): Ground truth data.
        selected_level (int, optional): Simulation level for labeling. Defaults to 2.
        save_dir (Optional[str], optional): Directory to save figures. Defaults to None.
        show_plots (bool, optional): Whethere to display plots (True) or just save (False). Defaults to False.
    """
    print("Generating all plots...")

    # Create save directory if it doesn't exist
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # 1. CIR Magnitude (Linear)
    plot_cir_lin(
        cir_data,
        frame_idx=0,
        title=f"Level {selected_level} - CIR Magnitude (Frame 1)",
        save_path=os.path.join(save_dir, "cir_lin.png") if save_dir else None,
        show=show_plots,
    )

    # 2. CIR Magnitude (dB)
    plot_cir_magnitude(
        cir_data,
        frame_idx=0,
        title=f"Level {selected_level} - CIR Magnitude (Frame 1)",
        save_path=os.path.join(save_dir, "cir_magnitude.png") if save_dir else None,
        show=show_plots,
    )

    # 3. CIR Taps Magnitude
    plot_cir_taps_magnitude(
        cir_data,
        taps=cfg.radar.cir_taps,
        frame_idx=0,
        title=f"Level {selected_level} - CIR Taps Magnitude",
        save_path=os.path.join(save_dir, "cir_taps_magnitude.png") if save_dir else None,
        show=show_plots,
    )

    # 4. Antenna Pattern (Polar)
    plot_antenna_pattern_polar(
        cfg,
        title="Antenna Radiation Pattern (Polar)",
        save_path=os.path.join(save_dir, "antenna_pattern_polar.png") if save_dir else None,
        show=show_plots,
    )

    # 5. Dual Antenna Coverage (Cartesian)
    plot_dual_antenna_coverage_cartesian(
        cfg,
        antennas_position,
        radius_m=3.0,
        title="Dual Antenna Coverage (Cartesian)",
        save_path=os.path.join(save_dir, "dual_antenna_coverage.png") if save_dir else None,
        show=show_plots,
    )

    # 6. Overlay CIR Taps dB - Antenna 0
    plot_cir_taps_overlay_db(
        cir_data,
        taps=cfg.radar.cir_taps,
        antenna_idx=0,
        max_frames=50,
        title="Overlay CIR Taps in dB - Antenna 0",
        save_path=os.path.join(save_dir, "cir_overlay_ant0.png") if save_dir else None,
        show=show_plots,
    )

    # 7. Overlay CIR Taps dB - Antenna 1
    plot_cir_taps_overlay_db(
        cir_data,
        taps=cfg.radar.cir_taps,
        antenna_idx=1,
        max_frames=50,
        title="Overlay CIR Taps in dB - Antenna 1",
        save_path=os.path.join(save_dir, "cir_overlay_ant1.png") if save_dir else None,
        show=show_plots,
    )

    # 8. Phase Difference vs AoA
    plot_phase_difference_vs_angle(
        cfg,
        angle_min_deg=-90,
        angle_max_deg=90,
        title="Phase Difference vs AoA (19 mm spacing)",
        save_path=os.path.join(save_dir, "phase_vs_aoa.png") if save_dir else None,
        show=show_plots,
    )

    # 9. CIR Heatmap - Antenna 1
    plot_cir_heatmap(
        cir_data,
        antenna_idx=1,
        title=f"Level {selected_level} - CIR Heatmap (Antenna 1)",
        save_path=os.path.join(save_dir, "cir_heatmap.png") if save_dir else None,
        show=show_plots,
    )

    # 10. Target Trajectories
    plot_target_trajectories(
        truth,
        title=f"Level {selected_level} - Target Trajectories",
        save_path=os.path.join(save_dir, "target_trajectories.png") if save_dir else None,
        show=show_plots,
    )

    # 11. Target Width Snapshot
    plot_target_width_snapshot(
        cfg,
        antennas_position,
        frame_idx=0,
        title=f"Level {selected_level} - Target Width / Scatter Points",
        save_path=os.path.join(save_dir, "target_snapshot.png") if save_dir else None,
        show=show_plots,
    )

    # 12. Tap Over Time
    plot_tap_over_time(
        cir_data,
        antenna_idx=1,
        tap_idx=14,
        title="Antenna 1 - Tap 14 Over Time",
        save_path=os.path.join(save_dir, "tap_over_time.png") if save_dir else None,
        show=show_plots,
    )

    print("All plots generated successfully!")