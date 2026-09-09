"""Synthetic CIR source that calls CIRgenerator CIRSimulator."""

from __future__ import annotations

from pathlib import Path

from radar_wrapper.paths import add_legacy_paths
from radar_wrapper.sources.base import CIRSource
from radar_wrapper.schemas.cir import CIRData, CIRMetadata

# Repo root: this file is radar_wrapper/sources/synthetic_cir.py -> parents[2].
# Used to resolve a relative default_syn_config_path so runs work from any CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SyntheticCIRSource(CIRSource):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def load(self) -> CIRData:
        add_legacy_paths()

        # Import CIRGenerator here so wrapper core stays backend-free
        from CIRDataGenerator import CIRSimulator, create_default_config  # type: ignore


        default_config_path = self.cfg.get("synthetic", "default_syn_config_path", fallback=None)
        if default_config_path:
            resolved_path = default_config_path
            if not Path(resolved_path).is_absolute():
                resolved_path = str(_PROJECT_ROOT / resolved_path)
            sim_cfg = create_default_config(resolved_path)
            print(f"[SyntheticCIRSource] cirgen base config: {resolved_path}")
        else:
            sim_cfg = create_default_config()
            print(f"[SyntheticCIRSource] cirgen base config: <CIRgenerator built-in default>")

        # Profile composition (single, logged precedence chain):
        #   cirgen base ini [profiles]  <  wrapper [synthetic]  ->  effective
        # The base ini supplies the defaults; the wrapper overrides a profile
        # ONLY when the key is present (override-if-present), so an omitted key
        # genuinely defers to the base ini instead of a divergent hardcoded default.
        for key in ("hardware_profile", "room_profile", "target_profile", "scenario_name"):
            base_val = getattr(sim_cfg, key)
            wrapper_val = self.cfg.get("synthetic", key, fallback=None)
            if wrapper_val is not None:
                setattr(sim_cfg, key, wrapper_val)
                print(f"[SyntheticCIRSource] {key}: base='{base_val}'  -> wrapper='{wrapper_val}' (effective)")
            else:
                print(f"[SyntheticCIRSource] {key}: base='{base_val}' (effective, no wrapper override)")

        # Radar ego motion, same override-if-present contract and log format as
        # the profiles above. It si handled separately only because it lives one
        # level down (sim_cfg.radar_motion.profile) rather than on sim_cfg.
        # A named preset is the ONLY ego-motion surface the wrapper exposes: the
        # numeric [radar_motion] section stays a platform-ini concern, and
        # combining the two raises in CIRSimulator._setup_radar_motion().
        base_motion_profile = sim_cfg.radar_motion.profile or ""
        wrapper_motion_profile = self.cfg.get("synthetic", "radar_motion_profile", fallback=None)
        if wrapper_motion_profile is not None:
            sim_cfg.radar_motion.profile = wrapper_motion_profile.strip()
            print(
                f"[SyntheticCIRSource] radar_motion_profile: base='{base_motion_profile}' "
                f"-> wrapper='{sim_cfg.radar_motion.profile}' (effective)"
            )
        else:
            print(
                f"[SyntheticCIRSource] radar_motion_profile: base='{base_motion_profile}' "
                f"(effective, no wrapper override)"
            )

        output_root = self.cfg.get("synthetic", "output_root", fallback=None)
        if output_root is not None:
            sim_cfg.output_root = output_root

        simulator = CIRSimulator(sim_cfg)
        cir_data_raw, truth = simulator.generate()

        # Keep the simulator reachable. Everything the wrapper needs for ego
        # motion post-analysis -- ego_pose_log, ego_motion_metadata,
        # antenna_positions -- lives on the instance and used to be discarded
        # here, so a caller had no way to audit the realised pose trajectory.
        self.simulator = simulator
        
        # CIRSimulator generates shape (num_frames, num_antennas, num_bins)
        num_frames = cir_data_raw.shape[0]
        num_antennas = cir_data_raw.shape[1]
        num_taps = cir_data_raw.shape[2]

        # radar.period in SimulationConfig is in ms (from dataconfig.RadarTestConfig, default 20)
        pri_s = float(sim_cfg.radar.period) / 1000.0

        metadata = CIRMetadata(
            source="synthetic",
            num_frames=num_frames,
            num_antennas=num_antennas,
            num_taps=num_taps,
            pri_s=pri_s,
            fast_time_resolution_ns=float(sim_cfg.cir.bin_time_s) * 1e9,
            channel=str(self.cfg.get("radar", "channel", fallback=sim_cfg.radar.channel)),
            center_frequency_hz=self.cfg.getfloat("radar", "center_frequency_hz", fallback=None),
            scenario_name=sim_cfg.scenario_name,
            raw_config=sim_cfg,
        )

        cir = CIRData(
            data=cir_data_raw,
            metadata=metadata,
            truth=truth,
        )
        cir.validate()
        return cir
        