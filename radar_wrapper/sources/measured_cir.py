"""Measured CIR source - parser for actual UA200 CIR dump logs.

Mirrors the CIRgenerator pattern:

    SimulationConfig  →  MeasuredCIRConfig
    create_default_config()  →  create_measured_config()
    CIRSimulator.generate()  →  MeasuredCIRSource.load()

The source reads a `.log` file containing hex-encoded CIR notification
packets and produces a standard ``CIRData`` with shape
``(num_frames, num_antennas, num_taps)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from radar_wrapper.sources.base import CIRSource
from radar_wrapper.schemas.cir import CIRData, CIRMetadata


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MeasuredCIRParam:
    """CIR-level parameters extracted from packets."""
    num_taps: int = 16
    num_antennas: int = 2
    num_frames: int = 0 # discovered from log


@dataclass
class MeasuredRadarParam:
    """Radar metadata parsed from the log header."""
    channel: int = 9
    period_ms: float = 20.0
    rx1_gain: int = 40
    rx2_gain: int = 40
    device_type: str = "UA200"
    cir_taps: int = 16
    rx_accumulation: int = 52
    packets: int = 128


@dataclass
class MeasuredCIRConfig:
    """Top-level config for a measured CIR source.

    Mirrors ``SimulationConfig`` from CIRgenerator.
    """
    path: str
    cir: MeasuredCIRParam = field(default_factory=MeasuredCIRParam)
    radar: MeasuredRadarParam = field(default_factory=MeasuredRadarParam)
    log_dirname: str = ""


# ---------------------------------------------------------------------------
# Regex helpers - CIR packet parser
# ---------------------------------------------------------------------------

LOG_TS_RE = re.compile(
    r"DEBUG\s+"
    r"(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})"
    r"\s*:"
)

READ_HEX_RE = re.compile(
    r":\s*read\s*-\s*(?P<hex>[0-9a-fA-F\s]+)\s*$"
)


def _normalize_hex(hex_str: str) -> str:
    return re.sub(r"\s+", "", hex_str).lower()


def _is_candidate_cir_hex(hex_str: str) -> bool:
    """Quick check: packet starts with 6c3f and BID is 0x13."""
    if hex_str is None:
        return False
    hs = _normalize_hex(hex_str)
    # Need at least 4 header hytes + 4 BID bytes = 8 bytes (16 hex chars)
    if len(hs) < 16:
        return False
    if not hs.startswith("6c3f"):
        return False
    if hs[8:16] != "13000000":
        return False
    return True


def _extract_read_hex(line: str) -> Optional[str]:
    m = READ_HEX_RE.search(line)
    if not m:
        return None
    return _normalize_hex(m.group("hex"))


# ---------------------------------------------------------------------------
# Single-packet parser
# ---------------------------------------------------------------------------

def _parse_cir_packet_hex(hex_str: str) -> dict[str, Any]:
    """Parse a single CIR notification hex packet.

    Layout (after UCI header):

        Type/BID             4 bytes  little-endian  (0x13)
        Status               1 byte
        Number of parameters 1 byte
        Sequence number      4 bytes  little-endian
        Antenna bitmap       2 bytes  little-endian
        Radar Rx gain        1 byte
        CIR Data size        1 byte  (number of complex taps)
        CIR Data             N * 4 bytes  (int16 I + int16 Q, LE)
    """
    hs = _normalize_hex(hex_str)
    if len(hs) % 2 != 0:
        raise ValueError("Odd-length hex string")

    b = bytes.fromhex(hs)
    if len(b) < 4 + 14:
        raise ValueError(f"Packet too short: {len(b)} bytes")

    payload = b[4:]  # skip UCI header (6c3f + length)

    if len(payload) < 14:
        raise ValueError(f"Payload too short: {len(payload)} bytes")

    bid = int.from_bytes(payload[0:4], byteorder="little", signed=False)
    if bid != 0x13:
        raise ValueError(f"Not a CIR packet: BID=0x{bid:08x}")

    status = payload[4]
    sequence = int.from_bytes(payload[6:10], byteorder="little", signed=False)
    antenna_bitmap = int.from_bytes(payload[10:12], byteorder="little", signed=False)
    rx_gain = payload[12]
    cir_size = payload[13]

    cir_data = payload[14:]
    expected_bytes = cir_size * 4  # each tap = int16 I + int16 Q

    if len(cir_data) < expected_bytes:
        n_taps = len(cir_data) // 4
        cir_data = cir_data[: n_taps * 4]
    else:
        n_taps = cir_size
        cir_data = cir_data[:expected_bytes]

    iq = np.frombuffer(cir_data, dtype="<i2").reshape(-1, 2)
    i = iq[:, 0].astype(np.float32)
    q = iq[:, 1].astype(np.float32)
    cir = (i + 1j * q).astype(np.complex64)

    # Antenna bitmap: bit0=RX1, bit1=RX2, bit8=TX1, bit9=TX2
    rx_antennas = []
    if antenna_bitmap & (1 << 0):
        rx_antennas.append(1)
    if antenna_bitmap & (1 << 1):
        rx_antennas.append(2)

    tx_antennas = []
    if antenna_bitmap & (1 << 8):
        tx_antennas.append(1)
    if antenna_bitmap & (1 << 9):
        tx_antennas.append(2)

    rx = rx_antennas[0] if len(rx_antennas) == 1 else None

    return {
        "bid": bid,
        "status": status,
        "sequence": sequence,
        "antenna_bitmap": antenna_bitmap,
        "rx_antennas": rx_antennas,
        "tx_antennas": tx_antennas,
        "rx": rx,
        "rx_gain": rx_gain,
        "n_taps": n_taps,
        "cir": cir,
    }


# ---------------------------------------------------------------------------
# Log header parser
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(r":\s*(?P<key>\w+)\s*=\s*(?P<value>\S+)")


def _parse_log_header(path: Path) -> dict[str, str]:
    """Extract key=value lines from the config header at the top of the log.

    Reads at most the first 200 lines (config block never extends past that).
    Returns a flat dict of ``{key: value_string}``.
    """
    result: dict[str, str] = {}
    with path.open("r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 200:
                break
            m = _KEY_RE.search(line)
            if m:
                result[m.group("key")] = m.group("value").strip()
    return result


def create_measured_config(
    wrapper_cfg,
) -> MeasuredCIRConfig:
    """Create MeasuredCIRConfig from wrapper config + log header.

    Mirrors ``create_default_config()`` from CIRgenerator.

    Priority:
        1. Log header (actual device config captured in the dump)
        2. Wrapper INI ``[radar]`` section (compiled-in defaults)
    """
    log_path = Path(
        wrapper_cfg.get("measured", "path", fallback="")
    )

    if not log_path.exists():
        raise FileNotFoundError(f"Measured CIR log not found: {log_path}")

    # Parse log header
    header = _parse_log_header(log_path)

    # --- Helper to safely read integers from header or cfg ---
    def _get_int(key: str, *, cfg_key: str = "", fallback: int = 0) -> int:
        """Primary: log header.  Fallback: wrapper cfg value.  Last: default."""
        if key in header:
            try:
                return int(header[key])
            except ValueError:
                pass
        if cfg_key:
            raw = wrapper_cfg.get("radar", cfg_key, fallback="")
            if raw:
                try:
                    return int(raw)
                except ValueError:
                    pass
        return fallback

    def _get_float(key:str, *, cfg_key: str = "", fallback = 0.0) -> float:
        """Primary: log header.  Fallback: wrapper cfg value.  Last: default."""
        if key in header:
            try:
                return float(header[key])
            except ValueError:
                pass
        if cfg_key:
            raw = wrapper_cfg.get("radar", cfg_key, fallback="")
            if raw:
                try:
                    return float(raw)
                except ValueError:
                    pass
        return fallback

    def _get_str(key: str, *, cfg_key: str = "", fallback: "") -> str:
        """Primary: log header.  Fallback: wrapper cfg value.  Last: default."""
        if key in header:
            return header[key]
        if cfg_key:
            raw = wrapper_cfg.get("radar", cfg_key, fallback="").strip()
            if raw:
                return raw
        return fallback

    # Build radar params with log-as-primary, cfg-as-fallback
    channel = _get_int("channel", cfg_key="channel", fallback=9)

    period_ms = _get_float("period", cfg_key="pri_s", fallback=0.02)
    # Header "period" is in ms; cfg "pri_s" is in seconds.
    # If header "period" was used, we already got the ms value.
    # If cfg pri_s was used, convert to ms.
    if "period" in header:
        period_ms = _get_float("period", fallback=20.0)
    else:
        pri_raw = wrapper_cfg.get("radar", "pri_s", fallback="")
        if pri_raw:
            try:
                period_ms = float(pri_raw) * 1000.0
            except ValueError:
                period_ms = 20.0

    rx1_gain = _get_int("rx1_gain", cfg_key="rx1_gain", fallback=40)
    rx2_gain = _get_int("rx2_gain", cfg_key="rx2_gain", fallback=40)

    device_type = _get_str("device_type", cfg_key="device_type", fallback="UA200")
    cir_taps = _get_int("cir_taps", fallback=16)
    rx_accumulation = _get_int("rx_accumulation", cfg_key="num_taps", fallback=64)

    packets_val = header.get("packets", "0")
    try:
        packets = int(packets_val)
    except ValueError:
        packets = 0

    cfg = MeasuredCIRConfig(
        path=str(log_path),
        cir=MeasuredCIRParam(
            num_taps=cir_taps,
        ),
        radar=MeasuredRadarParam(
            channel=channel,
            period_ms=period_ms,
            rx1_gain=rx1_gain,
            rx2_gain=rx2_gain,
            device_type=device_type,
            cir_taps=cir_taps,
            rx_accumulation=rx_accumulation,
            packets=packets,
        ),
        log_dirname=log_path.parent.name,
    )
    return cfg


# ---------------------------------------------------------------------------
# MeasuredCIRSource  (main class)
# ---------------------------------------------------------------------------

class MeasuredCIRSource(CIRSource):
    """Load CIR packets from a UA200 dump log into a standard ``CIRData``.

    Mirrors ``CIRSimulator`` from CIRgenerator:

        CIRSimulator.__init__(cfg)  →  MeasuredCIRSource.__init__(cfg)
        CIRSimulator.generate()     →  MeasuredCIRSource.load()
    """

    def __init__(self, wrapper_cfg) -> None:
        self.cfg = create_measured_config(wrapper_cfg)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def load(self) -> CIRData:
        packets = self._parse_log(Path(self.cfg.path))
        cube = self._build_cir_cube(packets)
        metadata = self._build_metadata(cube)
        return CIRData(
            data=cube,
            metadata=metadata,
            truth=None,
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_log(self, path: Path) -> list[dict[str, Any]]:
        """Extract all CIR packets from the log file.

        Returns a list of dicts, each containing at least:
            {sequence, rx, cir, antenna_bitmap, ...}
        """
        packets: list[dict[str, Any]] = []
        errors: int = 0

        with path.open("r", errors="ignore") as f:
            for line in f:
                hx = _extract_read_hex(line)
                if hx is None:
                    continue
                if not _is_candidate_cir_hex(hx):
                    continue

                try:
                    rec = _parse_cir_packet_hex(hx)
                    rec["file"] = str(path)
                    packets.append(rec)
                except Exception:
                    errors += 1

        if errors:
            print(f"[measured] {errors} packet parse error(s) in {path.name}")

        return packets

    # ------------------------------------------------------------------
    # Cube construction
    # ------------------------------------------------------------------

    def _build_cir_cube(self, packets: list[dict[str, Any]]) -> np.ndarray:
        """Build ``(num_frames, num_antennas, num_taps)`` cube from packets.

        - Group packets by ``sequence``.
        - Each group forms one frame; ``rx`` maps to antenna index (rx-1).
        - Missing antennas get zero-filled taps.
        - Sequences are sorted and re-indexed 0..N-1.
        """
        if not packets:
            raise ValueError("No CIR packets found in log")

        # Determine tap count from the majority of packets
        tap_counts = [p["n_taps"] for p in packets]
        num_taps = int(np.median(tap_counts))

        # Group by sequence
        frames: dict[int, dict[int, np.ndarray]] = {}
        for p in packets:
            seq = p["sequence"]
            rx = p["rx"]
            if rx is None:
                continue
            ant_idx = rx - 1  # rx=1 → 0, rx2 → 1
            frames.setdefault(seq, {}).setdefault(ant_idx, [])
            frames[seq][ant_idx].append(p["cir"])

        # Determine antenna count from the data
        ant_indices: set[int] = set()
        for ants in frames.values():
            ant_indices.update(ants.keys())
        num_antennas = max(ant_indices) + 1

        # Sort sequences in order of appearance
        sorted_seqs = sorted(frames.keys())
        num_frames = len(sorted_seqs)

        cube = np.zeros(
            (num_frames, num_antennas, num_taps),
            dtype=np.complex64,
        )

        for frame_idx, seq in enumerate(sorted_seqs):
            frame_data = frames[seq]
            for ant_idx in range(num_antennas):
                taps_list = frame_data.get(ant_idx, [])
                if taps_list:
                    # If multiple entries for same (seq, antenna), average them
                    stacked = np.stack(taps_list, axis=0)
                    avg = np.mean(stacked, axis=0).astype(np.complex64)
                    cube[frame_idx, ant_idx, : min(len(avg), num_taps)] = avg[:num_taps]
                # else: stays zero-initialized (missing antenna)

        return cube
    
    # ------------------------------------------------------------------
    # Metadatga
    # ------------------------------------------------------------------

    def _build_metadata(self, cube: np.ndarray) -> CIRMetadata:
        num_frames, num_antennas, num_taps = cube.shape
        pri_s = self.cfg.radar.period_ms / 1000.0

        return CIRMetadata(
            source="measured",
            num_frames=num_frames,
            num_antennas=num_antennas,
            num_taps=num_taps,
            pri_s=pri_s,
            # fast_time_resolution_ns: derived from hardware spec.
            # UA200 UWB channel 9 @ 998.4 MHz sample rate → ~1.0016 ns
            fast_time_resolution_ns=1.0 / 998.4e6 * 1e9,
            channel=str(self.cfg.radar.channel),
            center_frequency_hz=None,
            scenario_name=self.cfg.log_dirname,
            raw_config=self.cfg,
        )