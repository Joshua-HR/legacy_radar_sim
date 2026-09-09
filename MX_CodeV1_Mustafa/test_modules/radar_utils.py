import numpy as np
import matplotlib.pyplot as plt
import threading
from queue import Queue, Empty
from test_modules.utils import *
from test_modules.constants import SPEED_OF_LIGHT, CHANNEL_FREQ_DICT
import matplotlib.pyplot as plt
from scipy import signal
import os
import shutil


SEGMENT_LENGTH = 16     # number of CIRs for FFT processing
BETA = 60               # number of segments for averaging
THRESHOLD = 8           # dB
BEAM_DIRECTIONS_DEG = [0]
ANTENNA_SPACING = 19e-3
AoA_CAL_VALUE = [0, np.deg2rad(-57 - 5)]    # indoor calibration

HUMAN_DETECTION = False
HUMAN_DETECTION_VERTICAL = False
PLOT_DEBUG_DATA = False
PLOT_CFAR_DEBUG = False
segment_idx = 0

# ------------------------------------------------------------
# New target filtering / grouping / smoothing parameters
# ------------------------------------------------------------
MIN_RANGE_CM = 75.0
MAX_RANGE_CM = 420.0

# In the real no-motion log, weak residual clutter created many false cells.
# Use a conservative default; lower this only after raw CIR is stable
MIN_POWER_DB = -6.0
MAX_ABS_ANGLE_DEG = 60.0

# Ignore quasi-static leakage and the first Doppler side-lobe around DC.
# With period=10 ms and segment_size=16, Doppler bin 1 is about 0.117 m/s
# Bin-offset gating below rejects bin ±1; this value is a second safety gate.
MIN_ABS_DOPPLER_MPS = 0.18
MIN_DOPPLER_BIN_OFFSET = 2

# Group detections that are close to each other
GROUP_RANGE_CM = 45.0
GROUP_ANGLE_DEG = 10.0
GROUP_VELOCITY_MPS = 0.25

# Reject tiny clusters. A real moving body normally produces several adjacent
# range/Doppler cells; one or two cells are usually noise/leakage.
MIN_POINTS_PER_GROUP = 8

# Require the complete radar frame to have enough valid motion evidence.
MIN_FILTERED_TARGETS_PER_FRAME = 25
MIN_GROUP_CONFIRM_FRAMES = 3

# Limit what goes to GUI
MAX_GROUPS_TO_PLOT = 3

# Temporal smoothing
TRACK_ALPHA = 0.65

# Radar robustness controls
SKIP_TAPS = 6                   # skip early leakage/coupling taps; 6 taps ≈ 90 cm
CLUTTER_WARMUP_SEGMENTS = 8     # short no-motion baseline; do not move during this short calibration
EPS_MAG = 1e-12                 # numeric floor for log operations
DISABLE_SIGMA_DELTA_GATE = True # True = do not reject side targets before AoA

# RF health guard: if one RX stream is a fixed/dummy value, tow-antenna AoA is invalid.

RX_DUMMY_RANGE_STD_TH = 1e-5
RX_DUMMY_TIME_STD_TH = 1e-5
ENABLE_SINGLE_RX_FALLBACK = False


USE_DIAGNOSTIC_MOTION_DETECTION = True
DIAG_FORCE_RX_INDEX = 0             # 0 = Rx1/0101. Use RX1 first because RX2 was unstable/frozen in logs.
DIAG_MOTION_EXCESS_DB_TH = 6.0      # dynamic energy must exceed no-motion floor by this amount
DIAG_MOTION_MIN_RATIO_DB = -36.0    # less strict for live walking test
DIAG_MIN_ACTIVE_TAPS = 2            # require several range taps, not one noisy cell
DIAG_CONFIRM_FRAMES = 1
DIAG_BASELINE_ALPHA = 0.01          # update no-motion floor slowly only when decision is NO_MOTION



DIAG_USE_HYBRID_AMPLITUDE_DETECTOR = True
DIAG_AMP_DELTA_DB_TH = 2.5          # tap amplitude must rise above baseline by this amount
DIAG_AMP_Z_TH = 2.0                     # robust z-score relative to no-motion amplitude variation
DIAG_STRONG_AMP_DELTA_DB_TH = 5.0
DIAG_MIN_AMP_ACTIVE_TAPS = 1        # single strong body tap is enought if confirmed over time
DIAG_BASELINE_STD_FLOOR_DB = 0.8
DIAG_RANGE_MIN_CM = 90.0
DIAG_RANGE_MAX_CM = 420.0


# Function to clear all files in a dictionary, optionally including subdirectories
def clear_folder(folder_path, include_subfolders=False):
    """
    Deletes all files inside the specified folder.

    Parameters:
    - folder_path (str): Path to the foler to be cleared.
    - include_subfolders (bool): Whether to delete files in subfolders as well.
    """
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        if not include_subfolders:
            break  # Stop after the top-level folder


def plot_detections(detections, time):
    """
    detections: list of dictionary:
        {
            'range': float (cm),
            'power': float,
            'directions': float (degrees)
        }
    """


    ranges = np.array([d['range'] for d in detections])             # cm
    directions = np.array([d['directions'] for d in detections])    # deg
    powers = np.array([d['power'] for d in detections])


    theta = np.deg2rad(directions)

    x = ranges * np.sin(-(theta))
    y = ranges * np.cos(-(theta))


    plt.figure(figsize=(8, 8))
    sc = plt.scatter(x, y, c=power, s=15**2, cmap='viridis') # s = diameter

    
    cbar = plt.colorbar(sc)
    cbar.set_label("Power")


    plt.xlabel("X [cm]")
    plt.ylabel("Y [cm]")
    plt.title(f"Detection map (range + direction), time = {time}")



    plt.grid(True)
    #plt.axis('equal')
    plt.xlim([-400, 400])
    plt.ylim([-50, 800])
    plt.savefig(f'log/detection/position_time_{time:.2f}.png', dpi=300)
    plt.close()
    #plt.show(block = False)


def plot_detection_map(detection_map, segment_idx):
    fig = plt.figure()
    plt.imshow(detection_map)
    fig.savefig(f'log/map/detection_map_{segment_idx}.png', dpi=300)
    plt.close(fig)


def calculate_beamforming_weights(frequency, dir, d, N=2, differential_beam = False):
    """
    Calculate beamforming weights for given direction and antenna spacing

    :param frequency: carrier frequency in Hz
    :param dir: direction of main lobe
    :param d - antenna distance (assumption all distances are the same)
    :param N - number of antenna elements
    :param differential_beam - calculate weights for differential beam
    """
    lamb = SPEED_OF_LIGHT/frequency #wavelength lambda
    phases = np.zeros(N,dtype = float)
    for i in range(N):
        phases[i] = -2*np.pi/lamb*d*i*np.sin(np.deg2rad(dir))
    if differential_beam:
        for i in range(N>>1,N,1):
            phases[i] += np.pi
    return np.exp(1j*phases)



def CFAR(data_dB, threshold, sigma_grater_delta_indicator = None):
    """
    CFAR function for detecting targets
    Constant false alarm rate (CFAR) detection is a commmon form of adaptive algorithm used in radar systems
    to detect target returns against a background of noise, clutter and interference.
    description source: https://en.wikipedia.org/wiki/Constant_false_alarm_rate

    :param data: FFT response of collected data in dB scale: data_dB = 10*np.log10(np.abs(data_complex))
    :param threshold: threshold in dB for checking if there is a target
    :param sigma_grater_indicator: difference between sigma and delta beams in dB, if >0 target in scope
    :return: Detection map
    """

    if PLOT_DEBUG_DATA and PLOT_CFAR_DEBUG:
        global segment_idx
        if not os.path.exists(f'log/doppler/CFAR/{segment_idx}'):
            os.makedirs(f'log/doppler/CFAR/{segment_idx}')
    n = 8 # number of cells to use for calculating the average power level
    k = 3 # number of guard cells
    l = 2 # number of cell under test

    #data_dB = 10*np.log10(np.abs(data_dB))
    data_dB = np.nan_to_num(data_dB, nan=-300.0, posinf=300.0, neginf=-300.0)
    if sigma_grater_delta_indicator is None:
        sigma_grater_delta_indicator = np.ones_like(data_dB, dtype=bool)
    else:
        sigma_grater_delta_indicator = sigma_grater_delta_indicator.astype(bool)

    detection_map = np.zeros(data_dB.shape, dtype = int)



    M = data_dB.shape[1]
    for i in range(data_dB.shape[0]):
        if PLOT_DEBUG_DATA and PLOT_CFAR_DEBUG:
            fig = plt.figure()
            loc_cfar = []
        for j in range(M):
            if j < n+k+l:
                av = np.mean(data_dB[i,j+k+l:j+n+k+l+1])
            elif j > M-n-k:
                av = np.mean(data_dB[i,j-k-n:j-k+1])
            else:
                av1 = np.mean(data_dB[i,j+k+l:j+n+k+l+1])
                av2 = np.mean(data_dB[i,j-k-n:j-k+1])
                av = av1 if av1 > av2 else av2
            if (np.mean(data_dB[i,j:j+l]) - av > threshold) and sigma_grater_delta_indicator[i,j]:
                detection_map[i,j] = 1
            if PLOT_DEBUG_DATA and PLOT_CFAR_DEBUG:
                loc_cfar.append(np.mean(data_dB[i,j:j+l]) - av)
        if PLOT_DEBUG_DATA and PLOT_CFAR_DEBUG:
            plt.plot(loc_cfar)
            plt.plot(data_dB[i,:])
            plt.grid()
            fig.savefig(f'log/doppler/CFAR/{segment_idx}/{i}.png', dpi=300)
    return detection_map

def addaptive_thresholding(data_lin, threshold, sigma_grater_delta_indicator = None):
    """
    Docstring for addaptive_thresholding

    :param data_lin: Range doppler data in linear scale (only magnitude)
    :param threshold: Description
    :param sigma_grater_delta_indicator: indicator which show if signals comes from main beam
    :return detection_map: bool map which indicate if detection occur for doppler range cell
    """


    pass

class RadarPostprocessing():
    """
        RadarPostprocessing - class for collecting and analyzing Radar CIR data
    """
    def __init__(self, ant_num, slow_time_size, fast_time_size, PRI, fast_time_resolution_ns, channel, beam_dirs_deg:np.ndarray, CFAR_threshold, target_queue:Queue, use_threads=True):

        self.segment = np.zeros((ant_num, slow_time_size, fast_time_size), dtype=np.complex64)
        self.valid = np.zeros((ant_num, slow_time_size), dtype=bool)

        self.pending_pairs_by_seq = {}
        self.last_committed_sequence = None
        self.slow_write_idx = 0
        self.current_segment_id = 0
        self.dropped_unpaired_packets = 0
        self.max_pending_sequences = 8
        self.confirm_counter = 0
        self.clutter_map = np.ones((beam_dirs_deg.size, slow_time_size, fast_time_size), dtype=np.float32)
        self.PRI = PRI
        self.fast_time_resolution_ns = fast_time_resolution_ns
        self.frequency = CHANNEL_FREQ_DICT[str(channel)]
        self.segment_idx = 0
        self.use_threads = use_threads
        self.num_beams = beam_dirs_deg.size
        self.beam_dirs_deg = beam_dirs_deg
        self.beam_weights = np.zeros((self.num_beams, ant_num), dtype=np.complex64)
        self.diff_beam_weights = np.zeros((self.num_beams, ant_num), dtype=np.complex64)
        self.CFAR_threshold = CFAR_threshold
        self.target_queue = target_queue
        self.queue_plot = target_queue
        self.status_array = {0:[],1:[],2:[]}
        self.decision = [False, False, False]


        self.prev_grouped_targets = []

        self.diag_baseline_db = None
        self.diag_amp_baseline_db = None
        self.diag_amp_std_db = None
        self.diag_confirm_counter = 0
        self.diag_frame_counter = 0


        for i in range(self.num_beams):
            self.beam_weights[i,:] = calculate_beamforming_weights(self.frequency, beam_dirs_deg[i], ANTENNA_SPACING, N=ant_num)
            self.diff_beam_weights[i,:] = calculate_beamforming_weights(self.frequency, beam_dirs_deg[i], ANTENNA_SPACING, N=ant_num, differential_beam = True)

        self.doppler_frequency = np.fft.fftshift(np.fft.fftfreq(slow_time_size, d=self.PRI))
        range_step_cm = fast_time_resolution_ns*1e-9 * SPEED_OF_LIGHT * 100 / 2 # multiply by 100 to change from m to cm, divided by 2 because distance is 2 times in radar
        self.radar_range = np.arange(0,self.segment.shape[2]*range_step_cm,range_step_cm)
        self.doppler_velocity = -self.doppler_frequency*SPEED_OF_LIGHT/self.frequency/2
        self.do_postprocessing = True
        self.beta = 1

        if self.use_threads:

            self.segment_queue = Queue(maxsize=1)
            self.postproc_thread = threading.Thread(target=self.worker, daemon=True)
            self.postproc_thread.start()

    def _put_latest_targets(self, detections):
        """Put only the newest GUI result; never block the radar/postprocessing thread."""
        try:
            while True:
                self.target_queue.get_nowait()
        except Empty:
            pass
        try:
            self.target_queue.put_nowait(detections)
        except Exception:
            pass


    def fill_data_by_hex(self, cir_hex, ant_bitmap, st_idx):
        """
        Convert one CIR notification into the segment buffer.

        V3 pairing rule:
        - Prefer exact sequence pairing: ANT1 and ANT2 must have the same radar sequence.
        - This avoids artificial Doppler caused by pairing packets from different radar snapshots.
        - Very old incomplete sequence entries are dropped instead of being used.
        """
        try:
            ant_idx = int(ant_bitmap[:2], 16) - 1  # UA200 bitmap: 0101 -> ANT1, 0201 -> ANT2
        except Exception:
            print(f"Drop packet: invalid ant_bitmap={ant_bitmap}")
            return

        if ant_idx < 0 or ant_idx >= self.segment.shape[0]:
            print(f"Drop packet: ant_bitmap={ant_bitmap} maps to invalid ant_idx={ant_idx}")
            return

        sequence_id = int(st_idx) + 1

        # Ignore old duplicate packets after a sequence was already committed.
        if self.last_committed_sequence is not None and sequence_id <= self.last_committed_sequence:
            return

        cir_vec = np.zeros((self.segment.shape[2],), dtype=np.complex64)
        for ft_idx in range(self.segment.shape[2]):
            try:
                cir_vec[ft_idx] = hex_to_complex_q8_8(cir_hex[ft_idx * 8:(ft_idx + 1) * 8])
            except Exception:
                cir_vec[ft_idx] = 0
        
        pair = self.pending_pairs_by_seq.setdefault(sequence_id, {})
        pair[ant_idx] = cir_vec

        # Drop old incomplete sequences to avoid memory growth and stable pairing.
        if len(self.pending_pairs_by_seq) > self.max_pending_sequences:
            old_sequences = sorted(self.pending_pairs_by_seq.keys())[:-self.max_pending_sequences]
            dropped = 0
            for seq in old_sequences:
                dropped += len(self.pending_pairs_by_seq.get(seq, {}))
                self.pending_pairs_by_seq.pop(seq, None)
            self.dropped_unpaired_packets += dropped
            if dropped and (self.dropped_unpaired_packets <= 20 or self.dropped_unpaired_packets % 50 == 0):
                print(
                    f"Drop stale unpaired sequence packet(s), total={self.dropped_unpaired_packets}, "
                    f"latest_sequence={sequence_id}"
                )
        
        # Wait until the same sequence contains all expected antennas.
        if len(pair) < self.segment.shape[0]:
            return

        # Commit complete sequence.
        for rx_idx in range(self.segment.shape[0]):
            if rx_idx not in pair:
                return
            self.segment[rx_idx, self.slow_write_idx, :] = pair[rx_idx]
            self.valid[rx_idx, self.slow_write_idx] = True

        self.pending_pairs_by_seq.pop(sequence_id, None)
        self.last_committed_sequence = sequence_id
        self.slow_write_idx += 1

        if self.slow_write_idx >= self.segment.shape[1]:
            process_this_segment = bool(np.all(self.valid))
            segment_copy = self.segment.copy() if process_this_segment else None
            if not process_this_segment:
                missing = int(self.valid.size - np.count_nonzero(self.valid))
                print(f"Skip incomplete RX segment {self.current_segment_id}: missing_cells={missing}")

            # Reset the live write buffer before any postprocessing. This prevents
            # slow_write_idx staying at 16 if postprocessing throws an exception.
            self.segment.fill(0)
            self.valid.fill(False)
            self.slow_write_idx = 0
            self.current_segment_id += 1

            if process_this_segment:
                self.segment_idx += 1
                if self.use_threads:
                    try:
                        while True:
                            self.segment_queue.get_nowait()
                            self.segment_queue.task_done()
                    except Empty:
                        pass
                    try:
                        self.segment_queue.put_nowait(segment_copy)
                    except Exception:
                        pass
                else:
                    if HUMAN_DETECTION:
                        self.run_postprocessing_human_detection(segment_copy)
                    else:
                        self.run_postprocessing(segment_copy)
    

    def _process_completed_segment(self):
        """Process a completed segment using a copy so the live CIR buffer can be reset safely."""
        self.segment_idx += 1
        segment_copy = self.segment.copy()
        if self.use_threads:
            try:
                while True:
                    self.segment_queue.get_nowait()
                    self.segment_queue.task_done()
            except Empty:
                pass
            try:
                self.segment_queue.put_nowait(segment_copy)
            except Exception:
                pass
        else:
            if HUMAN_DETECTION:
                self.run_postprocessing_human_detection(segment_copy)
            else:
                self.run_postprocessing(segment_copy)
            if PLOT_DEBUG_DATA:
                try:
                    plot_detections(self.target_queue.get_nowait(), SEGMENT_LENGTH * self.PRI * self.segment_idx)
                except Exception:
                    pass




    def get_doppler_frequency(self):
        """
            return Doppler frequencies in form from -PRF/2 to PRF/2
        """
        return self.doppler_frequency

    def get_radar_range(self):
        """
            return distance values in cm from 0 to max range of radar (depends on fast time size)
        """
        return self.radar_range
    
    def get_doppler_velocity(self):
        """
            retrun Doppler velocity values
        """
        return self.doppler_velocity

    def calculate_doppler_response(self, segment:np.ndarray, weights = None):
        """
            calculate doppler response as fourier transform of sequencialy received CIR packets
        """
        if weights == None:
            weights = np.ones(segment.shape[0])
        
        time_data = np.zeros((segment.shape[1],segment.shape[2]), dtype=np.complex64)

        filter_weights = signal.windows.taylor(segment.shape[1], nbar=20, sll=40, norm=True)
        filter_weights = np.ones_like(filter_weights)

        for i in range(len(weights)):
            time_data += weights[i]*segment[i,:,:]

        return np.fft.fftshift(np.fft.fft(time_data*filter_weights[:, np.newaxis], axis=0),axes = 0)


    def plot_time_response(self, idx = None, taps_on_xaxis = False):
        """
        Plot CIR data in time domain for idx packet or for all

        :param idx: slice of data
        """
        if idx == None:
            idx = np.arange(0,self.segment.shape[1], 1)
        if taps_on_xaxis:
            xaxis_data = np.arange(0, self.segment.shape[2],1)
        else:
            xaxis_data = self.get_radar_range()

        fig, axs = plt.subplots(self.segment.shape[0], 1)
        if type(axs) is not np.ndarray:
            axs = np.array([axs])
        for i in range(self.segment.shape[0]):
            for st_idx in idx:
                axs[i].plot(xaxis_data, 10*np.log10(np.abs(self.segment[i,st_idx,:])))
            axs[i].grid(True)

    def plot_doppler_response(self, weights = None, type = 'velocity'):
        """
        Plot doppler response as imshow

        :param weights: weights for CIRs collected by antennas - digital beamforming
        :param type: 'velocity' or 'frequency' - what x-axis should be used for plotting
        """

        plt.figure()
        dr = 10*np.log10(np.abs(self.get_doppler_response(weights)))
        max_val = np.ceil(dr.max()/5)*5
        min_val = max_val - 50
        pos = plt.imshow(dr, vmin = min_val, vmax = max_val)
        plt.grid()
        plt.colorbar(pos)

    def plot_doppler_response_in_range(self, weights = None, idx = None):
        """
        doppler response in range for each doppler channel

        :param weights: weights for CIRs collected by antennas - digitfal beamforming
        :param idx: slice of data
        """
        if idx == None:
            idx = np.arange(0,self.segment.shape[1],4)
        
        doppler_response = 10*np.log10(np.abs(self.get_doppler_response(weights)))

        plt.figure()
        for df_idx in idx: # df - doppler frequency channel
            plt.plot(self.get_radar_range(), doppler_response[df_idx,:])
        plt.grid(True)

    def update_clutter_map(self, beam_idx, doppler_response_msg):
        """
        Update clutter map with new data magnitude - moving averaging
        Clutter map is created for all beams

        :param beam_idx - index of beam which show for which beam data comes and need to be updated
        :param doppler_response_msg - magnitude of doppler response for new segment
        """
        self.clutter_map[beam_idx,:,:] *= (self.beta - 1)/self.beta
        self.clutter_map[beam_idx,:,:] += 1/self.beta * doppler_response_msg.copy()
        if self.beta < BETA and (beam_idx == self.num_beams - 1): # update beta only after processing of last beam
            self.beta += 3 # increase beta every 3 to not be sensitive on first segments

    def target_parameter_estimation(self, doppler_response: np.ndarray, detection_map: np.ndarray, beam_direction_deg: float):
        """
        Convert detection map into raw target list.
        Each detected cell becomes a raw target candidate.
        """
        lamb = SPEED_OF_LIGHT / self.frequency
        targets = []

        for doppler_idx in range(detection_map.shape[0]):
            for range_idx in range(SKIP_TAPS, detection_map.shape[1]):
                if detection_map[dopper_idx, range_idx] > 0:
                    p12 = doppler_response[0, doppler_idx, range_idx] * np.conj(doppler_response[1, doppler_idx, range_idx])
                    phase_term = lamb * np.angle(p12) / (2 * np.pi * ANTENNA_SPACING)

                    # protect asin from slight overflow
                    phase_term = np.clip(phase_term, -1.0, 1.0)

                    angle_deg = np.rad2deg(np.arcsin(phase_term)) + beam_direction_deg
                    power_lin = np.sum(np.abs(doppler_response[:, doppler_idx, range_idx]) ** 2)
                    power_db = 10 * np.log10(np.maximum(power_lin, EPS_MAG))
                    vel_mps = self.doppler_velocity[doppler_idx]

                    targets.append({
                        'range': float(self.radar_range[range_idx]),
                        'power': float(power_db),
                        'direction': float(angle_deg),
                        'velocity': float(vel_mps),
                        'doppler_idx': int(doppler_idx),
                        'range_idx': int(range_idx),
                    })
        
        return targets

    
    def filter_targets(self, targets):
        """
        Keep only valid moving targets and reject weak / out-of-range / static-clutter.
        """
        filtered = []

        for t in targets:
            if t['range'] < MIN_RANGE_CM or t['range'] > MAX_RANGE_CM:
                continue
            if abs(t['direction']) > MAX_ABS_ANGLE_DEG:
                continue
            if t['power'] < MIN_POWER_DB:
                continue

            # Reject DC and the first Doppler side-lobe around DC. This is important
            # for no-motion cases where static leakage spreads into adjacent bins.
            dc_idx = len(self.doppler_velocity) // 2
            if abs(int(t.get('doppler_idx', dc_idx)) - dc_idx) < MIN_DOPPLER_BIN_OFFSET:
                continue
            
            if abs(t['velocity']) < MIN_ABS_DOPPLER_MPS:
                continue

            filtered.append(t)

        return filtered


    def group_targets(self, targets):
        """
        Group raw detections that belong to the same moving person.
        We cluster by proximity in range, angle, and velocity.
        """
        if not targets:
            return []

        # sort by power strongest first
        targets = sorted(targets, key=lambda x: x['power'], reverse=True)

        groups = []

        for t in targets:
            assigned = False

            for g in groups:
                if (
                    abs(t['range'] - g['range']) <= GROUP_RANGE_CM and
                    abs(t['direction'] - g['direction']) <= GROUP_ANGLE_DEG and
                    abs(t['velocity'] - g['velocity']) <= GROUP_VELOCITY_MPS
                ):
                    # weighted update using linear power weight
                    w_old = g['weight_sum']
                    w_new = max(10 ** (t['power'] / 10.0), 1e-12)
                    w_total = w_old + w_new

                    g['range'] = (g['range'] * w_old + t['range'] * w_new) / w_total
                    g['direction'] = (g['direction'] * w_old + t['direction'] * w_new) / w_total
                    g['velocity'] = (g['velocity'] * w_old + t['velocity'] * w_new) / w_total
                    g['power'] = max(g['power'], t['power'])
                    g['count'] += 1
                    g['weight_sum'] = w_total

                    assigned = True
                    break

            if not assigned:
                groups.append({
                    'range': t['range'],
                    'direction': t['direction'],
                    'velocity': t['velocity'],
                    'power': t['power'],
                    'count': 1,
                    'weight_sum': max(10 ** (t['power'] / 10.0), 1e-12)
                })

        # reject too-small groups
        grouped = []
        for g in groups:
            if g['count'] >= MIN_POINTS_PER_GROUP:
                grounped.append({
                    'range': float(g['range']),
                    'direction': float(g['direction']),
                    'velocity': float(g['velocity']),
                    'power': float(g['power']),
                    'count': int(g['count']),
                })

        # strongest first
        grouped = sorted(grouped, key=lambda x: (x['count'], x['power']), reverse=True)

        # limit number shown on GUI
        grouped = grouped[:MAX_GROUPS_TO_PLOT]

        return grouped


    def smooth_grouped_targets(self, grouped_targets):
        """
        Smooth grouped targets over time so one person does not jump frame to frame.
        Nearest-neighbor smoothing with previous frame.
        """
        if not grouped_targets:
            self.prev_grouped_targets = []
            return []

        if not self.prev_grouped_targets:
            self.prev_grouped_targets = [dect(t) for t in grouped_targets]
            return grouped_targets

        smoothed = []
        used_prev = set()

        for curr in grouped_targets:
            best_idx = None
            best_score = 1e9

            for idx, prev in enumerate(self.prev_grouped_targets):
                if idx in used_prev:
                    continue

                score = (
                    abs(curr['range'] - prev['range']) / max(GROUP_RANGE_CM, 1e-6) +
                    abs(curr['direction'] - prev['direction']) / max(GROUP_ANGLE_DEG, 1e-6)
                )

                if score < best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None and best_score < 3.0:
                prev = self.prev_grouped_targets[best_idx]
                used_prev.add(best_idx)

                smooth_target = {
                    'range': TRACK_ALPHA * prev['range'] + (1 - TRACK_ALPHA) * curr['range'],
                    'direction': TRACK_ALPHA * prev['direction'] + (1 - TRACK_ALPHA) * curr['direction'],
                    'velocity': TRACK_ALPHA * prev.get('velocity', curr['velocity']) + (1 - TRACK_ALPHA) * curr['velocity'],
                    'power': curr['power'],
                    'count': curr.get('count', 1),
                }
                smoothed.append(smooth_target)
            else:
                smoothed.append(curr)

        self.prev_grouped_targets = [dict(t) for t in smoothed]
        return smoothed



    def rx_health_check(self, segment):
        """Return active RX mask and print diagnostics for dummy/frozen RX streams."""
        active = []
        for ant_idx in range(segment.shape[0]):
            mag = np.abs(segment[ant_idx])
            range_std = float(np.std(np.mean(mag, axis=0)))
            time_std = float(np.mean(np.std(segment[ant_idx], axis=0)))
            is_dummy = (range_std < RX_DUMMY_RANGE_STD_TH and time_std < RX_DUMMY_TIME_STD_TH)
            active.append(not is_dummy)
            if is_dummy and (self.segment_idx <= CLUTTER_WARMUP_SEGMENTS + 2 or self.segment_idx % 20 == 0):
                print(
                    f"[RF_CHECK] RX{ant_idx+1} looks dummy/frozen: "
                    f"range_std={range_std:.3e}, time_std={time_std:.3e}. "
                    "AoA with 2 RX is not valid for this frame."
                )
        return np.array(active, dtype=bool)

    def _has_adjacent_true(self, mask, min_len):
        """Return True if a boolean vector has at least min_len consecutive True values."""
        run = 0
        for v in mask:
            if bool(v):
                run += 1
                if run >= min_len:
                    return True
            else:
                run = 0
        return False

    def run_diagnostic_motion_detector(self, segment, active_rx_mask):
        """
        V8 RF-focused hybrid movement detector.

        Why V8:
        - The V6/V7 dynamic/static ratio-only detector was too conservative.
        - Walking can appear as a range-profile amplitude change, especially with one usable RX.
        - Therefore V8 uses two independent checks:
            1) amplitude change against the no-motion range profile baseline;
            2) slow-time dynamic/static ratio against the no-motion dynamic baseline.
        
        This gives a practical movement detector while still avoiding the old CFAR/grouping
        false-positive behavior in a static room.
        """
        if DIAG_FORCE_RX_INDEX is not None and DIAG_FORCE_RX_INDEX < segment.shape[0]:
            rx_idx = int(DIAG_FORCE_RX_INDEX)
        else:
            active_indices = np.where(active_rx_mask)[0]
            if active_indices.size == 0:
                self._put_latest_targets([])
                print('[MOTION_DIAG] no valid RX stream; decision=NO_MOTION')
                return
            rx_idx = int(active_indices[0])

        x = segment[rx_idx, :, :].astype(np.complex64)
        if x.shape[1] <= SKIP_TAPS + 2:
            self._put_latest_targets([])
            print('[MOTION_DIAG] not enough CIR taps after leakage skip; decision=NO_MOTION')
            return

        x_valid = x[:, SKIP_TAPS:]
        valid_ranges = self.radar_range[SKIP_TAPS:SKIP_TAPS + x_valid.shape[1]]
        range_mask = (valid_ranges >= DIAG_RANGE_MIN_CM) & (valid_ranges <= DIAG_RANGE_MAX_CM)
        if not np.any(range_mask):
            self._put_latest_targets([])
            print('[MOTION_DIAG] no valid range bins in diagnostic ROI; decision=NO_MOTION')
            return

        # Amplitude/range-profile branch: compare current average magnitude profile
        # against the baseline learned during the no-motion warm-up.
        amp_profile_db = 20.0 * np.log10(np.mean(np.abs(x_valid), axis=0) + EPS_MAG)

        # Dynamic/phase branch: remove static complex component per tap and estimate
        # dynamic/static energy ratio. This is useful for micro-motion or vibration.
        static_power = np.mean(np.abs(x_valid) ** 2, axis=0) + EPS_MAG
        x_static_removed = x_valid - np.mean(x_valid, axis=0, keepdims=True)
        dynamic_power = np.mean(np.abs(x_static_removed) ** 2, axis=0) + EPS_MAG
        ratio_db = 10.0 * np.log10(dynamic_power / static_power)

        # Warm-up baseline: user must not move here.
        if self.segment_idx <= CLUTTER_WARMUP_SEGMENTS:
            if self.diag_baseline_db is None:
                self.diag_baseline_db = ratio_db.copy()
                self.diag_amp_baseline_db = amp_profile_db.copy()
                self.diag_amp_std_db = np.ones_like(amp_profile_db) * DIAG_BASELINE_STD_FLOOR_DB
            else:
                n = max(self.segment_idx, 1)
                old_amp = self.diag_amp_baseline_db.copy()
                self.diag_baseline_db = ((n - 1) / n) * self.diag_baseline_db + (1 / n) * ratio_db
                self.diag_amp_baseline_db = ((n - 1) / n) * self.diag_amp_baseline_db + (1 / n) * amp_profile_db
                abs_dev = np.abs(amp_profile_db - old_amp)
                self.diag_amp_std_db = np.maximum(
                    ((n - 1) / n) * self.diag_amp_std_db + (1 / n) * abs_dev,
                    DIAG_BASELINE_STD_FLOOR_DB
                )
            self.diag_confirm_counter = 0
            print(f'[SmartCal] diagnostic no-motion baseline warm-up {self.segment_idx}/{CLUTTER_WARMUP_SEGMENTS} using RX{rx_idx+1}')
            self._put_latest_targets([])
            return

        if (self.diag_baseline_db is None or self.diag_baseline_db.shape != ratio_db.shape or
                self.diag_amp_baseline_db is None or self.diag_amp_baseline_db.shape != amp_profile_db.shape):
            self.diag_baseline_db = ratio_db.copy()
            self.diag_amp_baseline_db = amp_profile_db.copy()
            self.diag_amp_std_db = np.ones_like(amp_profile_db) * DIAG_BASELINE_STD_FLOOR_DB
            self.diag_confirm_counter = 0
            self._put_latest_targets([])
            print('[MOTION_DIAG] baseline initialized after warm-up; decision=NO_MOTION')
            return

        amp_delta_db = amp_profile_db - self.diag_amp_baseline_db
        amp_z = amp_delta_db / np.maximum(self.diag_amp_std_db, DIAG_BASELINE_STD_FLOOR_DB)

        ratio_excess_db = ratio_db - self.diag_baseline_db
        ratio_mask = (ratio_excess_db >= DIAG_MOTION_EXCESS_DB_TH) & (ratio_db >= DIAG_MOTION_MIN_RATIO_DB) * range_mask
        amp_mask = (amp_delta_db >= DIAG_AMP_DELTA_DB_TH) & (amp_z >= DIAG_AMP_Z_TH) & range_mask
        strong_amp_mask = (amp_delta_db >= DIAG_STRONG_AMP_DELTA_DB_TH) & range_mask
        motion_mask = ratio_mask | amp_mask | strong_amp_mask

        active_taps = int(np.count_nonzero(motion_mask))
        amp_active_taps = int(np.count_nonzero(amp_mask | strong_amp_mask))
        ratio_active_taps = int(np.count_nonzero(ratio_mask))

        if active_taps > 0:
            score = np.where(motion_mask, np.maximum(amp_delta_db, ratio_excess_db), -1e9)
            strongest_local_idx = int(np.argmax(score))
        else:
            # Still report the most suspicious range for debugging.
            combined_score = np.maximum(amp_delta_db, ratio_excess_db)
            combined_score = np.where(range_mask, combined_score, -1e9)
            strongest_local_idx = int(np.argmax(combined_score))

        strongest_tap = strongest_local_idx + SKIP_TAPS
        strongest_range_cm = float(self.radar_range[strongest_tap]) if strongest_tap < len(self.radar_range) else float('nan')
        max_ratio_db = float(np.max(ratio_db[range_mask]))
        max_ratio_excess_db = float(np.max(ratio_excess_db[range_mask]))
        max_amp_delta_db = float(np.max(amp_delta_db[range_mask]))
        max_amp_z = float(np.max(amp_z[range_mask]))

        candidate_motion = active_taps >= DIAG_MIN_AMP_ACTIVE_TAPS
        if candidate_motion:
            self.diag_confirm_counter += 1
        else:
            self.diag_confirm_counter = 0
            self.diag_baseline_db = (1.0 - DIAG_BASELINE_ALPHA) * self.diag_baseline_db + DIAG_BASELINE_ALPHA * ratio_db
            self.diag_amp_baseline_db = (1.0 - DIAG_BASELINE_ALPHA) * self.diag_amp_baseline_db + DIAG_BASELINE_ALPHA * amp_profile_db
            self.diag_amp_std_db = np.maximum(
                (1.0 - DIAG_BASELINE_ALPHA) * self.diag_amp_std_db + DIAG_BASELINE_ALPHA * np.abs(amp_delta_db),
                DIAG_BASELINE_STD_FLOOR_DB
            )
        
        if self.diag_confirm_counter >= DIAG_CONFIRM_FRAMES:
            decision = 'MOTION'
            target = {
                'range': strongest_range_cm,
                'direction': 0.0,
                'velocity': 0.0,
                'power': max(max_amp_delta_db, max_ratio_excess_db),
                'count': active_taps,
                'classification': 'Hybrid motion RX1',
                'human_score': max(0.0, min(1.0, max(max_amp_delta_db, max_ratio_excess_db) / 25.0)),
            }
            self._put_latest_targets([target])
        elif candidate_motion:
            decision = 'WAIT_CONFIRM'
            self._put_latest_targets([])
        else:
            decision = 'NO_MOTION'
            self._put_latest_targets([])

        print(
            f'[MOTION_DIAG_V9] rx=RX{rx_idx+1}, amp_delta={max_amp_delta_db:.1f}dB, '
            f'amp_z={max_amp_z:.1f}, ratio={max_ratio_db:.1f} dB, '
            f'ratio_excess={max_ratio_excess_db:.1f} dB, active={active_taps} '
            f'(amp={amp_active_taps}, ratio={ratio_active_taps}), '
            f'range={strongest_range_cm:.0f} cm, confirm={self.diag_confirm_counter}/{DIAG_CONFIRM_FRAMES}, '
            f'decision={decision}'
        )
    def run_single_rx_motion_postprocessing(self, segment, active_rx_mask):
        """Fallback movement detection using one valid RX channel. Direction is set to 0 deg because AoA is unavailable."""
        active_indices = np.where(active_rx_mask)[0]
        if active_indices.size == 0:
            self._put_latest_targets([])
            return

        rx_idx = int(active_indices[0])
        doppler_response = self.calculate_doppler_response(segment[rx_idx:rx_idx+1, :, :])
        beam_idx = 0

        if self.segment_idx <= CLUTTER_WARMUP_SEGMENTS:
            self.update_clutter_map(beam_idx, np.abs(doppler_response))
            print(f'[SmartCal] single-RX clutter warm-up {self.segment_idx}/{CLUTTER_WARMUP_SEGMENTS} using RX{rx_idx+1}')
            self._put_latest_targets([])
            return

        doppler_response_wo_clutter_dB = (
            10 * np.log10(np.maximum(np.abs(doppler_response), EPS_MAG))
            - 10 * np.log10(np.maximum(np.abs(self.clutter_map[beam_idx, :, :]), EPS_MAG))
        )
        self.update_clutter_map(beam_idx, np.abs(doppler_response))

        gate_mask = np.ones_like(doppler_response_wo_clutter_dB, dtype=bool)
        detection_map = CFAR(doppler_response_wo_clutter_dB, self.CFAR_threshold, gate_mask)

        targets = []
        dc_idx = len(self.doppler_velocity) // 2
        for doppler_idx in range(detection_map.shape[0]):
            if abs(doppler_idx - dc_idx) < MIN_DOPPLER_BIN_OFFSET:
                continue
            vel_mps = float(self.doppler_velocity[doppler_idx])
            if abs(vel_mps) < MIN_ABS_DOPPLER_MPS:
                continue
            for range_idx in range(SKIP_TAPS, detection_map.shape[1]):
                if detection_map[doppler_idx, range_idx] <= 0:
                    continue
                power_lin = float(np.abs(doppler_response[doppler_idx, range_idx]) ** 2)
                power_db = 10 * np.log10(max(power_lin, EPS_MAG))
                t = {
                    'range': float(self.radar_range[range_idx]),
                    'power': float(power_db),
                    'direction': 0.0,
                    'velocity': vel_mps,
                    'doppler_idx': int(doppler_idx),
                    'range_idx': int(range_idx),
                }
                targets.append(t)

        raw_target_count = len(targets)
        filtered_targets = self.filter_targets(targets)
        if len(filtered_targets) < MIN_FILTERED_TARGETS_PER_FRAME:
            self.confirm_counter = 0
            self.prev_grouped_targets = []
            print(f'[SINGLE_RX] raw targets: {raw_target_count}, filtered: {len(filtered_targets)}, grouped: 0, decision=NO_MOTION')
            self._put_latest_targets([])
            return

        grouped_targets = self.group_targets(filtered_targets)
        if grouped_targets:
            self.confirm_counter += 1
        else:
            self.confirm_counter = 0

        if self.confirm_counter < MIN_GROUP_CONFIRM_FRAMES:
            print(f'[SINGLE_RX] raw targets: {raw_target_count}, filtered: {len(filtered_targets)}, grouped: {len(grouped_targets)}, decision=WAIT_CONFIRM')
            self._put_latest_targets([])
            return

        grouped_targets = self.smooth_grouped_targets(grouped_targets)
        print(f'[SINGLE_RX] raw targets: {raw_target_count}, filtered: {len(filtered_targets)}, grouped: {len(grouped_targets)}, decision=MOTION')
        self._put_latest_targets(grouped_targets)


    def run_postprocessing(self, segment):
        """
        run all postprocessing to detect targets from segment of CIRs
        """

        active_rx_mask = self.rx_health_check(segment)
        if USE_DIAGNOSTIC_MOTION_DETECTION:
            self.run_diagnostic_motion_detector(segment, active_rx_mask)
            return

        if np.count_nonzero(active_rx_mask) < 2:
            if ENABLE_SINGLE_RX_FALLBACK:
                self.run_single_rx_motion_postprocessing(segment, active_rx_mask)
            else:
                print('[RF CHECK] Less than 2 valid RX channels. Dropping fame because AoA is invalid.')
                self._put_latest_targets([])
            return

        doppler_responses = np.zeros_like(segment)
        for ant_idx in range(segment.shape[0]):
            doppler_responses[ant_idx,:,:] = self.calculate_doppler_response(segment[ant_idx,:,:][None, :, :]) * np.exp(1j * AoA_CAL_VALUE[ant_idx])
        targets = []
        if PLOT_DEBUG_DATA:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
            fig2, axes2 = plt.subplots(nrows=3, ncols=3, figsize=(15, 9))

        for beam_idx in range(self.num_beams):
            doppler_response = np.zeros((doppler_responses.shape[1], doppler_responses.shape[2]), dtype=np.complex64)
            doppler_response_diff = np.zeros_like(doppler_response)
            for ant_idx in range(segment.shape[0]):
                doppler_response += self.beam_weights[beam_idx, ant_idx]*doppler_response[ant_idx, :, :]
                doppler_response_diff += self.diff_beam_weights[beam_idx, ant_idx]*doppler_response[ant_idx, :, :]

            sigma_grater_delta_indicator = np.abs(doppler_response) - np.abs(doppler_response_diff) > 0

            doppler_response_wo_clutter_dB = 10 * np.log10(np.maximum(np.abs(doppler_response.copy()), EPS_MAG)) - 10 * np.log10(np.maximum(np.abs(self.clutter_map[beam_idx, :, :]), EPS_MAG))


            global segment_idx
            segment_idx = self.segment_idx
            if self.segment_idx <= CLUTTER_WARMUP_SEGMENTS:
                self.update_clutter_map(beam_idx, np.abs(doppler_response))
                continue

            if DISABLE_SIGMA_DELTA_GATE:
                gate_mask = np.ones_like(sigma_grater_delta_indicator, dtype=bool)
            else:
                gate_mask = sigma_grater_delta_indicator

            detection_map = CFAR(doppler_response_wo_clutter_dB, self.CFAR_threshold, gate_mask)

            if PLOT_DEBUG_DATA:
                axes[1, beam_idx].imshow(detection_map)

            if PLOT_DEBUG_DATA:
                axes2[0, beam_idx].plot((np.transpose(doppler_response_wo_clutter_dB)))
                axes2[0, beam_idx].grid(True)

                axes2[1, beam_idx].plot((np.transpose(10 * np.log10(np.maximum(np.abs(doppler_response.copy()), EPS_MAG)))))
                axes2[1, beam_idx].grid(True)

                axes2[2, beam_idx].plot((np.transpose(10 * np.log10(np.maximum(np.abs(self.clutter_map[beam_idx,:,:]), EPS_MAG)))))
                axes2[2, beam_idx].grid(True)


            self.update_clutter_map(beam_idx, np.abs(doppler_response))
            beam_targets = self.target_parameter_estimation(doppler_responses, detection_map, self.beam_dirs_deg[beam_idx])
            targets.extend(beam_targets)

        if self.segment_idx <= CLUTTER_WARMUP_SEGMENTS:
            print(f'[SmartCal] clutter warm-up {self.segment_idx}/{CLUTTER_WARMUP_SEGMENTS}')
            self._put_latest_targets([])
            return

        if PLOT_DEBUG_DATA:
            fig.savefig( f'log/map/detection_map_{self.segment_idx}.png', dpi=300)
            fig2.savefig( f'log/doppler/detection_map_{self.segment_idx}.png', dpi=300)
            plt.close('all')

        raw_target_count = len(targets)

        # 1) keep only valid moving detections
        filtered_targets = self.filter_targets(targets)

        # If the frame does not contain enough motion evidence, treat it as no movement.
        if len(filtered_targets) < MIN_FILTERED_TARGETS_PER_FRAME:
            self.confirm_counter = 0
            self.prev_grouped_targets = []
            printf(f'raw targets: {raw_target_count}, filtered: {len(filtered_targets)}, grouped: 0, decision=NO_MOTION')
            self._put_latest_targets([])
            return

        # 2) group nearby detections into person-level targets
        grouped_targets = self.group_targets(filtered_targets)

        # Required confirmation in consecutive completed radar frames before displaying.
        if grouped_targets:
            self.confirm_counter += 1
        else:
            self.confirm_counter = 0

        if self.confirm_counter < MIN_GROUP_CONFIRM_FRAMES:
            print(f'raw targets: {raw_target_count}, filtered: {len(filtered_targets)}, grouped: {len(filtered_targets)}, decision=WAIT_CONFIRM')
            self._put_latest_targets([])
            return

        # 3) smooth grouped targets across frames
        grouped_targets = self.smooth_grouped_targets(grouped_targets)

        print(f'raw targets: {raw_target_count}, filtered: {len(filtered_targets)}, grouped: {len(grouped_targets)}, decision=MOTION')

        # send only grouped targets to GUI
        self._put_latest_targets(grouped_targets)


    #copy of run_postprocessing() for human detection
    #TODO:
    def run_postprocessing_human_detection(self, segment):
        """
        run all postprocessing to detect targets from segment of CIRs
        """

        doppler_responses = np.zeros_like(segment)
        for ant_idx in range(segment.shape[0]):
            doppler_responses[ant_idx,:,:] = self.calculate_doppler_response(segment[ant_idx,:,:][None, :, :]) * np.exp(1j * AoA_CAL_VALUE[ant_idx])
        targets = []
        if PLOT_DEBUG_DATA:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
            fig2, axes2 = plt.subplots(nrows=3, ncols=3, figsize=(15, 9))

        for beam_idx in range(self.num_beams):
            doppler_response = np.zeros((doppler_responses.shape[1],doppler_responses.shape[2]), dtype=np.complex64)
            doppler_response_diff = np.zeros_like(doppler_response)
            for ant_idx in range(segment.shape[0]):
                doppler_response += self.beam_weights[beam_idx, ant_idx] * doppler_responses[ant_idx, :, :]
                doppler_response_diff += self.diff_beam_weights[beam_idx, ant_idx] * doppler_responses[ant_idx, :, :]

            sigma_grater_delta_indicator = np.abs(doppler_response) - np.abs(doppler_response_diff) >0 # if sigma is greater than delta is our area of interest

            doppler_response_wo_clutter_dB = 10 * np.log10(np.maximum(np.abs(doppler_response.copy()), EPS_MAG)) - 10 * np.log10(np.maximum(np.abs(self.clutter_map[beam_idx, :, :]), EPS_MAG))
            #doppler_response_wo_clutter_dB = 10 * np.log10(np.abs(np.abs(doppler_response.copy()) - np.abs(self.clutter_map[beam_idx, :, :])))

            # detection_map = CFAR(10*np.log10(np.abs(doppler_response_wo_clutter_mag)), self.CFAR_threshold, np.ones_like(sigma_grater_delta_indicator))
            # if PLOT_DEBUG_DATA:
            #     axes[0, beam_idx].imshow(detection_map)
            global segment_idx
            segment_idx = self.segment_idx
            detection_map = CFAR(doppler_response_wo_clutter_dB, self.CFAR_threshold, np.ones_like(sigma_grater_delta_indicator))
            #detection_map = CFAR(doppler_response_wo_clutter_mag, self.CFAR_threshold, sigma_grater_delta_indicator)
            if PLOT_DEBUG_DATA:
                axes[1, beam_idx].imshow(detection_map)

            if PLOT_DEBUG_DATA:
                axes2[0, beam_idx].plot((np.transpose(doppler_response_wo_clutter_dB)))
                axes2[0, beam_idx].grid(True)

                axes2[1, beam_idx].plot((np.transpose(10 * np.log10(np.maximum(np.abs(doppler_response.copy()), EPS_MAG)))))
                axes2[1, beam_idx].grid(True)

                axes2[2, beam_idx].plot((np.transpose(10 * np.log10(np.maximum(np.abs(self.clutter_map[beam_idx, :, :]), EPS_MAG)))))
                axes2[2, beam_idx].grid(True)


                #axes2[1, beam_idx].plot(10*np.log10(np.abs(np.transpose(doppler_response.copy()))))

            if self.segment_idx <= CLUTTER_WARMUP_SEGMENTS:
                self.update_clutter_map(beam_idx, np.abs(doppler_response))
                continue
            self.update_clutter_map(beam_idx, np.abs(doppler_response))
            beam_targets = self.target_parameter_estimation(doppler_responses, detection_map, self.beam_dirs_deg[beam_idx])
            targets.extend(beam_targets)

        if self.segment_idx <= CLUTTER_WARMUP_SEGMENTS:
            print(f'[SmartCal] clutter warm-up {self.segment_idx}/{CLUTTER_WARMUP_SEGMENTS}')
            self._put_latest_targets([])
            return

        if PLOT_DEBUG_DATA:
            fig.savefig( f'log/map/detection_map_{self.segment_idx}.png', dpi=300)
            fig2.savefig( f'log/doppler/detection_map_{self.segment_idx}.png', dpi=300)
            plt.close('all')
        
        print(f'num of targets: {len(targets)}')
        self._put_latest_targets(targets)

        


    def worker(self):
        """
        check if data are in buffer to start postprocessing
        """

        while self.do_postprocessing:
            try:
                data = self.segment_queue.get(timeout=1)
            except Empty:
                continue
            #data = self.segment_queue.get() # waits for buffer
            self.run_postprocessing(data)
            self.segment_queue.task_done()

    def finish_postprocessing(self):
        """
        stop loop of waiting for data for postprocessing
        """
        self.do_postprocessing = False


if __name__ == '__main__':

    clear_folder('log/detection')
    clear_folder('log/map')
    clear_folder('log/doppler', True)
    if HUMAN_DETECTION:
        log_file_empty = r"C:\Users\a.raniszewsk\Documents\CODE\GIT\RFtestUCI2_0\RFtestsUCI2_0\log\RADARlogs_ooo_50.log"
    log_file_target = r"C:\Users\a.raniszewsk\Documents\CODE\CIG\RFtestUCI2_0\RFtestsUCI2_0\log\RADARlogs.log"
    
    if HUMAN_DETECTION:
        f = open(log_file_empty, 'r')
        lines_empty = f.readlines()
        f.close()
        idx_e = 0
    #read log for data testing
    f = open(log_file_target, 'r')
    lines_target = f.readlines()
    f.close()
    idx = 0

    while (lines_target[idx].find('cir_taps') < 0):
        idx += 1
    cir_taps = int(lines_target[idx].split('=')[-1])
    while (lines_target[idx].find('period') < 0):
        idx += 1
    PRI = int(lines_target[idx].split('=')[-1])
    while (lines_target[idx].find('ant') < 0):
        idx += 1
    ant = (lines_target[idx].split('=')[-1])
    while ant[0] == ' ':
        ant = ant[1:]

    target_queue = Queue()
    postprocessing_target = RadarPostprocessing(len(ant.split('-')[0]),
                                                    SEGMENT_LENGTH,
                                                    cir_taps,
                                                    PRI*1e-3,
                                                    1,
                                                    '9',
                                                    np.array(BEAM_DIRECTIONS_DEG),
                                                    THRESHOLD,
                                                    target_queue,
                                                    False)


    if HUMAN_DETECTION:
        while (lines_empty[idx_e].find('radar start...') < 0):
            idx_e += 1
    while (lines_target[idx].find('radar start...') < 0):
        idx += 1
    if HUMAN_DETECTION:
        for i in range(idx_e + 1, ide_e + 1 + SEGMENT_LENGTH * 2):
            data = lines_empty[i].split(',')
            seq = int(data[1].split(':')[-1])
            ant_bitmap = data[2].split(': ')[-1]
            cir_temp = data[-1].split(':')[-1].replace(" ", "").replace("\n", "")
            cir = ''
            for i in range(len(cir_temp) >> 3):
                cir += revert_hex(cir_temp[i*8:(i*8+8)])
            try:
                postprocessing_target.fill_data_by_hex(cir, ant_bitmap, seq-1)
            except:
                print(f'parsing exception, line idx: {i}')


    idx += 1
    while idx < len(lines_target):

        data = lines_target[idx].split(',')
        seq = int(data[1].split(':')[-1])
        ant_bitmap = data[2].splot(': ')[-1]
        cir_temp = data[-1].split(':')[-1].replace(" ","").replace("\n", "")
        cir = ''
        for i in range(len(cir_temp) >> 3):
            cir += revert_hex(cir_temp[i*8:(i*8+8)])
        try:
            postprocessing_target.fill_data_by_hex(cir, ant_bitmap, seq-1+SEGMENT_LENGTH*int(HUMAN_DETECTION))
        except:
            print(f'parsing exception, lide index: {idx}')


        idx += 1

    a = 5
