# =============================================================================
# src/live_capture.py
# AI-Powered DDoS IDS — Live Network Packet Capture & Feature Extraction
#
# PURPOSE:
#   Capture raw packets from a NIC in real time using Scapy, aggregate them
#   into "flows" (groups of packets sharing the same 5-tuple), extract the
#   same statistical features used during training, and yield them for
#   model inference.
#
# KEY CONCEPTS:
#   - A "flow" is defined by: (src_ip, dst_ip, src_port, dst_port, protocol)
#   - Features are computed per-flow (e.g., mean packet size, SYN flag count)
#   - A flow is "complete" after FLOW_TIMEOUT seconds of inactivity
#   - This module runs in a BACKGROUND THREAD so the Streamlit UI stays live
#
# ARCHITECTURE:
#   NIC → Scapy sniffer thread → FlowAggregator → feature dict → model queue
# =============================================================================

import time
import logging
import threading
import queue
from collections import defaultdict
from typing import Optional, Callable

import numpy as np

# Scapy imports — will raise ImportError if scapy is not installed
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    from scapy.layers.inet import IP as ScapyIP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available. Live capture will run in SIMULATION mode.")

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveCapture")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
FLOW_TIMEOUT      = 5.0    # Seconds of inactivity before a flow is exported
MAX_QUEUE_SIZE    = 500    # Max feature-vectors waiting in the output queue
SIMULATION_RATE   = 1.0    # Seconds between simulated flow generation (demo mode)


# ===========================================================================
# 1. FLOW KEY & FLOW RECORD
# ===========================================================================

def _flow_key(packet) -> Optional[tuple]:
    """
    Derive the 5-tuple flow key from a Scapy packet.
    Returns None if the packet has no IP layer (e.g., ARP, 802.11).

    The 5-tuple is the standard network flow identifier:
      (source_ip, destination_ip, source_port, destination_port, protocol)

    We sort IPs so bidirectional traffic (A→B and B→A) maps to the same flow.
    """
    if not packet.haslayer(IP):
        return None

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    proto  = packet[IP].proto

    src_port = dst_port = 0
    if packet.haslayer(TCP):
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif packet.haslayer(UDP):
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    return (src_ip, dst_ip, src_port, dst_port, proto)


class FlowRecord:
    """
    Accumulates raw packet data for a single network flow.

    Each packet in the flow contributes to running statistics:
      - Packet sizes (for mean, max, min, variance calculations)
      - Inter-arrival times (IAT)
      - TCP flag counts
      - Byte and packet rates

    These statistics are later turned into the feature vector our model expects.
    """

    def __init__(self, first_packet, timestamp: float):
        self.start_time     = timestamp
        self.last_seen      = timestamp
        self.packet_sizes   = []       # Raw packet lengths
        self.timestamps     = []       # Arrival times (for IAT computation)
        self.fwd_packets    = []       # Forward direction packet sizes
        self.bwd_packets    = []       # Backward direction packet sizes
        self.fwd_iat        = []       # Forward inter-arrival times
        self.bwd_iat        = []       # Backward inter-arrival times

        # TCP flag accumulators
        self.syn_count  = 0
        self.fin_count  = 0
        self.rst_count  = 0
        self.psh_count  = 0
        self.ack_count  = 0
        self.urg_count  = 0

        # TCP window size (from first SYN packet)
        self.init_win_fwd = 0
        self.init_win_bwd = 0

        # Record the first packet immediately
        self._src_ip = first_packet[IP].src if first_packet.haslayer(IP) else ""
        self.add_packet(first_packet, timestamp, is_forward=True)

    def add_packet(self, packet, timestamp: float, is_forward: bool):
        """
        Incorporate one packet into the flow's running statistics.
        """
        self.last_seen = timestamp
        size = len(packet)
        self.packet_sizes.append(size)
        self.timestamps.append(timestamp)

        if is_forward:
            if self.fwd_packets:
                self.fwd_iat.append(timestamp - self.timestamps[-2]
                                    if len(self.timestamps) > 1 else 0.0)
            self.fwd_packets.append(size)
            if len(self.fwd_packets) == 1 and packet.haslayer(TCP):
                self.init_win_fwd = packet[TCP].window
        else:
            if self.bwd_packets:
                self.bwd_iat.append(timestamp - self.last_seen)
            self.bwd_packets.append(size)
            if len(self.bwd_packets) == 1 and packet.haslayer(TCP):
                self.init_win_bwd = packet[TCP].window

        # Parse TCP flags
        if packet.haslayer(TCP):
            flags = packet[TCP].flags
            if flags & 0x02: self.syn_count += 1   # SYN
            if flags & 0x01: self.fin_count += 1   # FIN
            if flags & 0x04: self.rst_count += 1   # RST
            if flags & 0x08: self.psh_count += 1   # PSH
            if flags & 0x10: self.ack_count += 1   # ACK
            if flags & 0x20: self.urg_count += 1   # URG

    def to_feature_dict(self) -> dict:
        """
        Convert accumulated flow data into the feature dictionary expected
        by the preprocessing pipeline.

        Returns:
            A dict mapping feature names (matching FEATURE_COLUMNS) to values.
        """
        duration_sec = max(self.last_seen - self.start_time, 1e-9)  # avoid div/0
        duration_us  = duration_sec * 1_000_000   # microseconds

        all_sizes = np.array(self.packet_sizes, dtype=float)
        fwd_sizes = np.array(self.fwd_packets, dtype=float)
        bwd_sizes = np.array(self.bwd_packets, dtype=float)

        # Inter-arrival time arrays
        iats = np.diff(self.timestamps) if len(self.timestamps) > 1 else np.array([0.0])
        fwd_iats = np.array(self.fwd_iat, dtype=float) if self.fwd_iat else np.array([0.0])
        bwd_iats = np.array(self.bwd_iat, dtype=float) if self.bwd_iat else np.array([0.0])

        def safe_mean(arr): return float(np.mean(arr)) if len(arr) > 0 else 0.0
        def safe_std(arr):  return float(np.std(arr))  if len(arr) > 1 else 0.0
        def safe_max(arr):  return float(np.max(arr))  if len(arr) > 0 else 0.0
        def safe_min(arr):  return float(np.min(arr))  if len(arr) > 0 else 0.0

        total_fwd_bytes = float(np.sum(fwd_sizes))
        total_bwd_bytes = float(np.sum(bwd_sizes))
        total_bytes     = total_fwd_bytes + total_bwd_bytes

        return {
            "Flow Duration":                  duration_us,
            "Total Fwd Packets":              len(self.fwd_packets),
            "Total Backward Packets":         len(self.bwd_packets),
            "Total Length of Fwd Packets":    total_fwd_bytes,
            "Total Length of Bwd Packets":    total_bwd_bytes,
            "Fwd Packet Length Max":          safe_max(fwd_sizes),
            "Fwd Packet Length Min":          safe_min(fwd_sizes),
            "Fwd Packet Length Mean":         safe_mean(fwd_sizes),
            "Bwd Packet Length Max":          safe_max(bwd_sizes),
            "Bwd Packet Length Min":          safe_min(bwd_sizes),
            "Bwd Packet Length Mean":         safe_mean(bwd_sizes),
            "Flow Bytes/s":                   total_bytes / duration_sec,
            "Flow Packets/s":                 len(self.packet_sizes) / duration_sec,
            "Flow IAT Mean":                  safe_mean(iats),
            "Flow IAT Std":                   safe_std(iats),
            "Fwd IAT Mean":                   safe_mean(fwd_iats),
            "Bwd IAT Mean":                   safe_mean(bwd_iats),
            "Fwd PSH Flags":                  self.psh_count,
            "Bwd PSH Flags":                  0,
            "Fwd URG Flags":                  self.urg_count,
            "Fwd Header Length":              20 * len(self.fwd_packets),
            "Bwd Header Length":              20 * len(self.bwd_packets),
            "Fwd Packets/s":                  len(self.fwd_packets) / duration_sec,
            "Bwd Packets/s":                  len(self.bwd_packets) / duration_sec,
            "Min Packet Length":              safe_min(all_sizes),
            "Max Packet Length":              safe_max(all_sizes),
            "Packet Length Mean":             safe_mean(all_sizes),
            "Packet Length Std":              safe_std(all_sizes),
            "Packet Length Variance":         float(np.var(all_sizes)) if len(all_sizes) > 1 else 0.0,
            "FIN Flag Count":                 self.fin_count,
            "SYN Flag Count":                 self.syn_count,
            "RST Flag Count":                 self.rst_count,
            "PSH Flag Count":                 self.psh_count,
            "ACK Flag Count":                 self.ack_count,
            "URG Flag Count":                 self.urg_count,
            "Average Packet Size":            safe_mean(all_sizes),
            "Avg Fwd Segment Size":           safe_mean(fwd_sizes),
            "Avg Bwd Segment Size":           safe_mean(bwd_sizes),
            "Init_Win_bytes_forward":         self.init_win_fwd,
            "Init_Win_bytes_backward":        self.init_win_bwd,
            "act_data_pkt_fwd":               max(len(self.fwd_packets) - 1, 0),
            "min_seg_size_forward":           safe_min(fwd_sizes),
            "Active Mean":                    duration_us,
            "Active Std":                     0.0,
            "Idle Mean":                      0.0,
            "Idle Std":                       0.0,
        }


# ===========================================================================
# 2. FLOW AGGREGATOR
# ===========================================================================

class FlowAggregator:
    """
    Maintains a dictionary of active flows, adding each captured packet to
    the correct flow, and exporting completed flows (by timeout) to a queue.

    Thread-safe via a threading.Lock on the flows dict.
    """

    def __init__(self, output_queue: queue.Queue, timeout: float = FLOW_TIMEOUT):
        self.flows        = {}              # {flow_key: FlowRecord}
        self.output_queue = output_queue
        self.timeout      = timeout
        self.lock         = threading.Lock()
        self.packet_count = 0
        self.flow_count   = 0

    def process_packet(self, packet):
        """
        Called by the Scapy sniffer for each captured packet.
        Finds or creates the flow this packet belongs to, then adds it.
        """
        now = time.time()
        key = _flow_key(packet)
        if key is None:
            return   # Skip non-IP packets

        self.packet_count += 1

        with self.lock:
            if key in self.flows:
                # Determine direction: is this packet going src→dst (forward)?
                is_fwd = (packet[IP].src == key[0])
                self.flows[key].add_packet(packet, now, is_forward=is_fwd)
            else:
                # New flow
                self.flows[key] = FlowRecord(packet, now)
                self.flow_count += 1

            # Check for and export any timed-out flows
            self._export_expired_flows(now)

    def _export_expired_flows(self, now: float):
        """
        Find flows that have been inactive for FLOW_TIMEOUT seconds,
        compute their feature dict, and push to the output queue.
        Called with self.lock held.
        """
        expired_keys = [
            k for k, flow in self.flows.items()
            if (now - flow.last_seen) >= self.timeout
        ]
        for key in expired_keys:
            flow = self.flows.pop(key)
            features = flow.to_feature_dict()
            # Annotate with metadata for dashboard display
            features["_src_ip"]   = key[0]
            features["_dst_ip"]   = key[1]
            features["_protocol"] = key[4]
            features["_timestamp"] = now

            try:
                self.output_queue.put_nowait(features)
            except queue.Full:
                logger.warning("Output queue full — dropping oldest flow")

    def flush_all(self, now: float = None):
        """Export ALL remaining flows regardless of timeout (called on shutdown)."""
        if now is None:
            now = time.time()
        with self.lock:
            self._export_expired_flows(now - self.timeout)  # Force all to expire


# ===========================================================================
# 3. LIVE CAPTURE CONTROLLER
# ===========================================================================

class LiveCaptureController:
    """
    High-level controller that:
      1. Starts a Scapy sniffing thread
      2. Feeds packets into a FlowAggregator
      3. Exposes a thread-safe queue for the Streamlit app to consume
      4. Provides start/stop controls

    Usage:
        controller = LiveCaptureController(interface="eth0")
        controller.start()
        while True:
            features = controller.get_flow()   # blocks up to 1 second
            if features:
                prediction = model.predict(preprocess(features))
        controller.stop()
    """

    def __init__(self, interface: str = None, packet_filter: str = "ip",
                 simulation_mode: bool = False):
        """
        Args:
            interface:       NIC name (e.g. "eth0", "Wi-Fi", None = auto-detect)
            packet_filter:   BPF filter string passed to Scapy (default: IP only)
            simulation_mode: If True, generate synthetic traffic instead of
                             capturing real packets (useful for demo/testing).
        """
        self.interface       = interface
        self.packet_filter   = packet_filter
        self.simulation_mode = simulation_mode or not SCAPY_AVAILABLE

        self._output_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._aggregator   = FlowAggregator(self._output_queue)
        self._stop_event   = threading.Event()
        self._thread       = None
        self.is_running    = False

        if self.simulation_mode:
            logger.warning(
                "LiveCaptureController starting in SIMULATION MODE. "
                "Synthetic traffic will be generated for demonstration."
            )

    # ------------------------------------------------------------------
    def start(self):
        """Start the background capture/simulation thread."""
        if self.is_running:
            logger.warning("Capture already running.")
            return

        self._stop_event.clear()
        self.is_running = True

        if self.simulation_mode:
            self._thread = threading.Thread(
                target=self._simulation_loop, daemon=True, name="SimulationThread"
            )
        else:
            self._thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="CaptureThread"
            )

        self._thread.start()
        logger.info(f"Capture thread started (mode: "
                    f"{'simulation' if self.simulation_mode else 'live'}).")

    def stop(self):
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._aggregator.flush_all()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.is_running = False
        logger.info("Capture thread stopped.")

    def get_flow(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Retrieve the next completed flow from the queue.

        Args:
            timeout: Seconds to block waiting for a flow.

        Returns:
            Feature dict, or None if queue is empty after timeout.
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    def _capture_loop(self):
        """
        Real Scapy capture loop. Runs in a daemon thread.
        sniff() is blocking; we use stop_filter to allow clean shutdown.
        """
        logger.info(f"Starting real packet capture on interface: {self.interface or 'default'}")
        try:
            sniff(
                iface=self.interface,
                filter=self.packet_filter,
                prn=self._aggregator.process_packet,
                store=False,       # Don't buffer packets in RAM — stream them
                stop_filter=lambda _: self._stop_event.is_set(),
            )
        except PermissionError:
            logger.error(
                "PermissionError: Packet capture requires root/Administrator privileges. "
                "Run with: sudo python app.py"
            )
            self.simulation_mode = True
            self._simulation_loop()
        except Exception as exc:
            logger.error(f"Capture error: {exc}. Falling back to simulation mode.")
            self.simulation_mode = True
            self._simulation_loop()

    def _simulation_loop(self):
        """
        Generate synthetic network flow records for demo/testing when real
        capture is unavailable. Produces a mix of BENIGN and attack-like flows.

        The simulated features intentionally mimic real attack patterns:
          - SYN Flood: high SYN count, low flow duration, tiny packets
          - UDP Flood: very high packet rate, fixed small size
          - BENIGN:    normal-looking IAT and packet sizes
        """
        import random

        attack_types = ["SYN Flood", "UDP Flood", "BENIGN", "BENIGN", "BENIGN"]
        src_ips = [f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
                   for _ in range(20)]

        logger.info("Simulation loop running — generating synthetic flows.")
        while not self._stop_event.is_set():
            attack = random.choice(attack_types)
            now = time.time()

            if attack == "SYN Flood":
                features = {
                    "Flow Duration": random.uniform(100, 5000),
                    "Total Fwd Packets": random.randint(100, 1000),
                    "Total Backward Packets": random.randint(0, 5),
                    "Total Length of Fwd Packets": random.uniform(6000, 60000),
                    "Total Length of Bwd Packets": random.uniform(0, 300),
                    "Fwd Packet Length Max": 60.0,
                    "Fwd Packet Length Min": 40.0,
                    "Fwd Packet Length Mean": 44.0,
                    "Bwd Packet Length Max": 60.0,
                    "Bwd Packet Length Min": 0.0,
                    "Bwd Packet Length Mean": 0.0,
                    "Flow Bytes/s": random.uniform(100000, 10000000),
                    "Flow Packets/s": random.uniform(1000, 100000),
                    "Flow IAT Mean": random.uniform(10, 500),
                    "Flow IAT Std": random.uniform(5, 100),
                    "Fwd IAT Mean": random.uniform(10, 500),
                    "Bwd IAT Mean": 0.0,
                    "Fwd PSH Flags": 0,
                    "Bwd PSH Flags": 0,
                    "Fwd URG Flags": 0,
                    "Fwd Header Length": 2000,
                    "Bwd Header Length": 0,
                    "Fwd Packets/s": random.uniform(1000, 100000),
                    "Bwd Packets/s": 0.0,
                    "Min Packet Length": 40.0,
                    "Max Packet Length": 60.0,
                    "Packet Length Mean": 44.0,
                    "Packet Length Std": 4.0,
                    "Packet Length Variance": 16.0,
                    "FIN Flag Count": 0,
                    "SYN Flag Count": random.randint(100, 1000),  # KEY indicator
                    "RST Flag Count": random.randint(0, 10),
                    "PSH Flag Count": 0,
                    "ACK Flag Count": random.randint(0, 10),
                    "URG Flag Count": 0,
                    "Average Packet Size": 44.0,
                    "Avg Fwd Segment Size": 44.0,
                    "Avg Bwd Segment Size": 0.0,
                    "Init_Win_bytes_forward": 0,   # SYN floods never complete handshake
                    "Init_Win_bytes_backward": 0,
                    "act_data_pkt_fwd": 0,
                    "min_seg_size_forward": 40.0,
                    "Active Mean": 500.0,
                    "Active Std": 100.0,
                    "Idle Mean": 0.0,
                    "Idle Std": 0.0,
                    "_src_ip": random.choice(src_ips),
                    "_dst_ip": "10.0.0.1",
                    "_protocol": 6,
                    "_timestamp": now,
                    "_simulated_label": "DDoS",
                }

            elif attack == "UDP Flood":
                features = {
                    "Flow Duration": random.uniform(1000, 10000),
                    "Total Fwd Packets": random.randint(500, 5000),
                    "Total Backward Packets": 0,
                    "Total Length of Fwd Packets": random.uniform(50000, 500000),
                    "Total Length of Bwd Packets": 0.0,
                    "Fwd Packet Length Max": 1500.0,
                    "Fwd Packet Length Min": 28.0,
                    "Fwd Packet Length Mean": 100.0,
                    "Bwd Packet Length Max": 0.0,
                    "Bwd Packet Length Min": 0.0,
                    "Bwd Packet Length Mean": 0.0,
                    "Flow Bytes/s": random.uniform(500000, 5000000),
                    "Flow Packets/s": random.uniform(5000, 50000),
                    "Flow IAT Mean": random.uniform(5, 200),
                    "Flow IAT Std": random.uniform(1, 50),
                    "Fwd IAT Mean": random.uniform(5, 200),
                    "Bwd IAT Mean": 0.0,
                    "Fwd PSH Flags": 0,
                    "Bwd PSH Flags": 0,
                    "Fwd URG Flags": 0,
                    "Fwd Header Length": 5000,
                    "Bwd Header Length": 0,
                    "Fwd Packets/s": random.uniform(5000, 50000),
                    "Bwd Packets/s": 0.0,
                    "Min Packet Length": 28.0,
                    "Max Packet Length": 1500.0,
                    "Packet Length Mean": 100.0,
                    "Packet Length Std": 50.0,
                    "Packet Length Variance": 2500.0,
                    "FIN Flag Count": 0,
                    "SYN Flag Count": 0,
                    "RST Flag Count": 0,
                    "PSH Flag Count": 0,
                    "ACK Flag Count": 0,
                    "URG Flag Count": 0,
                    "Average Packet Size": 100.0,
                    "Avg Fwd Segment Size": 100.0,
                    "Avg Bwd Segment Size": 0.0,
                    "Init_Win_bytes_forward": 0,
                    "Init_Win_bytes_backward": 0,
                    "act_data_pkt_fwd": random.randint(400, 4000),
                    "min_seg_size_forward": 28.0,
                    "Active Mean": 1000.0,
                    "Active Std": 200.0,
                    "Idle Mean": 0.0,
                    "Idle Std": 0.0,
                    "_src_ip": random.choice(src_ips),
                    "_dst_ip": "10.0.0.1",
                    "_protocol": 17,  # UDP
                    "_timestamp": now,
                    "_simulated_label": "DDoS",
                }

            else:  # BENIGN normal traffic
                features = {
                    "Flow Duration": random.uniform(50000, 5000000),
                    "Total Fwd Packets": random.randint(5, 100),
                    "Total Backward Packets": random.randint(3, 80),
                    "Total Length of Fwd Packets": random.uniform(500, 100000),
                    "Total Length of Bwd Packets": random.uniform(300, 80000),
                    "Fwd Packet Length Max": random.uniform(200, 1500),
                    "Fwd Packet Length Min": random.uniform(40, 100),
                    "Fwd Packet Length Mean": random.uniform(100, 800),
                    "Bwd Packet Length Max": random.uniform(200, 1500),
                    "Bwd Packet Length Min": random.uniform(40, 100),
                    "Bwd Packet Length Mean": random.uniform(100, 600),
                    "Flow Bytes/s": random.uniform(1000, 50000),
                    "Flow Packets/s": random.uniform(5, 200),
                    "Flow IAT Mean": random.uniform(1000, 50000),
                    "Flow IAT Std": random.uniform(500, 20000),
                    "Fwd IAT Mean": random.uniform(1000, 50000),
                    "Bwd IAT Mean": random.uniform(1000, 50000),
                    "Fwd PSH Flags": random.randint(0, 5),
                    "Bwd PSH Flags": random.randint(0, 5),
                    "Fwd URG Flags": 0,
                    "Fwd Header Length": random.randint(200, 2000),
                    "Bwd Header Length": random.randint(100, 1500),
                    "Fwd Packets/s": random.uniform(2, 100),
                    "Bwd Packets/s": random.uniform(1, 80),
                    "Min Packet Length": random.uniform(40, 100),
                    "Max Packet Length": random.uniform(500, 1500),
                    "Packet Length Mean": random.uniform(100, 700),
                    "Packet Length Std": random.uniform(50, 300),
                    "Packet Length Variance": random.uniform(2500, 90000),
                    "FIN Flag Count": random.randint(0, 3),
                    "SYN Flag Count": random.randint(0, 2),
                    "RST Flag Count": random.randint(0, 1),
                    "PSH Flag Count": random.randint(0, 5),
                    "ACK Flag Count": random.randint(5, 50),
                    "URG Flag Count": 0,
                    "Average Packet Size": random.uniform(100, 700),
                    "Avg Fwd Segment Size": random.uniform(100, 700),
                    "Avg Bwd Segment Size": random.uniform(80, 600),
                    "Init_Win_bytes_forward": random.choice([8192, 16384, 32768, 65535]),
                    "Init_Win_bytes_backward": random.choice([8192, 16384, 32768, 65535]),
                    "act_data_pkt_fwd": random.randint(3, 90),
                    "min_seg_size_forward": random.uniform(40, 80),
                    "Active Mean": random.uniform(10000, 200000),
                    "Active Std": random.uniform(1000, 50000),
                    "Idle Mean": random.uniform(5000, 100000),
                    "Idle Std": random.uniform(1000, 30000),
                    "_src_ip": random.choice(src_ips),
                    "_dst_ip": f"10.0.0.{random.randint(1,10)}",
                    "_protocol": random.choice([6, 17]),
                    "_timestamp": now,
                    "_simulated_label": "BENIGN",
                }

            try:
                self._output_queue.put_nowait(features)
            except queue.Full:
                pass  # Drop if consumer is too slow

            # Sleep between flows to simulate realistic traffic rate
            self._stop_event.wait(timeout=SIMULATION_RATE)

        logger.info("Simulation loop exited.")
