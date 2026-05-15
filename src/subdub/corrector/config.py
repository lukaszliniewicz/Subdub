from dataclasses import dataclass


@dataclass
class CorrectorConfig:
    # Gap detection
    min_gap_for_check: float = 0.1

    # Energy analysis
    forward_window: float = 0.1
    backward_step: float = 0.025
    max_backward_steps: int = 120

    # Energy thresholds
    high_energy_threshold: float = 0.5
    low_energy_threshold: float = 0.15
    spike_threshold: float = 1.5

    # Boundary detection
    contaminated_windows_skip: int = 6
    lookback_window: int = 2
    boundary_buffer_steps: int = 4

    # Segment overlap correction
    overlap_buffer: float = 0.02

    # Audio processing
    sample_rate: int = 16000

    # Visualization
    window_padding: float = 0.5

    # Playback
    playback_padding: float = 0.5

    # Logging
    log_level: str = "INFO"
