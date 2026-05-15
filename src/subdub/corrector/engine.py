import json
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import srt
from datetime import timedelta

from .config import CorrectorConfig

logger = logging.getLogger(__name__)

def load_audio(audio_path: str, sr: int) -> np.ndarray:
    """Load audio file and return as numpy array."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio

def calculate_energy(audio: np.ndarray, start_sample: int, end_sample: int) -> float:
    """Calculate RMS energy for audio segment."""
    if start_sample >= len(audio) or end_sample > len(audio) or start_sample >= end_sample:
        return 0.0
    segment = audio[start_sample:end_sample]
    return np.sqrt(np.mean(segment**2))

def time_to_samples(time_seconds: float, sr: int) -> int:
    """Convert time in seconds to sample index."""
    return int(time_seconds * sr)

def samples_to_time(samples: int, sr: int) -> float:
    """Convert sample index to time in seconds."""
    return samples / sr

def check_and_correct_overlaps(segments: List[Dict], config: CorrectorConfig) -> List[Dict]:
    """Check for overlapping segments and correct them."""
    corrections = []
    
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        if current['end'] > next_seg['start']:
            old_end = current['end']
            new_end = next_seg['start'] - config.overlap_buffer
            current['end'] = new_end
            
            correction = {
                'type': 'overlap',
                'segment_index': i,
                'old_end': old_end,
                'new_end': new_end,
                'reason': f"Segment {i} end ({old_end:.3f}s) was after segment {i+1} start ({next_seg['start']:.3f}s)"
            }
            corrections.append(correction)
            logger.info(f"Corrected overlap: {correction['reason']}")
    
    return corrections

def correct_segment_boundary(segment: Dict, next_segment: Dict, audio: np.ndarray, 
                           config: CorrectorConfig, segment_index: int) -> Optional[Dict]:
    """Analyze and correct segment boundary using local spike detection."""
    sr = config.sample_rate
    gap = next_segment['start'] - segment['end']
    
    if gap >= config.min_gap_for_check:
        return None
    
    forward_start = time_to_samples(segment['end'], sr)
    forward_end = time_to_samples(segment['end'] + config.forward_window, sr)
    forward_energy = calculate_energy(audio, forward_start, forward_end)
    
    if forward_energy == 0:
        logger.debug(f"Segment {segment_index}: Zero forward energy, skipping correction")
        return None
    
    energies = []
    times = []
    
    for step in range(config.max_backward_steps):
        step_end = segment['end'] - (step * config.backward_step)
        step_start = step_end - config.backward_step
        
        if step_start < segment['start']:
            break
        
        step_start_sample = time_to_samples(step_start, sr)
        step_end_sample = time_to_samples(step_end, sr)
        step_energy = calculate_energy(audio, step_start_sample, step_end_sample)
        
        energies.append(step_energy)
        times.append(step_start)
    
    skip_windows = config.contaminated_windows_skip
    lookback_window = config.lookback_window
    min_windows_needed = skip_windows + lookback_window + 1
    
    if len(energies) < min_windows_needed:
        logger.debug(f"Segment {segment_index}: Not enough windows for analysis")
        return None
    
    speech_onset_step = None
    
    for i in range(skip_windows + lookback_window, len(energies)):
        current_energy = energies[i]
        
        prev_start = max(skip_windows, i - lookback_window)
        prev_energies = energies[prev_start:i]
        
        if len(prev_energies) == 0:
            continue
            
        avg_prev_energy = np.mean(prev_energies)
        max_prev_energy = np.max(prev_energies)
        
        spike_ratio = current_energy / (avg_prev_energy + 1e-10)
        absolute_ratio = current_energy / forward_energy
        
        if (spike_ratio > config.spike_threshold and 
            absolute_ratio > config.low_energy_threshold and
            current_energy > max_prev_energy * 1.2):
            
            speech_onset_step = i
            break
    
    if speech_onset_step is None:
        logger.debug(f"Segment {segment_index}: No clear speech onset detected")
        return None
    
    buffered_boundary_step = max(0, speech_onset_step - config.boundary_buffer_steps)
    boundary_step = None
    
    for candidate_step in range(buffered_boundary_step, -1, -1):
        candidate_energy = energies[candidate_step]
        candidate_ratio = candidate_energy / forward_energy
        
        if candidate_ratio <= config.low_energy_threshold:
            boundary_step = candidate_step
            break
    
    if boundary_step is None:
        for candidate_step in range(buffered_boundary_step, -1, -1):
            candidate_energy = energies[candidate_step]
            candidate_ratio = candidate_energy / forward_energy
            
            if candidate_ratio < config.high_energy_threshold:
                boundary_step = candidate_step
                break
    
    if boundary_step is not None:
        new_end = times[boundary_step]
        
        min_change = config.backward_step
        if abs(new_end - segment['end']) > min_change and new_end < segment['end']:
            correction = {
                'type': 'energy_boundary',
                'segment_index': segment_index,
                'old_end': segment['end'],
                'new_end': new_end,
                'gap_size': gap,
                'forward_energy': forward_energy,
                'speech_onset_energy': energies[speech_onset_step],
                'boundary_energy': energies[boundary_step],
                'boundary_energy_ratio': energies[boundary_step] / forward_energy,
                'spike_ratio': energies[speech_onset_step] / np.mean(energies[max(skip_windows, speech_onset_step-lookback_window):speech_onset_step]),
                'speech_onset_step': speech_onset_step,
                'boundary_step': boundary_step,
                'buffered_step': buffered_boundary_step,
                'reason': f"Speech onset at step {speech_onset_step} ({times[speech_onset_step]:.3f}s), buffered to step {buffered_boundary_step}, cut at step {boundary_step} ({new_end:.3f}s) with energy ratio {energies[boundary_step] / forward_energy:.2f}"
            }
            
            segment['end'] = new_end
            logger.info(f"Corrected boundary: Segment {segment_index} end moved from {correction['old_end']:.3f}s to {correction['new_end']:.3f}s")
            return correction
    
    logger.debug(f"Segment {segment_index}: No suitable cut point found")
    return None

def correct_word_boundary(word: Dict, next_word: Dict, audio: np.ndarray, 
                           config: CorrectorConfig, word_index: int) -> Optional[Dict]:
    """Analyze and correct word boundary using local spike detection."""
    sr = config.sample_rate
    gap = next_word['start'] - word['end']
    
    if gap >= config.min_gap_for_check:
        return None
    
    forward_start = time_to_samples(word['end'], sr)
    forward_end = time_to_samples(word['end'] + config.forward_window, sr)
    forward_energy = calculate_energy(audio, forward_start, forward_end)
    
    if forward_energy == 0:
        logger.debug(f"Word {word_index}: Zero forward energy, skipping correction")
        return None
    
    energies = []
    times = []
    
    for step in range(config.max_backward_steps):
        step_end = word['end'] - (step * config.backward_step)
        step_start = step_end - config.backward_step
        
        if step_start < word['start']:
            break
        
        step_start_sample = time_to_samples(step_start, sr)
        step_end_sample = time_to_samples(step_end, sr)
        step_energy = calculate_energy(audio, step_start_sample, step_end_sample)
        
        energies.append(step_energy)
        times.append(step_start)
    
    skip_windows = config.contaminated_windows_skip
    lookback_window = config.lookback_window
    min_windows_needed = skip_windows + lookback_window + 1
    
    if len(energies) < min_windows_needed:
        logger.debug(f"Word {word_index}: Not enough windows for analysis")
        return None
    
    speech_onset_step = None
    
    for i in range(skip_windows + lookback_window, len(energies)):
        current_energy = energies[i]
        
        prev_start = max(skip_windows, i - lookback_window)
        prev_energies = energies[prev_start:i]
        
        if len(prev_energies) == 0:
            continue
            
        avg_prev_energy = np.mean(prev_energies)
        max_prev_energy = np.max(prev_energies)
        
        spike_ratio = current_energy / (avg_prev_energy + 1e-10)
        absolute_ratio = current_energy / forward_energy
        
        if (spike_ratio > config.spike_threshold and 
            absolute_ratio > config.low_energy_threshold and
            current_energy > max_prev_energy * 1.2):
            
            speech_onset_step = i
            break
    
    if speech_onset_step is None:
        logger.debug(f"Word {word_index}: No clear speech onset detected")
        return None
    
    buffered_boundary_step = max(0, speech_onset_step - config.boundary_buffer_steps)
    boundary_step = None
    
    for candidate_step in range(buffered_boundary_step, -1, -1):
        candidate_energy = energies[candidate_step]
        candidate_ratio = candidate_energy / forward_energy
        
        if candidate_ratio <= config.low_energy_threshold:
            boundary_step = candidate_step
            break
    
    if boundary_step is None:
        for candidate_step in range(buffered_boundary_step, -1, -1):
            candidate_energy = energies[candidate_step]
            candidate_ratio = candidate_energy / forward_energy
            
            if candidate_ratio < config.high_energy_threshold:
                boundary_step = candidate_step
                break
    
    if boundary_step is not None:
        new_end = times[boundary_step]
        
        min_change = config.backward_step
        if abs(new_end - word['end']) > min_change and new_end < word['end']:
            correction = {
                'type': 'word_boundary',
                'word_index': word_index,
                'old_end': word['end'],
                'new_end': new_end,
                'reason': f"Corrected word '{word.get('word', '')}'"
            }
            
            word['end'] = new_end
            logger.info(f"Corrected word boundary: Word {word_index} ('{word['word']}') end moved from {correction['old_end']:.3f}s to {correction['new_end']:.3f}s")
            return correction
    
    logger.debug(f"Word {word_index}: No suitable cut point found")
    return None

def process_word_boundaries_at_segment_ends(segments: List[Dict], audio: np.ndarray, config: CorrectorConfig) -> Tuple[List[Dict], List[Dict]]:
    """
    Applies energy-based boundary correction to the last few words of segments
    that have a small gap to the next segment.
    """
    import copy
    corrected_segments = copy.deepcopy(segments)
    corrections = []
    
    WORDS_TO_CHECK_AT_SEGMENT_END = 3

    for i in range(len(corrected_segments) - 1):
        current_seg = corrected_segments[i]
        next_seg = corrected_segments[i+1]

        gap = next_seg.get('start', 0) - current_seg.get('end', 0)
        if gap >= config.min_gap_for_check:
            continue

        if not current_seg.get('words') or not next_seg.get('words'):
            continue
            
        first_word_of_next_seg = next_seg['words'][0]
        words_to_check = current_seg['words'][-WORDS_TO_CHECK_AT_SEGMENT_END:]
        
        for word_to_correct in words_to_check:
            correction = correct_word_boundary(
                word_to_correct, 
                first_word_of_next_seg, 
                audio, 
                config, 
                -1
            )
            if correction:
                corrections.append(correction)

    all_words = []
    for seg in corrected_segments:
        if 'words' in seg and seg['words']:
            all_words.extend(seg['words'])
            
    for i in range(len(all_words) - 1):
        current_word = all_words[i]
        next_word = all_words[i+1]
        
        if current_word['end'] > next_word['start']:
            logger.debug(f"Fixing word overlap post-correction: Word {i} end {current_word['end']:.3f} > Word {i+1} start {next_word['start']:.3f}. Clipping.")
            current_word['end'] = next_word['start']

    logger.info(f"Total word-level boundary corrections at segment ends: {len(corrections)}")
    return all_words, corrections

def load_srt_file(srt_path: str) -> List[Dict]:
    """Load segments from an SRT file."""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    subtitles = srt.parse(content)
    segments = []
    
    for subtitle in subtitles:
        segment = {
            'start': subtitle.start.total_seconds(),
            'end': subtitle.end.total_seconds(),
            'text': subtitle.content
        }
        segments.append(segment)
    
    return segments

def process_whisperx_json(json_path: str, audio_path: str, config: CorrectorConfig) -> Tuple[List[Dict], List[Dict], np.ndarray, str]:
    """Process WhisperX JSON and correct segment boundaries."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data['segments']
    logger.info(f"Loaded {len(segments)} segments from {json_path}")
    
    audio = load_audio(audio_path, config.sample_rate)
    logger.info(f"Loaded audio from {audio_path}")
    
    corrections = []
    
    overlap_corrections = check_and_correct_overlaps(segments, config)
    corrections.extend(overlap_corrections)
    
    for i in range(len(segments) - 1):
        correction = correct_segment_boundary(
            segments[i], 
            segments[i + 1], 
            audio, 
            config, 
            i
        )
        if correction:
            corrections.append(correction)
    
    logger.info(f"Total corrections made: {len(corrections)}")
    return segments, corrections, audio, json_path

def segments_to_srt(segments: List[Dict]) -> str:
    """Convert segments to SRT format."""
    srt_segments = []
    
    for i, segment in enumerate(segments):
        subtitle = srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=segment['start']),
            end=timedelta(seconds=segment['end']),
            content=segment['text'].strip()
        )
        srt_segments.append(subtitle)
    
    return srt.compose(srt_segments)
