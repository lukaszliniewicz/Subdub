import os
import logging
from typing import List, Dict
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def align_audio_blocks(alignment_blocks: List[Dict], session_folder: str, delay_start: int = 1500, speed_up: int = 100) -> str:
    final_audio = AudioSegment.silent(duration=0)
    current_time = 0
    total_shift = 0

    def timedelta_to_ms(td):
        return td.total_seconds() * 1000

    def speed_up_audio_ffmpeg(audio_segment, factor, session_folder):
        import subprocess
        
        temp_input = os.path.join(session_folder, "temp_speedup_input.wav")
        temp_output = os.path.join(session_folder, "temp_speedup_output.wav")
        
        try:
            audio_segment.export(temp_input, format="wav")
            
            atempo_filters = []
            remaining_factor = factor
            
            while remaining_factor > 2.0:
                atempo_filters.append("atempo=2.0")
                remaining_factor /= 2.0
            
            while remaining_factor < 0.5:
                atempo_filters.append("atempo=0.5")
                remaining_factor /= 0.5
            
            if abs(remaining_factor - 1.0) > 0.01:
                atempo_filters.append(f"atempo={remaining_factor:.3f}")
            
            if not atempo_filters:
                return audio_segment
                
            filter_string = ",".join(atempo_filters)
            
            cmd = ['ffmpeg', '-i', temp_input, '-af', filter_string, '-y', temp_output]
            subprocess.run(cmd, check=True, capture_output=True)
            return AudioSegment.from_wav(temp_output)
            
        finally:
            for temp_file in [temp_input, temp_output]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    sentence_wavs_folder = os.path.join(session_folder, "Sentence_wavs")

    for i, block in enumerate(alignment_blocks):
        block_start_ms = timedelta_to_ms(block["start"])
        
        if i < len(alignment_blocks) - 1:
            next_block_start_ms = timedelta_to_ms(alignment_blocks[i + 1]["start"])
            block_duration = next_block_start_ms - block_start_ms
        else:
            block_duration = timedelta_to_ms(block["end"] - block["start"])

        adjusted_block_start = block_start_ms + total_shift
        
        if adjusted_block_start > current_time:
            silence_duration = adjusted_block_start - current_time
            final_audio += AudioSegment.silent(duration=silence_duration)
            current_time = adjusted_block_start

        block_audio = AudioSegment.silent(duration=0)
        for audio_file in block["audio_files"]:
            wav_path = os.path.join(sentence_wavs_folder, audio_file)
            if os.path.exists(wav_path):
                wav_audio = AudioSegment.from_wav(wav_path)
                if len(block_audio) > 0:
                    block_audio += AudioSegment.silent(duration=100)
                block_audio += wav_audio
            else:
                logger.error(f"Audio file not found: {wav_path}")

        original_audio_duration = len(block_audio)
        processed_audio = block_audio
        audio_delay = 0

        if original_audio_duration < block_duration and total_shift <= 0:
            available_time = block_duration - original_audio_duration
            max_delay = min(delay_start, int(available_time * 0.7))
            audio_delay = max_delay
            logger.info(f"Block {i+1}: Applying delay of {audio_delay}ms "
                       f"(audio: {original_audio_duration}ms, subtitle: {block_duration}ms)")

        should_speed_up = False
        actual_speedup_factor = 1.0
        speed_up_reason = ""

        if total_shift > 0 and speed_up > 100:
            should_speed_up = True
            speed_up_reason = f"eliminating accumulated shift of {total_shift}ms"
            
            speedup_needed_for_shift = (original_audio_duration + total_shift) / original_audio_duration
            max_allowed_speedup = speed_up / 100.0
            
            actual_speedup_factor = min(speedup_needed_for_shift, max_allowed_speedup)
            
            logger.info(f"Block {i+1}: Shift elimination - need {speedup_needed_for_shift:.2f}x, "
                       f"max allowed {max_allowed_speedup:.2f}x, using {actual_speedup_factor:.2f}x")
            
        elif total_shift <= 0 and original_audio_duration > block_duration and speed_up > 100:
            should_speed_up = True
            speed_up_reason = "fitting current audio to subtitle"
            
            speedup_needed_for_subtitle = original_audio_duration / block_duration
            max_allowed_speedup = speed_up / 100.0
            
            actual_speedup_factor = min(speedup_needed_for_subtitle, max_allowed_speedup)
            
            logger.info(f"Block {i+1}: Subtitle fitting - need {speedup_needed_for_subtitle:.2f}x, "
                       f"max allowed {max_allowed_speedup:.2f}x, using {actual_speedup_factor:.2f}x")

        if should_speed_up and actual_speedup_factor > 1.01:
            processed_audio = speed_up_audio_ffmpeg(block_audio, actual_speedup_factor, session_folder)
            new_duration = len(processed_audio)
            logger.info(f"Block {i+1}: Applied {actual_speedup_factor:.2f}x speedup for {speed_up_reason} "
                       f"({original_audio_duration}ms -> {new_duration}ms)")
        elif total_shift > 0 and speed_up <= 100:
            logger.info(f"Block {i+1}: Would speed up for shift elimination, but speed_up={speed_up} (disabled)")
        elif should_speed_up:
            logger.info(f"Block {i+1}: Speedup factor {actual_speedup_factor:.2f}x too small, skipping")

        if audio_delay > 0:
            final_audio += AudioSegment.silent(duration=audio_delay)
            current_time += audio_delay

        final_audio += processed_audio
        current_time += len(processed_audio)

        actual_audio_duration = len(processed_audio) + audio_delay
        
        if actual_audio_duration > block_duration:
            additional_shift = actual_audio_duration - block_duration
            total_shift += additional_shift
            logger.info(f"Block {i+1}: Added {additional_shift}ms to shift, total_shift now: {total_shift}ms")
        else:
            silence_needed = block_duration - actual_audio_duration
            if silence_needed >= total_shift:
                silence_to_add = silence_needed - total_shift
                final_audio += AudioSegment.silent(duration=silence_to_add)
                current_time += silence_to_add
                logger.info(f"Block {i+1}: Eliminated all shift ({total_shift}ms) plus {silence_to_add}ms silence")
                total_shift = 0
            else:
                total_shift -= silence_needed
                logger.info(f"Block {i+1}: Reduced shift by {silence_needed}ms, total_shift now: {total_shift}ms")

    output_path = os.path.join(session_folder, "aligned_audio.wav")
    final_audio.export(output_path, format="wav")
    logger.info(f"Alignment completed. Final total_shift: {total_shift}ms")
    return output_path
