import os
import subprocess
import logging

from ..errors import ExternalToolError

logger = logging.getLogger(__name__)

def safe_decode(byte_string):
    try:
        return byte_string.decode('utf-8')
    except UnicodeDecodeError:
        return byte_string.decode('utf-8', errors='ignore')

def extract_audio(video_path: str, session_folder: str, video_name: str) -> str:
    audio_path = os.path.join(session_folder, f"{video_name}.wav")
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-af', 'aresample,loudnorm',
        '-y',
        audio_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
        if result.stderr:
            logger.warning(f"FFmpeg warning: {safe_decode(result.stderr)}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed. Error output:\n{safe_decode(e.stderr)}")
        raise ExternalToolError("FFmpeg failed to extract audio.") from e
    return audio_path

def mix_audio_tracks(video_path: str, synced_audio_path: str, session_folder: str, video_name: str, target_language: str, evaluated: bool = False) -> str:
    evaluation_suffix = "_eval" if evaluated else ""
    amplified_dubbed_audio_path = os.path.join(session_folder, f"amplified_dubbed_audio{evaluation_suffix}.wav")
    mixed_audio_path = os.path.join(session_folder, f"mixed_audio{evaluation_suffix}.wav")
    output_path = os.path.join(session_folder, f"final_output{evaluation_suffix}.mp4")

    original_audio_path = os.path.join(session_folder, "original_audio.wav")
    extract_audio_command = [
        'ffmpeg',
        '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '44100', '-ac', '2',
        original_audio_path
    ]
    try:
        subprocess.run(extract_audio_command, check=True)
    except subprocess.CalledProcessError as e:
        raise ExternalToolError("FFmpeg failed to extract original audio track.") from e

    analyze_command = [
        'ffmpeg',
        '-i', synced_audio_path,
        '-af', 'volumedetect',
        '-vn', '-sn', '-dn',
        '-f', 'null',
        os.devnull,
    ]
    try:
        result = subprocess.run(analyze_command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise ExternalToolError("FFmpeg volume analysis failed.") from e

    max_volume_lines = [line for line in result.stderr.split('\n') if 'max_volume' in line]
    if not max_volume_lines:
        raise ExternalToolError("FFmpeg did not return max_volume during volume analysis.")

    max_volume_line = max_volume_lines[0]
    max_volume = float(max_volume_line.split(':')[1].strip().split()[0])
    
    amplification = -max_volume

    amplify_command = [
        'ffmpeg',
        '-i', synced_audio_path,
        '-af', f'volume={amplification}dB',
        amplified_dubbed_audio_path
    ]
    try:
        subprocess.run(amplify_command, check=True)
    except subprocess.CalledProcessError as e:
        raise ExternalToolError("FFmpeg failed to amplify dubbed audio.") from e

    ffmpeg_command = [
        'ffmpeg',
        '-i', original_audio_path,
        '-i', amplified_dubbed_audio_path,
        '-filter_complex',
        "[1]silencedetect=n=-30dB:d=2[silence];"
        "[silence]aformat=sample_fmts=u8:sample_rates=44100:channel_layouts=mono,"
        "aresample=async=1000,pan=1c|c0=c0,"
        "aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=mono[silence_mono];"
        "[0][silence_mono]sidechaincompress=threshold=0.01:ratio=20:attack=100:release=500:makeup=1[gated];"
        "[1]volume=2[subtitles];"
        "[gated][subtitles]amix=inputs=2[mixed]",
        '-map', '[mixed]',
        mixed_audio_path
    ]
    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        raise ExternalToolError("FFmpeg failed while mixing audio tracks.") from e

    replace_audio_command = [
        'ffmpeg',
        '-i', video_path,
        '-i', mixed_audio_path,
        '-c:v', 'copy', 
        '-c:a', 'aac',   
        '-map', '0:v',   
        '-map', '1:a',   
        output_path
    ]
    try:
        subprocess.run(replace_audio_command, check=True)
    except subprocess.CalledProcessError as e:
        raise ExternalToolError("FFmpeg failed while muxing final output video.") from e

    return output_path
