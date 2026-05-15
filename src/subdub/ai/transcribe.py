import os
import subprocess
import logging

from ..errors import ConfigurationError, ExternalToolError

logger = logging.getLogger(__name__)

def safe_decode(byte_string):
    try:
        return byte_string.decode('utf-8')
    except UnicodeDecodeError:
        return byte_string.decode('utf-8', errors='ignore')

def transcribe_audio(audio_path: str, language: str, session_folder: str, video_name: str, whisper_model: str, align_model: str = None, initial_prompt: str = None, diarize: bool = False, hf_token: str = None, chunk_size: int = None, boundary_correction: bool = True, save_txt: bool = False) -> str:
    # Determine output format based on boundary correction setting
    # We assume CORRECTOR_AVAILABLE is handled at a higher level or passed in, for now we'll just use the flag
    expected_format = 'json' if boundary_correction else 'srt'
    whisperx_output_format = 'all' if save_txt else expected_format
    output_file = os.path.join(session_folder, f"{video_name}.{expected_format}")
    
    base_whisperx_args = [
        audio_path,
        '--model', whisper_model,
        '--language', language,
        '--output_format', whisperx_output_format,
        '--output_dir', session_folder,
        '--print_progress', 'True',
        '--vad_method', 'silero',
        '--chunk_size', str(chunk_size)        
    ]

    if align_model:
        base_whisperx_args.extend(['--align_model', align_model])
        logger.info(f"WhisperX will use alignment model: {align_model}")
    else:
        logger.info("WhisperX will use its default alignment model for the specified language (if any).")

    if diarize:
        if not hf_token:
            raise ConfigurationError("HF token is required for diarization. Please provide --hf_token, set HF_TOKEN config, or set HF_TOKEN environment variable.")
        base_whisperx_args.extend(['--diarize'])
        base_whisperx_args.extend(['--hf_token', hf_token])
        logger.info("WhisperX will perform speaker diarization")

    try:
        whisperx_command = ['whisperx'] + base_whisperx_args
        logger.info(f"Attempting direct whisperx command: {' '.join(whisperx_command)}")
        result = subprocess.run(whisperx_command, check=True, capture_output=True)
        if result.stderr:
            logger.warning(f"WhisperX warning: {safe_decode(result.stderr)}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Direct whisperx command failed, trying conda run method. Error: {str(e)}")
        try:
            conda_exe = os.environ.get("CONDA_EXE", "../conda/Scripts/conda.exe")
            conda_env = os.environ.get("WHISPERX_CONDA_ENV", "../conda/envs/whisperx_installer")
            conda_whisperx_command = [
                conda_exe, "run", "-p", conda_env, "--no-capture-output",
                "python", "-m", "whisperx"
            ] + base_whisperx_args
            logger.info(f"Attempting conda run whisperx command: {' '.join(conda_whisperx_command)}")
            result = subprocess.run(conda_whisperx_command, check=True, capture_output=True)
            if result.stderr:
                logger.warning(f"WhisperX warning: {safe_decode(result.stderr)}")
        except subprocess.CalledProcessError as e_conda:
            logger.error(f"WhisperX command failed using both methods.")
            if isinstance(e, subprocess.CalledProcessError):
                 logger.error(f"Direct WhisperX Error output:\n{safe_decode(e.stderr)}")
            logger.error(f"Conda WhisperX Error output:\n{safe_decode(e_conda.stderr)}")
            raise ExternalToolError("WhisperX failed with both direct and conda execution paths.") from e_conda
    
    whisperx_output_filename_base = os.path.splitext(os.path.basename(audio_path))[0]
    whisperx_generated_file_path = os.path.join(session_folder, f"{whisperx_output_filename_base}.{expected_format}")
    
    if os.path.exists(whisperx_generated_file_path):
        os.rename(whisperx_generated_file_path, output_file)
    else:
        potential_files = [f for f in os.listdir(session_folder) if f.startswith(whisperx_output_filename_base) and f.endswith(f".{expected_format}")]
        if potential_files:
            actual_whisperx_output = os.path.join(session_folder, potential_files[0])
            logger.warning(f"Expected WhisperX {expected_format.upper()} file not found at {whisperx_generated_file_path}. Found and using: {actual_whisperx_output}")
            os.rename(actual_whisperx_output, output_file)
        else:
            raise ExternalToolError(
                f"WhisperX did not produce the expected {expected_format.upper()} output file. "
                f"Looked for {whisperx_generated_file_path} and similar patterns."
            )
            
    return output_file
