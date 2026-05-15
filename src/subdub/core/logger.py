import logging
import os
import sys

def setup_logging(session_folder: str, log_to_file: bool) -> None:
    handlers = []

    # Configure console handler (always on)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # Configure file handler if requested
    if log_to_file:
        log_file = os.path.join(session_folder, 'subtitle_app.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # force=True removes any existing handlers (e.g., from imported libraries)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
        force=True
    )

    log_message = "Logging initialized"
    if log_to_file:
        log_message += " (file and console)."
    else:
        log_message += " (console only)."
        
    logging.info(log_message)
