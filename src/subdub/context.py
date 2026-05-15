import os
import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SessionContext:
    session_folder: str
    video_name: str
    video_path: str = ""
    audio_path: str = ""
    srt_path: str = ""
    
    @classmethod
    def create(cls, video_name: str, session_arg: str = None) -> 'SessionContext':
        if session_arg:
            if os.path.isabs(session_arg):
                session_folder = session_arg
            else:
                session_folder = os.path.abspath(session_arg)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_folder = f"subtitle_session_{video_name}_{timestamp}"
            session_folder = os.path.abspath(session_folder)
        
        os.makedirs(session_folder, exist_ok=True)
        logger.info(f"Using session folder: {session_folder}")
        
        return cls(session_folder=session_folder, video_name=video_name)
