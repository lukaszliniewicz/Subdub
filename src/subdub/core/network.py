import os
import yt_dlp
from typing import Tuple

def download_from_url(url: str, session_folder: str) -> Tuple[str, str]:
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prefer MP4, but fall back to best available format
        'outtmpl': {
            'default': os.path.join(session_folder, '%(title)s.%(ext)s')
        },
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info['title']
        sanitized_title = ''.join(e for e in video_title if e.isalnum() or e in ['-', '_', ' '])
        ydl_opts['outtmpl']['default'] = os.path.join(session_folder, f'{sanitized_title}.%(ext)s')
        
        ydl.download([url])
        
    downloaded_files = os.listdir(session_folder)
    video_file = next((f for f in downloaded_files if f.startswith(sanitized_title)), None)
    
    if not video_file:
        raise FileNotFoundError("Downloaded video file not found in the session folder.")
    
    return os.path.join(session_folder, video_file), sanitized_title
