import numpy as np
import sounddevice as sd

class AudioPlayer:
    """Simple audio player for boundary playback."""
    
    def __init__(self, audio: np.ndarray, sr: int):
        self.audio = audio
        self.sr = sr
        self.is_playing = False
        self.start_idx = 0
        self.current_position = 0
        self.callback_position = None
        self.stream = None
    
    def play_segment(self, start_time: float, duration: float, position_callback=None):
        """Play audio segment."""
        self.stop()
        
        start_sample = int(start_time * self.sr)
        end_sample = int((start_time + duration) * self.sr)
        
        # Ensure valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio), end_sample)
        
        if start_sample >= end_sample:
            return
        
        audio_segment = self.audio[start_sample:end_sample]
        self.start_idx = start_sample
        self.callback_position = position_callback
        self.is_playing = True
        self.current_position = 0
        
        def callback(outdata, frames, time, status):
            if self.is_playing and self.callback_position:
                current_time = (self.start_idx + self.current_position) / self.sr
                self.callback_position(current_time)
            
            # Get the next chunk of audio
            chunksize = min(len(audio_segment) - self.current_position, frames)
            if chunksize > 0:
                outdata[:chunksize] = audio_segment[self.current_position:self.current_position + chunksize].reshape(-1, 1)
            
            if chunksize < frames:
                outdata[chunksize:] = 0
                self.is_playing = False
                raise sd.CallbackStop()
            
            self.current_position += chunksize
        
        self.stream = sd.OutputStream(callback=callback, samplerate=self.sr, channels=1)
        self.stream.start()
    
    def stop(self):
        """Stop playback."""
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.callback_position:
            self.callback_position(-1)  # Clear position marker
