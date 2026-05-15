import numpy as np
from typing import List, Dict
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import cursors
from PyQt6.QtCore import Qt, pyqtSignal

class InteractiveWaveformCanvas(FigureCanvas):
    """Interactive canvas for displaying and editing waveform boundaries."""
    
    boundary_changed = pyqtSignal(int, str, float)
    view_changed = pyqtSignal(float, float)
    
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        
        self.audio = None
        self.sr = None
        self.segments = []
        self.corrections = []
        self.time_axis = None
        self.current_segment_index = None
        
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        
        self.draggable_lines = {}
        self.selected_line = None
        self.drag_data = None
        
        self.playback_line = None
        self.playback_region = None
        
        self.fine_mode = False
        self.coarse_step = 0.01
        self.fine_step = 0.001
        
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.mpl_connect('key_press_event', self.on_key_press)
        self.mpl_connect('key_release_event', self.on_key_release)
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.hover_line = None
        self.mpl_connect('motion_notify_event', self.on_hover)
        
        self.ax.callbacks.connect('xlim_changed', self.on_xlim_changed)
    
    def on_xlim_changed(self, ax):
        xlim = ax.get_xlim()
        self.view_changed.emit(xlim[0], xlim[1])
    
    def on_key_press(self, event):
        if event.key == 'shift':
            self.fine_mode = True
    
    def on_key_release(self, event):
        if event.key == 'shift':
            self.fine_mode = False
    
    def snap_to_grid(self, time_value):
        step = self.fine_step if self.fine_mode else self.coarse_step
        return round(time_value / step) * step
    
    def on_hover(self, event):
        if event.inaxes != self.ax:
            if self.hover_line:
                self.hover_line.set_linewidth(3)
                self.hover_line = None
                self.set_cursor(cursors.POINTER)
                self.draw_idle()
            return
        
        for line in self.draggable_lines:
            if line.contains(event)[0]:
                if self.hover_line != line:
                    if self.hover_line:
                        self.hover_line.set_linewidth(3)
                    line.set_linewidth(5)
                    self.hover_line = line
                    self.set_cursor(cursors.RESIZE_HORIZONTAL)
                    self.draw_idle()
                return
        
        if self.hover_line:
            self.hover_line.set_linewidth(3)
            self.hover_line = None
            self.set_cursor(cursors.POINTER)
            self.draw_idle()
    
    def on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        
        for line, info in self.draggable_lines.items():
            if line.contains(event)[0]:
                self.selected_line = line
                self.drag_data = {
                    'info': info,
                    'start_x': event.xdata,
                    'original_x': line.get_xdata()[0]
                }
                return
    
    def on_mouse_motion(self, event):
        if self.selected_line is None or event.inaxes != self.ax:
            return
        
        info = self.drag_data['info']
        new_x = event.xdata
        
        if new_x is None:
            return
        
        new_x = self.snap_to_grid(new_x)
        new_x = max(info['min_time'], min(info['max_time'], new_x))
        
        self.selected_line.set_xdata([new_x, new_x])
        
        if self.current_segment_index is not None:
            self.update_gap_indicator()
        
        self.draw_idle()
    
    def on_mouse_release(self, event):
        if self.selected_line is None:
            return
        
        new_time = self.selected_line.get_xdata()[0]
        info = self.drag_data['info']
        
        self.boundary_changed.emit(info['segment_index'], info['boundary_type'], new_time)
        
        self.selected_line = None
        self.drag_data = None
    
    def set_audio_data(self, audio: np.ndarray, sr: int, segments: List[Dict], corrections: List[Dict]):
        self.audio = audio
        self.sr = sr
        self.segments = segments
        self.corrections = corrections
        self.time_axis = np.arange(len(audio)) / sr
    
    def plot_segment(self, segment_index: int, config):
        self.ax.clear()
        self.draggable_lines = {}
        self.current_segment_index = segment_index
        self.playback_line = None
        self.playback_region = None
        
        if segment_index >= len(self.segments):
            return
        
        segment = self.segments[segment_index]
        
        original_correction = None
        for corr in self.corrections:
            if corr['segment_index'] == segment_index and corr['type'] == 'energy_boundary':
                original_correction = corr
                break
        
        prev_segment = self.segments[segment_index - 1] if segment_index > 0 else None
        next_segment = self.segments[segment_index + 1] if segment_index < len(self.segments) - 1 else None
        
        if prev_segment:
            context_start = prev_segment['start'] + (prev_segment['end'] - prev_segment['start']) / 2
        else:
            context_start = max(0, segment['start'] - config.window_padding)
        
        if next_segment:
            context_end = next_segment['start'] + (next_segment['end'] - next_segment['start']) / 2
        else:
            context_end = min(segment['end'] + config.window_padding, self.time_axis[-1])
        
        start_sample = int(context_start * self.sr)
        end_sample = min(int(context_end * self.sr), len(self.audio))
        
        audio_segment = self.audio[start_sample:end_sample]
        time_segment = self.time_axis[start_sample:end_sample]
        
        self.ax.plot(time_segment, audio_segment, 'b-', alpha=0.7, linewidth=0.5)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True, alpha=0.3)
        
        y_min, y_max = self.ax.get_ylim()
        
        seg_rect = Rectangle((segment['start'], y_min), segment['end'] - segment['start'], 
                           y_max - y_min, alpha=0.2, color='blue')
        self.ax.add_patch(seg_rect)
        
        if prev_segment:
            prev_start = max(context_start, prev_segment['start'] + (prev_segment['end'] - prev_segment['start']) / 2)
            prev_rect = Rectangle((prev_start, y_min), prev_segment['end'] - prev_start, 
                                y_max - y_min, alpha=0.1, color='green')
            self.ax.add_patch(prev_rect)
        
        if next_segment:
            next_end = min(context_end, next_segment['start'] + (next_segment['end'] - next_segment['start']) / 2)
            next_rect = Rectangle((next_segment['start'], y_min), next_end - next_segment['start'], 
                                y_max - y_min, alpha=0.1, color='orange')
            self.ax.add_patch(next_rect)
        
        if segment_index > 0:
            line = self.ax.axvline(x=segment['start'], color='green', linewidth=3, 
                                label='Segment Start', alpha=0.8, picker=5)
            self.draggable_lines[line] = {
                'segment_index': segment_index,
                'boundary_type': 'start',
                'min_time': prev_segment['end'] + 0.02 if prev_segment else max(0, context_start),
                'max_time': segment['end'] - 0.05
            }
        
        if segment_index < len(self.segments) - 1:
            line = self.ax.axvline(x=segment['end'], color='red', linewidth=3, 
                                label='Segment End', alpha=0.8, picker=5)
            self.draggable_lines[line] = {
                'segment_index': segment_index,
                'boundary_type': 'end',
                'min_time': segment['start'] + 0.05,
                'max_time': next_segment['start'] - 0.02 if next_segment else min(context_end, self.time_axis[-1])
            }
            
            gap_ms = (next_segment['start'] - segment['end']) * 1000
            gap_center = (segment['end'] + next_segment['start']) / 2
            self.gap_text = self.ax.text(gap_center, y_max * 0.9, f'{gap_ms:.0f}ms', 
                        ha='center', va='center', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        if original_correction and abs(original_correction['old_end'] - segment['end']) > 0.001:
            self.ax.axvline(x=original_correction['old_end'], color='red', linewidth=1, 
                        linestyle='--', alpha=0.5, label='Original End')
        
        self.ax.legend(loc='upper right')
        self.ax.set_xlim(context_start, context_end)
        
        self.draw()
    
    def update_gap_indicator(self):
        if self.current_segment_index is None or self.current_segment_index >= len(self.segments) - 1:
            return
        
        segment = self.segments[self.current_segment_index]
        next_segment = self.segments[self.current_segment_index + 1]
        
        gap_ms = (next_segment['start'] - segment['end']) * 1000
        gap_center = (segment['end'] + next_segment['start']) / 2
        
        if hasattr(self, 'gap_text'):
            self.gap_text.set_text(f'{gap_ms:.0f}ms')
            self.gap_text.set_position((gap_center, self.gap_text.get_position()[1]))
    
    def show_playback_region(self, start_time: float, end_time: float):
        if self.playback_region:
            self.playback_region.remove()
        
        y_min, y_max = self.ax.get_ylim()
        self.playback_region = Rectangle((start_time, y_min), end_time - start_time,
                                       y_max - y_min, alpha=0.2, color='purple')
        self.ax.add_patch(self.playback_region)
        self.draw_idle()
    
    def update_playback_position(self, time_seconds: float):
        if self.playback_line:
            self.playback_line.remove()
            self.playback_line = None
        
        if time_seconds >= 0:
            self.playback_line = self.ax.axvline(x=time_seconds, color='purple', 
                                                linewidth=2, alpha=0.8)
            self.draw_idle()
        else:
            if self.playback_region:
                self.playback_region.remove()
                self.playback_region = None
            self.draw_idle()
    
    def zoom_in(self):
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        width = (xlim[1] - xlim[0]) * 0.8
        self.ax.set_xlim(center - width/2, center + width/2)
        self.draw()
    
    def zoom_out(self):
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        width = (xlim[1] - xlim[0]) * 1.25
        self.ax.set_xlim(center - width/2, center + width/2)
        self.draw()
    
    def set_view_range(self, start_time: float, end_time: float):
        self.ax.set_xlim(start_time, end_time)
        self.draw()
    
    def clear_plot(self):
        self.ax.clear()
        self.draggable_lines = {}
        self.current_segment_index = None
        self.playback_line = None
        self.playback_region = None
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True, alpha=0.3)
        self.draw()
