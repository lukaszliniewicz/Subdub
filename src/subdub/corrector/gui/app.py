import os
import json
import numpy as np
from typing import List, Dict
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QLabel, QTableWidget, QSplitter, 
                             QTableWidgetItem, QCheckBox, QHeaderView, 
                             QAbstractItemView, QFileDialog, QMessageBox, 
                             QComboBox, QScrollBar, QDialog)
from PyQt6.QtGui import QKeySequence, QShortcut, QAction
from PyQt6.QtCore import Qt, QTimer

from ..config import CorrectorConfig
from ..engine import process_whisperx_json, segments_to_srt
from .player import AudioPlayer
from .canvas import InteractiveWaveformCanvas
from .dialogs import SplitSubtitleDialog, SearchReplaceDialog

class BoundaryVisualizerWindow(QMainWindow):
    """Main window for visualizing and editing boundary corrections."""
    
    def __init__(self, audio: np.ndarray = None, sr: int = None, segments: List[Dict] = None, 
                corrections: List[Dict] = None, config: CorrectorConfig = None, json_path: str = None,
                save_folder: str = None):  
        super().__init__()
        self.audio = audio
        self.sr = sr
        self.segments = segments or []
        self.corrections = corrections or []
        self.config = config or CorrectorConfig()
        self.json_path = json_path
        self.save_folder = save_folder  
        
        self.original_segments = [seg.copy() for seg in self.segments] if self.segments else []
        self.pre_correction_segments = self.reconstruct_pre_correction_segments(self.segments, self.corrections) if self.segments else []
        
        self.setWindowTitle("WhisperX Boundary Correction Editor")
        self.setGeometry(100, 100, 1600, 900)
        
        self.player = AudioPlayer(audio, sr) if audio is not None and sr is not None else None
        
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback_position)
        self.playback_timer.setInterval(50)
        
        self.search_dialog = None
        
        self.init_ui()
        self.create_menu_bar()
        self.setup_shortcuts()
        if self.segments:
            self.populate_table()
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('&File')
        
        open_new_action = QAction('&Open New...', self)
        open_new_action.setShortcut('Ctrl+O')
        open_new_action.triggered.connect(self.open_new_files)
        file_menu.addAction(open_new_action)
        
        file_menu.addSeparator()
        
        save_srt_action = QAction('Save as &SRT...', self)
        save_srt_action.setShortcut('Ctrl+S')
        save_srt_action.triggered.connect(self.save_as_srt)
        file_menu.addAction(save_srt_action)
        
        save_json_action = QAction('Save as &JSON...', self)
        save_json_action.setShortcut('Ctrl+Shift+S')
        save_json_action.triggered.connect(self.save_as_json)
        file_menu.addAction(save_json_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        edit_menu = menubar.addMenu('&Edit')
        
        search_action = QAction('&Search and Replace...', self)
        search_action.setShortcut('Ctrl+F')
        search_action.triggered.connect(self.show_search_dialog)
        edit_menu.addAction(search_action)
        
        edit_menu.addSeparator()
        
        split_action = QAction('&Split Subtitle...', self)
        split_action.setShortcut('Ctrl+T')
        split_action.triggered.connect(self.split_current_subtitle)
        edit_menu.addAction(split_action)
        
        view_menu = menubar.addMenu('&View')
        
        self.show_corrected_action = QAction('Show Only &Corrected', self)
        self.show_corrected_action.setCheckable(True)
        self.show_corrected_action.triggered.connect(self.filter_table)
        view_menu.addAction(self.show_corrected_action)
    
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        table_widget = QWidget()
        table_layout = QVBoxLayout()
        table_widget.setLayout(table_layout)
        
        controls_layout = QHBoxLayout()
        
        self.show_only_corrected = QCheckBox("Show Only Corrected")
        self.show_only_corrected.stateChanged.connect(self.filter_table)
        controls_layout.addWidget(self.show_only_corrected)
        
        controls_layout.addWidget(QLabel("  |  "))
        
        controls_layout.addWidget(QLabel("Play Mode:"))
        self.play_mode_combo = QComboBox()
        self.play_mode_combo.addItems(["Play Subtitle", "Play Context"])
        self.play_mode_combo.setCurrentText("Play Subtitle")
        controls_layout.addWidget(self.play_mode_combo)
        
        self.play_btn = QPushButton("▶ Play (Space)")
        self.play_btn.clicked.connect(self.play_current_mode)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⬛ Stop (S)")
        self.stop_btn.clicked.connect(self.stop_playback)
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addStretch()
        
        self.search_btn = QPushButton("🔍 Search (Ctrl+F)")
        self.search_btn.clicked.connect(self.show_search_dialog)
        controls_layout.addWidget(self.search_btn)
        
        self.split_btn = QPushButton("✂️ Split (Ctrl+T)")
        self.split_btn.clicked.connect(self.split_current_subtitle)
        controls_layout.addWidget(self.split_btn)
        
        controls_layout.addStretch()
        
        self.save_srt_btn = QPushButton("Save as SRT")
        self.save_srt_btn.clicked.connect(self.save_as_srt)
        controls_layout.addWidget(self.save_srt_btn)
        
        self.save_json_btn = QPushButton("Save as JSON")
        self.save_json_btn.clicked.connect(self.save_as_json)
        controls_layout.addWidget(self.save_json_btn)
        
        table_layout.addLayout(controls_layout)
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Subtitle Text", 
            "End Time", 
            "Gap to Next",
            "Merge Up",
            "Merge Down"
        ])
        
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        self.table.itemChanged.connect(self.on_item_changed)
        
        table_layout.addWidget(self.table)
        
        waveform_widget = QWidget()
        waveform_layout = QVBoxLayout()
        waveform_widget.setLayout(waveform_layout)
        
        controls_layout = QHBoxLayout()
        
        self.zoom_in_btn = QPushButton("🔍+ Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        controls_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("🔍- Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        controls_layout.addWidget(self.zoom_out_btn)
        
        self.reset_zoom_btn = QPushButton("🏠 Reset View")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        controls_layout.addWidget(self.reset_zoom_btn)
        
        controls_layout.addStretch()
        
        instruction_label = QLabel("💡 Hold Shift while dragging for fine adjustment (1ms steps). Normal: 10ms steps. Both start and end boundaries are editable.")
        instruction_label.setStyleSheet("color: #666; font-size: 10px; padding: 2px;")
        controls_layout.addWidget(instruction_label)
        
        waveform_layout.addLayout(controls_layout)
        
        canvas_layout = QVBoxLayout()
        
        self.canvas = InteractiveWaveformCanvas(width=12, height=6)
        if self.audio is not None and self.sr is not None:
            self.canvas.set_audio_data(self.audio, self.sr, self.segments, self.corrections)
        self.canvas.boundary_changed.connect(self.on_boundary_changed)
        self.canvas.view_changed.connect(self.on_view_changed)
        
        canvas_layout.addWidget(self.canvas)
        
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.valueChanged.connect(self.on_scrollbar_changed)
        self.scrollbar.setEnabled(False)
        canvas_layout.addWidget(self.scrollbar)
        
        waveform_layout.addLayout(canvas_layout)
        
        splitter.addWidget(table_widget)
        splitter.addWidget(waveform_widget)
        splitter.setSizes([450, 450])
        
        main_layout.addWidget(splitter)
    
    def setup_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, self.play_current_mode)
        QShortcut(QKeySequence("S"), self, self.stop_playback)
        
        QShortcut(QKeySequence("Up"), self, self.select_previous_row)
        QShortcut(QKeySequence("Down"), self, self.select_next_row)
        
        QShortcut(QKeySequence("Ctrl++"), self, self.zoom_in)
        QShortcut(QKeySequence("Ctrl+="), self, self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.reset_zoom)
    
    def open_new_files(self):
        json_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON File", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if not json_path:
            return
        
        audio_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)"
        )
        
        if not audio_path:
            return
        
        try:
            segments, corrections, audio, json_file_path = process_whisperx_json(json_path, audio_path, self.config)
            
            self.audio = audio
            self.sr = self.config.sample_rate
            self.segments = segments
            self.corrections = corrections
            self.json_path = json_file_path
            
            self.original_segments = [seg.copy() for seg in self.segments]
            self.pre_correction_segments = self.reconstruct_pre_correction_segments(self.segments, self.corrections)
            
            self.player = AudioPlayer(self.audio, self.sr)
            
            self.canvas.set_audio_data(self.audio, self.sr, self.segments, self.corrections)
            
            self.populate_table()
            
            QMessageBox.information(self, "Success", f"Loaded {len(self.segments)} segments from {json_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load files: {str(e)}")
    
    def show_search_dialog(self):
        if self.search_dialog is None:
            self.search_dialog = SearchReplaceDialog(self)
        
        self.search_dialog.show()
        self.search_dialog.raise_()
        self.search_dialog.activateWindow()
    
    def split_current_subtitle(self):
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.segments):
            QMessageBox.warning(self, "No Selection", "Please select a subtitle to split.")
            return
        
        segment = self.segments[current_row]
        
        words = segment['text'].strip().split()
        if len(words) < 2:
            QMessageBox.warning(self, "Cannot Split", "Subtitle must have at least 2 words to split.")
            return
        
        dialog = SplitSubtitleDialog(segment, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.perform_split(current_row, dialog.split_position, dialog.split_time)
    
    def perform_split(self, segment_index: int, split_position: int, split_time: float):
        segment = self.segments[segment_index]
        words = segment['text'].strip().split()
        
        text1 = ' '.join(words[:split_position + 1])
        text2 = ' '.join(words[split_position + 1:])
        
        segment1 = {
            'start': segment['start'],
            'end': split_time,
            'text': text1
        }
        
        segment2 = {
            'start': split_time,
            'end': segment['end'],
            'text': text2
        }
        
        if 'words' in segment and segment['words']:
            try:
                words_data = segment['words']
                
                word_split_index = min(split_position + 1, len(words_data))
                
                if word_split_index < len(words_data):
                    segment1['words'] = words_data[:word_split_index]
                    segment2['words'] = words_data[word_split_index:]
                else:
                    segment1['words'] = words_data
            except (IndexError, KeyError):
                pass
        
        self.segments[segment_index] = segment1
        self.segments.insert(segment_index + 1, segment2)
        
        original_segment = self.original_segments[segment_index].copy()
        self.original_segments[segment_index] = original_segment
        self.original_segments.insert(segment_index + 1, original_segment.copy())
        
        if segment_index < len(self.pre_correction_segments):
            pre_correction_segment = self.pre_correction_segments[segment_index].copy()
            self.pre_correction_segments[segment_index] = pre_correction_segment
            self.pre_correction_segments.insert(segment_index + 1, pre_correction_segment.copy())
        
        new_corrections = []
        for correction in self.corrections:
            if correction['segment_index'] > segment_index:
                correction['segment_index'] += 1
            new_corrections.append(correction)
        self.corrections = new_corrections
        
        self.canvas.set_audio_data(self.audio, self.sr, self.segments, self.corrections)
        
        self.populate_table()
        
        self.table.selectRow(segment_index)
        
        QMessageBox.information(self, "Split Complete", f"Subtitle split into 2 parts at {split_time:.3f}s")
    
    def zoom_in(self):
        self.canvas.zoom_in()

    def zoom_out(self):
        self.canvas.zoom_out()

    def reset_zoom(self):
        if self.canvas.current_segment_index is not None:
            self.canvas.plot_segment(self.canvas.current_segment_index, self.config)
    
    def on_view_changed(self, start_time: float, end_time: float):
        if not self.audio or not self.segments:
            return
        
        total_start = 0
        total_end = len(self.audio) / self.sr
        total_range = total_end - total_start
        view_range = end_time - start_time
        
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(int((total_range - view_range) * 1000))
        self.scrollbar.setPageStep(int(view_range * 1000))
        self.scrollbar.setValue(int((start_time - total_start) * 1000))
        self.scrollbar.setEnabled(True)
    
    def on_scrollbar_changed(self, value):
        if not self.audio or not self.segments:
            return
        
        total_start = 0
        total_end = len(self.audio) / self.sr
        
        xlim = self.canvas.ax.get_xlim()
        view_range = xlim[1] - xlim[0]
        
        new_start = total_start + (value / 1000.0)
        new_end = new_start + view_range
        
        if new_end > total_end:
            new_end = total_end
            new_start = new_end - view_range
        
        self.canvas.set_view_range(new_start, new_end)
    
    def populate_table(self):
        self.table.setRowCount(len(self.segments))
        
        self.table.itemChanged.disconnect()
        
        for i in range(len(self.segments)):
            self.update_table_row(i)
        
        for i in range(len(self.segments)):
            self.set_row_background_color(i, i)
        
        self.table.itemChanged.connect(self.on_item_changed)

    def update_table_row(self, segment_index: int):
        if segment_index >= len(self.segments):
            return
            
        segment = self.segments[segment_index]
        
        pre_corr = self.pre_correction_segments[segment_index] if segment_index < len(self.pre_correction_segments) else segment
        original = self.original_segments[segment_index] if segment_index < len(self.original_segments) else segment
        
        auto_corrected = (segment['end'] != pre_corr['end'] or 
                        segment['start'] != pre_corr['start'])
        
        manually_edited = (segment['end'] != original['end'] or 
                        segment['start'] != original['start'] or
                        segment['text'] != original['text'])
        
        was_modified = auto_corrected or manually_edited
        
        gap_text = ""
        if segment_index < len(self.segments) - 1:
            next_segment = self.segments[segment_index + 1]
            gap = (next_segment['start'] - segment['end']) * 1000
            gap_text = f"{gap:.1f}ms"
        
        text_item = self.table.item(segment_index, 0)
        if text_item is None:
            text_item = QTableWidgetItem(segment['text'])
            text_item.setFlags(text_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(segment_index, 0, text_item)
        else:
            text_item.setText(segment['text'])
        text_item.setData(Qt.ItemDataRole.UserRole, {'segment_index': segment_index, 'was_modified': was_modified})
        
        end_item = self.table.item(segment_index, 1)
        if end_item is None:
            end_item = QTableWidgetItem(f"{segment['end']:.3f}")
            end_item.setFlags(end_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(segment_index, 1, end_item)
        else:
            end_item.setText(f"{segment['end']:.3f}")
        
        gap_item = self.table.item(segment_index, 2)
        if gap_item is None:
            gap_item = QTableWidgetItem(gap_text)
            gap_item.setFlags(gap_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(segment_index, 2, gap_item)
        else:
            gap_item.setText(gap_text)
        
        font = gap_item.font()
        font.setBold(was_modified)
        gap_item.setFont(font)
        
        if self.table.cellWidget(segment_index, 3) is None:
            merge_up_btn = QPushButton("↑ Merge Up")
            merge_up_btn.setEnabled(segment_index > 0)
            merge_up_btn.clicked.connect(lambda checked, idx=segment_index: self.merge_up(idx))
            self.table.setCellWidget(segment_index, 3, merge_up_btn)
        
        if self.table.cellWidget(segment_index, 4) is None:
            merge_down_btn = QPushButton("↓ Merge Down")
            merge_down_btn.setEnabled(segment_index < len(self.segments) - 1)
            merge_down_btn.clicked.connect(lambda checked, idx=segment_index: self.merge_down(idx))
            self.table.setCellWidget(segment_index, 4, merge_down_btn)

    def merge_up(self, segment_index: int):
        if segment_index <= 0:
            return
        
        current = self.segments[segment_index]
        previous = self.segments[segment_index - 1]
        
        merged_text = previous['text'] + ' ' + current['text']
        merged_segment = {
            'start': previous['start'],
            'end': current['end'],
            'text': merged_text
        }
        
        if 'words' in previous and 'words' in current:
            merged_segment['words'] = previous['words'] + current['words']
        elif 'words' in previous:
            merged_segment['words'] = previous['words']
        elif 'words' in current:
            merged_segment['words'] = current['words']
        
        self.segments[segment_index - 1] = merged_segment
        del self.segments[segment_index]
        
        del self.original_segments[segment_index]
        
        if segment_index < len(self.pre_correction_segments):
            del self.pre_correction_segments[segment_index]
        
        new_corrections = []
        for correction in self.corrections:
            if correction['segment_index'] > segment_index:
                correction['segment_index'] -= 1
                new_corrections.append(correction)
            elif correction['segment_index'] < segment_index:
                new_corrections.append(correction)
        self.corrections = new_corrections
        
        if self.audio is not None and self.sr is not None:
            self.canvas.set_audio_data(self.audio, self.sr, self.segments, self.corrections)
        
        self.populate_table()
        
        self.table.selectRow(segment_index - 1)

    def merge_down(self, segment_index: int):
        if segment_index >= len(self.segments) - 1:
            return
        
        current = self.segments[segment_index]
        next_seg = self.segments[segment_index + 1]
        
        merged_text = current['text'] + ' ' + next_seg['text']
        merged_segment = {
            'start': current['start'],
            'end': next_seg['end'],
            'text': merged_text
        }
        
        if 'words' in current and 'words' in next_seg:
            merged_segment['words'] = current['words'] + next_seg['words']
        elif 'words' in current:
            merged_segment['words'] = current['words']
        elif 'words' in next_seg:
            merged_segment['words'] = next_seg['words']
        
        self.segments[segment_index] = merged_segment
        del self.segments[segment_index + 1]
        
        del self.original_segments[segment_index + 1]
        
        if segment_index + 1 < len(self.pre_correction_segments):
            del self.pre_correction_segments[segment_index + 1]
        
        new_corrections = []
        for correction in self.corrections:
            if correction['segment_index'] > segment_index + 1:
                correction['segment_index'] -= 1
                new_corrections.append(correction)
            elif correction['segment_index'] <= segment_index:
                new_corrections.append(correction)
        self.corrections = new_corrections
        
        if self.audio is not None and self.sr is not None:
            self.canvas.set_audio_data(self.audio, self.sr, self.segments, self.corrections)
        
        self.populate_table()
        
        self.table.selectRow(segment_index)

    def update_corrections(self):
        corrections = []
        
        for i, (seg, orig) in enumerate(zip(self.segments, self.original_segments)):
            if i < len(self.original_segments):
                if seg['end'] != orig['end']:
                    corrections.append({
                        'type': 'manual_edit',
                        'segment_index': i,
                        'old_end': orig['end'],
                        'new_end': seg['end']
                    })
                if seg['start'] != orig['start']:
                    corrections.append({
                        'type': 'manual_edit',
                        'segment_index': i,
                        'old_start': orig['start'],
                        'new_start': seg['start']
                    })
                if seg['text'] != orig['text']:
                    corrections.append({
                        'type': 'text_edit',
                        'segment_index': i,
                        'old_text': orig['text'],
                        'new_text': seg['text']
                    })
        
        return corrections

    def set_row_background_color(self, segment_index: int, visual_index: int):
        from PyQt6.QtGui import QColor, QPalette
        palette = self.palette()
        
        if visual_index % 2 == 0:
            base_color = palette.color(QPalette.ColorRole.Base)
            bg_color = base_color.lighter(110)
        else:
            base_color = palette.color(QPalette.ColorRole.Base)
            bg_color = base_color.darker(105)
        
        for col in range(3):
            item = self.table.item(segment_index, col)
            if item:
                item.setBackground(bg_color)

    def filter_table(self):
        show_only_corrected = self.show_only_corrected.isChecked()
        
        self.show_corrected_action.setChecked(show_only_corrected)
        
        visible_segments = []
        
        for segment_index in range(len(self.segments)):
            item = self.table.item(segment_index, 0)
            if item:
                data = item.data(Qt.ItemDataRole.UserRole)
                was_modified = data.get('was_modified', False) if data else False
                
                if show_only_corrected and not was_modified:
                    self.table.hideRow(segment_index)
                else:
                    self.table.showRow(segment_index)
                    visible_segments.append(segment_index)
        
        for visual_index, segment_index in enumerate(visible_segments):
            self.set_row_background_color(segment_index, visual_index)

    def reconstruct_pre_correction_segments(self, segments: List[Dict], corrections: List[Dict]) -> List[Dict]:
        pre_correction = [seg.copy() for seg in segments]
        
        for correction in corrections:
            if correction['type'] == 'energy_boundary':
                idx = correction['segment_index']
                if idx < len(pre_correction):
                    pre_correction[idx]['end'] = correction['old_end']
            elif correction['type'] == 'overlap':
                idx = correction['segment_index']
                if idx < len(pre_correction):
                    pre_correction[idx]['end'] = correction['old_end']
        
        return pre_correction
    
    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == event.Type.PaletteChange:
            if self.segments:
                self.populate_table()

    def on_item_changed(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        
        segment_index = data['segment_index']
        new_text = item.text()
        
        if segment_index < len(self.segments):
            self.segments[segment_index]['text'] = new_text
            if 'words' in self.segments[segment_index]:
                del self.segments[segment_index]['words']
        
        self.mark_row_as_modified(segment_index)

    def mark_row_as_modified(self, segment_index):
        if segment_index < len(self.segments):
            item = self.table.item(segment_index, 0)
            if item:
                data = item.data(Qt.ItemDataRole.UserRole)
                if data:
                    data['was_modified'] = True
                    item.setData(Qt.ItemDataRole.UserRole, data)
            
            self.update_table_row(segment_index)
            
            if self.show_only_corrected.isChecked():
                self.table.showRow(segment_index)

    def on_selection_changed(self):
        if self.player:
            self.stop_playback()
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            segment_index = selected_rows[0].row()
            if segment_index < len(self.segments) and self.audio is not None:
                self.canvas.plot_segment(segment_index, self.config)
        else:
            self.canvas.clear_plot()
    
    def on_boundary_changed(self, segment_index: int, boundary_type: str, new_time: float):
        if segment_index < len(self.segments):
            if boundary_type == 'end':
                self.segments[segment_index]['end'] = new_time
            else:
                self.segments[segment_index]['start'] = new_time
            
            if hasattr(self.canvas, 'draggable_lines'):
                for line, info in self.canvas.draggable_lines.items():
                    if info['segment_index'] == segment_index:
                        if boundary_type == 'start' and info['boundary_type'] == 'end':
                            info['min_time'] = new_time + 0.05
                        elif boundary_type == 'end' and info['boundary_type'] == 'start':
                            info['max_time'] = new_time - 0.05
                    elif info['segment_index'] == segment_index - 1 and boundary_type == 'start':
                        if info['boundary_type'] == 'end':
                            info['max_time'] = new_time - 0.02
                    elif info['segment_index'] == segment_index + 1 and boundary_type == 'end':
                        if info['boundary_type'] == 'start':
                            info['min_time'] = new_time + 0.02
            
            self.update_table_row(segment_index)
            if segment_index > 0:
                self.update_table_row(segment_index - 1)
            if segment_index < len(self.segments) - 1:
                self.update_table_row(segment_index + 1)
            
            current_xlim = self.canvas.ax.get_xlim()
            current_ylim = self.canvas.ax.get_ylim()
            
            self.canvas.plot_segment(segment_index, self.config)
            
            self.canvas.ax.set_xlim(current_xlim)
            self.canvas.ax.set_ylim(current_ylim)
            self.canvas.draw()

    def play_current_mode(self):
        if not self.player:
            QMessageBox.warning(self, "No Audio", "Please load an audio file first.")
            return
            
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.segments):
            return
        
        mode = self.play_mode_combo.currentText()
        
        if mode == "Play Subtitle":
            self.play_subtitle()
        elif mode == "Play Context":
            self.play_context()

    def play_subtitle(self):
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.segments):
            return
        
        segment = self.segments[current_row]
        
        start = segment['start']
        end = segment['end']
        duration = end - start
        
        self.canvas.show_playback_region(start, end)
        
        self.playback_timer.start()
        self.player.play_segment(start, duration, self.canvas.update_playback_position)

    def play_context(self):
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.segments):
            return
        
        segment = self.segments[current_row]
        prev_segment = self.segments[current_row - 1] if current_row > 0 else None
        next_segment = self.segments[current_row + 1] if current_row < len(self.segments) - 1 else None
        
        if prev_segment:
            context_start = prev_segment['start'] + (prev_segment['end'] - prev_segment['start']) / 2
        else:
            context_start = max(0, segment['start'] - self.config.window_padding)
        
        if next_segment:
            context_end = next_segment['start'] + (next_segment['end'] - next_segment['start']) / 2
        else:
            context_end = min(segment['end'] + self.config.window_padding, len(self.audio) / self.sr)
        
        duration = context_end - context_start
        
        self.canvas.show_playback_region(context_start, context_end)
        
        self.playback_timer.start()
        self.player.play_segment(context_start, duration, self.canvas.update_playback_position)

    def stop_playback(self):
        if self.player:
            self.player.stop()
        self.playback_timer.stop()
        self.canvas.update_playback_position(-1)
    
    def update_playback_position(self):
        if not self.player or not self.player.is_playing:
            self.playback_timer.stop()
    
    def select_previous_row(self):
        current_row = self.table.currentRow()
        if current_row > 0:
            self.table.selectRow(current_row - 1)

    def select_next_row(self):
        current_row = self.table.currentRow()
        if current_row < self.table.rowCount() - 1:
            self.table.selectRow(current_row + 1)
    
    def save_as_srt(self):
        if not self.segments:
            QMessageBox.warning(self, "No Data", "Please load segments first.")
            return
        
        if self.save_folder:
            if self.json_path:
                base_name = os.path.splitext(os.path.basename(self.json_path))[0]
            else:
                base_name = "corrected"
            
            file_path = os.path.join(self.save_folder, f"{base_name}.srt")
            
            os.makedirs(self.save_folder, exist_ok=True)
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save SRT File", "", "SRT Files (*.srt);;All Files (*)"
            )
            
            if not file_path:
                return
        
        try:
            srt_content = segments_to_srt(self.segments)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            self.update_corrections()
            
            QMessageBox.information(self, "Success", f"Saved SRT to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save SRT: {str(e)}")

    def save_as_json(self):
        if not self.segments:
            QMessageBox.warning(self, "No Data", "Please load segments first.")
            return
        
        if self.save_folder:
            if self.json_path:
                base_name = os.path.splitext(os.path.basename(self.json_path))[0]
            else:
                base_name = "corrected"
            
            file_path = os.path.join(self.save_folder, f"{base_name}_corrected.json")
            
            os.makedirs(self.save_folder, exist_ok=True)
        else:
            default_path = self.json_path.replace('.json', '_corrected.json') if self.json_path else ""
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save JSON File", default_path, "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return
        
        try:
            data = {
                'segments': self.segments,
                'corrections': self.update_corrections()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self, "Success", f"Saved JSON to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON: {str(e)}")
