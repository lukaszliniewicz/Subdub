from typing import Dict, List
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QPushButton, 
                             QLabel, QScrollArea, QGroupBox, QRadioButton, 
                             QButtonGroup, QDialog, QDialogButtonBox, QGridLayout,
                             QLineEdit, QCheckBox, QMessageBox)
from PyQt6.QtCore import Qt

class SplitSubtitleDialog(QDialog):
    """Dialog for splitting a subtitle at a specific word position."""
    
    def __init__(self, segment: Dict, parent=None):
        super().__init__(parent)
        self.segment = segment
        self.split_position = None
        self.split_time = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Split Subtitle")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout()
        
        instruction_label = QLabel("Click after a word to split the subtitle at that position:")
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)
        
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QHBoxLayout()
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        
        self.word_buttons = []
        words = self.segment['text'].strip().split()
        
        has_word_timing = 'words' in self.segment and self.segment['words']
        
        from PyQt6.QtGui import QPalette
        palette = self.palette()
        
        for i, word in enumerate(words):
            word_btn = QPushButton(word)
            word_btn.setFlat(True)
            
            word_btn.setStyleSheet(f"""
                QPushButton {{ 
                    text-align: left; 
                    padding: 4px 8px; 
                    margin: 2px; 
                    border: 1px solid {palette.color(QPalette.ColorRole.Mid).name()};
                    background-color: {palette.color(QPalette.ColorRole.Base).name()};
                    color: {palette.color(QPalette.ColorRole.Text).name()};
                }}
                QPushButton:hover {{
                    background-color: {palette.color(QPalette.ColorRole.Highlight).name()};
                    color: {palette.color(QPalette.ColorRole.HighlightedText).name()};
                }}
            """)
            
            word_btn.clicked.connect(lambda checked, idx=i: self.set_split_position(idx))
            self.word_buttons.append(word_btn)
            scroll_layout.addWidget(word_btn)
            
            if i < len(words) - 1:
                split_btn = QPushButton("|")
                split_btn.setMaximumWidth(20)
                split_btn.setStyleSheet(f"""
                    QPushButton {{ 
                        color: red; 
                        font-weight: bold; 
                        background-color: {palette.color(QPalette.ColorRole.Base).name()};
                        border: 1px solid {palette.color(QPalette.ColorRole.Mid).name()};
                    }}
                    QPushButton:hover {{
                        background-color: {palette.color(QPalette.ColorRole.Highlight).name()};
                    }}
                """)
                split_btn.clicked.connect(lambda checked, idx=i: self.set_split_position(idx))
                scroll_layout.addWidget(split_btn)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(100)
        layout.addWidget(scroll_area)
        
        self.position_label = QLabel("No split position selected")
        layout.addWidget(self.position_label)
        
        timing_group = QGroupBox("Timing Method")
        timing_layout = QVBoxLayout()
        
        self.timing_button_group = QButtonGroup()
        
        if has_word_timing:
            self.precise_radio = QRadioButton("Use precise word timing (recommended)")
            self.precise_radio.setChecked(True)
            self.timing_button_group.addButton(self.precise_radio, 0)
            timing_layout.addWidget(self.precise_radio)
        
        self.estimate_radio = QRadioButton("Estimate timing based on character position")
        if not has_word_timing:
            self.estimate_radio.setChecked(True)
        self.timing_button_group.addButton(self.estimate_radio, 1 if has_word_timing else 0)
        timing_layout.addWidget(self.estimate_radio)
        
        timing_group.setLayout(timing_layout)
        layout.addWidget(timing_group)
        
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; margin: 5px; }")
        layout.addWidget(self.preview_label)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        self.ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setEnabled(False)
        
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def set_split_position(self, position: int):
        self.split_position = position
        
        words = self.segment['text'].strip().split()
        
        from PyQt6.QtGui import QPalette
        palette = self.palette()
        
        for btn in self.word_buttons:
            btn.setStyleSheet(f"""
                QPushButton {{ 
                    text-align: left; 
                    padding: 4px 8px; 
                    margin: 2px; 
                    border: 1px solid {palette.color(QPalette.ColorRole.Mid).name()};
                    background-color: {palette.color(QPalette.ColorRole.Base).name()};
                    color: {palette.color(QPalette.ColorRole.Text).name()};
                }}
                QPushButton:hover {{
                    background-color: {palette.color(QPalette.ColorRole.Highlight).name()};
                    color: {palette.color(QPalette.ColorRole.HighlightedText).name()};
                }}
            """)
        
        if position < len(self.word_buttons):
            self.word_buttons[position].setStyleSheet(f"""
                QPushButton {{ 
                    text-align: left; 
                    padding: 4px 8px; 
                    margin: 2px; 
                    background-color: #ffeb3b; 
                    color: black;
                    border: 2px solid orange;
                }}
            """)
        
        self.calculate_split_time()
        self.position_label.setText(f"Split after word {position + 1}: '{words[position]}'")
        self.update_preview()
        self.ok_button.setEnabled(True)
    
    def calculate_split_time(self):
        if self.split_position is None:
            return
        
        words = self.segment['text'].strip().split()
        has_word_timing = 'words' in self.segment and self.segment['words']
        
        use_precise = (has_word_timing and 
                      hasattr(self, 'precise_radio') and 
                      self.precise_radio.isChecked())
        
        if use_precise:
            try:
                word_data = self.segment['words']
                if self.split_position < len(word_data):
                    self.split_time = word_data[self.split_position]['end']
                else:
                    self.split_time = self.estimate_split_time(words)
            except (KeyError, IndexError):
                self.split_time = self.estimate_split_time(words)
        else:
            self.split_time = self.estimate_split_time(words)
    
    def estimate_split_time(self, words: List[str]) -> float:
        text_before = ' '.join(words[:self.split_position + 1])
        total_text = self.segment['text'].strip()
        
        char_position = len(text_before)
        total_chars = len(total_text)
        
        duration = self.segment['end'] - self.segment['start']
        char_ratio = char_position / total_chars if total_chars > 0 else 0.5
        
        return self.segment['start'] + (duration * char_ratio)
    
    def update_preview(self):
        if self.split_position is None:
            self.preview_label.setText("")
            return
        
        words = self.segment['text'].strip().split()
        
        text1 = ' '.join(words[:self.split_position + 1])
        text2 = ' '.join(words[self.split_position + 1:])
        
        preview_text = f"<b>First segment:</b> {self.segment['start']:.3f}s - {self.split_time:.3f}s<br>"
        preview_text += f"Text: \"{text1}\"<br><br>"
        preview_text += f"<b>Second segment:</b> {self.split_time:.3f}s - {self.segment['end']:.3f}s<br>"
        preview_text += f"Text: \"{text2}\""
        
        self.preview_label.setText(preview_text)

class SearchReplaceDialog(QDialog):
    """Dialog for search and replace functionality."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.current_match_index = -1
        self.matches = []
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Search and Replace")
        self.setModal(False)
        self.resize(400, 200)
        
        layout = QGridLayout()
        
        layout.addWidget(QLabel("Find:"), 0, 0)
        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(self.on_search_text_changed)
        layout.addWidget(self.search_input, 0, 1)
        
        layout.addWidget(QLabel("Replace:"), 1, 0)
        self.replace_input = QLineEdit()
        layout.addWidget(self.replace_input, 1, 1)
        
        button_layout = QHBoxLayout()
        
        self.find_next_btn = QPushButton("Find Next")
        self.find_next_btn.clicked.connect(self.find_next)
        button_layout.addWidget(self.find_next_btn)
        
        self.find_prev_btn = QPushButton("Find Previous")
        self.find_prev_btn.clicked.connect(self.find_previous)
        button_layout.addWidget(self.find_prev_btn)
        
        self.replace_btn = QPushButton("Replace")
        self.replace_btn.clicked.connect(self.replace_current)
        button_layout.addWidget(self.replace_btn)
        
        self.replace_all_btn = QPushButton("Replace All")
        self.replace_all_btn.clicked.connect(self.replace_all)
        button_layout.addWidget(self.replace_all_btn)
        
        layout.addLayout(button_layout, 2, 0, 1, 2)
        
        self.match_label = QLabel("No matches")
        layout.addWidget(self.match_label, 3, 0, 1, 2)
        
        self.case_sensitive = QCheckBox("Case sensitive")
        layout.addWidget(self.case_sensitive, 4, 0, 1, 2)
        self.case_sensitive.stateChanged.connect(self.on_search_text_changed)
        
        self.setLayout(layout)
        self.search_input.setFocus()
    
    def on_search_text_changed(self):
        self.find_matches()
        self.current_match_index = -1
        self.update_match_label()
        
        if self.matches:
            self.find_next()
    
    def find_matches(self):
        self.matches = []
        search_text = self.search_input.text()
        
        if not search_text:
            return
        
        case_sensitive = self.case_sensitive.isChecked()
        
        for i, segment in enumerate(self.parent_window.segments):
            text = segment['text']
            search_in = text if case_sensitive else text.lower()
            search_for = search_text if case_sensitive else search_text.lower()
            
            start_pos = 0
            while True:
                pos = search_in.find(search_for, start_pos)
                if pos == -1:
                    break
                
                self.matches.append({
                    'segment_index': i,
                    'start_pos': pos,
                    'end_pos': pos + len(search_text),
                    'text': text[pos:pos + len(search_text)]
                })
                start_pos = pos + 1
    
    def update_match_label(self):
        if not self.matches:
            self.match_label.setText("No matches")
        else:
            current = self.current_match_index + 1 if self.current_match_index >= 0 else 0
            self.match_label.setText(f"Match {current} of {len(self.matches)}")
    
    def find_next(self):
        if not self.matches:
            return
        
        self.current_match_index = (self.current_match_index + 1) % len(self.matches)
        self.highlight_current_match()
    
    def find_previous(self):
        if not self.matches:
            return
        
        self.current_match_index = (self.current_match_index - 1) % len(self.matches)
        self.highlight_current_match()
    
    def highlight_current_match(self):
        if self.current_match_index < 0 or self.current_match_index >= len(self.matches):
            return
        
        match = self.matches[self.current_match_index]
        
        self.parent_window.table.selectRow(match['segment_index'])
        self.parent_window.table.scrollToItem(
            self.parent_window.table.item(match['segment_index'], 0)
        )
        
        self.update_match_label()
    
    def replace_current(self):
        if self.current_match_index < 0 or self.current_match_index >= len(self.matches):
            return
        
        match = self.matches[self.current_match_index]
        segment = self.parent_window.segments[match['segment_index']]
        
        old_text = segment['text']
        new_text = (old_text[:match['start_pos']] + 
                   self.replace_input.text() + 
                   old_text[match['end_pos']:])
        
        segment['text'] = new_text
        
        self.parent_window.update_table_row(match['segment_index'])
        self.parent_window.mark_row_as_modified(match['segment_index'])
        
        self.find_matches()
        
        if self.matches and self.current_match_index < len(self.matches):
            self.highlight_current_match()
        else:
            self.current_match_index = -1
            self.update_match_label()
    
    def replace_all(self):
        if not self.matches:
            return
        
        replace_count = 0
        search_text = self.search_input.text()
        replace_text = self.replace_input.text()
        case_sensitive = self.case_sensitive.isChecked()
        
        processed_segments = set()
        
        for match in reversed(self.matches):
            segment_index = match['segment_index']
            
            if segment_index in processed_segments:
                continue
            
            segment = self.parent_window.segments[segment_index]
            old_text = segment['text']
            
            if case_sensitive:
                new_text = old_text.replace(search_text, replace_text)
            else:
                import re
                pattern = re.compile(re.escape(search_text), re.IGNORECASE)
                new_text = pattern.sub(replace_text, old_text)
            
            if new_text != old_text:
                segment['text'] = new_text
                self.parent_window.update_table_row(segment_index)
                self.parent_window.mark_row_as_modified(segment_index)
                replace_count += 1
            
            processed_segments.add(segment_index)
        
        QMessageBox.information(self, "Replace All", f"Replaced {replace_count} occurrences in {len(processed_segments)} segments.")
        
        self.find_matches()
        self.current_match_index = -1
        self.update_match_label()
