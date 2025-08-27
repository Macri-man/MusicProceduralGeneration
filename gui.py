import sys
import numpy as np
import os
import random
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QComboBox, QFileDialog, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment

from audio_utils import process_effects, apply_pan, SAMPLE_RATE
from procedural_generator import generate_procedural_chunk, SCALES
from lfo import LFO, LayerLFO

DURATION_CHUNK = 5  # seconds per chunk

INSTRUMENTS = ['sine', 'square', 'triangle', 'sawtooth', 'fm_sine', 'noise_pad']

PRESET_FOLDER = "presets"

if not os.path.exists(PRESET_FOLDER):
    os.makedirs(PRESET_FOLDER)

class ProceduralMusicApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cinematic Procedural Ambient DAW")
        self.layout = QVBoxLayout()

        self.init_ui()
        self.init_lfos()
        self.init_audio_stream()
        self.connect_signals()

        self.recording_buffer = []
        self.time_accumulator = 0.0
        self.audio_stream = None
        self.tempo = 60

        self.timer = QTimer()
        self.timer.timeout.connect(self.stream_chunk)

    def init_ui(self):
        # Tempo slider
        self.tempo_label = QLabel(f"Tempo: {self.tempo} BPM")
        self.tempo_slider = QSlider(Qt.Orientation.Horizontal)
        self.tempo_slider.setRange(30, 200)
        self.tempo_slider.setValue(self.tempo)
        self.layout.addWidget(self.tempo_label)
        self.layout.addWidget(self.tempo_slider)

        # Scale selector
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(SCALES.keys())
        self.scale_combo.setCurrentText('minor')
        self.layout.addWidget(QLabel("Scale"))
        self.layout.addWidget(self.scale_combo)

        # Instrument selector
        self.inst_combo = QComboBox()
        self.inst_combo.addItems(INSTRUMENTS)
        self.layout.addWidget(QLabel("Instrument"))
        self.layout.addWidget(self.inst_combo)

        # Arpeggio toggle
        self.arpeggio_toggle = QCheckBox("Use Arpeggios / Inversions")
        self.arpeggio_toggle.setChecked(True)
        self.layout.addWidget(self.arpeggio_toggle)

        # Effects sliders
        self.reverb_slider = QSlider(Qt.Orientation.Horizontal)
        self.reverb_slider.setRange(0, 100)
        self.reverb_slider.setValue(30)
        self.layout.addWidget(QLabel("Reverb"))
        self.layout.addWidget(self.reverb_slider)

        self.delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.delay_slider.setRange(0, 100)
        self.delay_slider.setValue(30)
        self.layout.addWidget(QLabel("Delay"))
        self.layout.addWidget(self.delay_slider)

        self.chorus_slider = QSlider(Qt.Orientation.Horizontal)
        self.chorus_slider.setRange(0,100)
        self.chorus_slider.setValue(0)
        self.layout.addWidget(QLabel("Chorus"))
        self.layout.addWidget(self.chorus_slider)

        self.phaser_slider = QSlider(Qt.Orientation.Horizontal)
        self.phaser_slider.setRange(0,100)
        self.phaser_slider.setValue(0)
        self.layout.addWidget(QLabel("Phaser"))
        self.layout.addWidget(self.phaser_slider)

        self.stereo_slider = QSlider(Qt.Orientation.Horizontal)
        self.stereo_slider.setRange(0,100)
        self.stereo_slider.setValue(0)
        self.layout.addWidget(QLabel("Stereo Widen"))
        self.layout.addWidget(self.stereo_slider)

        self.lowpass_slider = QSlider(Qt.Orientation.Horizontal)
        self.lowpass_slider.setRange(20, 20000)
        self.lowpass_slider.setValue(15000)
        self.layout.addWidget(QLabel("Low-pass Filter"))
        self.layout.addWidget(self.lowpass_slider)

        self.highpass_slider = QSlider(Qt.Orientation.Horizontal)
        self.highpass_slider.setRange(20, 20000)
        self.highpass_slider.setValue(20)
        self.layout.addWidget(QLabel("High-pass Filter"))
        self.layout.addWidget(self.highpass_slider)

        # Preview and recording buttons
        self.preview_btn = QPushButton("Start/Stop Live Preview")
        self.preview_btn.setCheckable(True)
        self.layout.addWidget(self.preview_btn)

        self.record_btn = QPushButton("Start/Stop Recording")
        self.record_btn.setCheckable(True)
        self.layout.addWidget(self.record_btn)

        self.save_preset_btn = QPushButton("Save Preset")
        self.load_preset_btn = QPushButton("Load Preset")
        self.layout.addWidget(self.save_preset_btn)
        self.layout.addWidget(self.load_preset_btn)

        self.preset_combo = QComboBox()
        self.layout.addWidget(QLabel("Load Preset"))
        self.layout.addWidget(self.preset_combo)

        self.refresh_presets()

        self.random_preset_btn = QPushButton("Random Preset")
        self.layout.addWidget(self.random_preset_btn)

        self.setLayout(self.layout)

    def init_lfos(self):
        self.layer_lfos = [LayerLFO() for _ in range(4)]
        self.lfo_reverb = LFO(0.02, 0.3)
        self.lfo_delay = LFO(0.01, 0.2)

    def init_audio_stream(self):
        self.audio_stream = None

    def connect_signals(self):
        self.tempo_slider.valueChanged.connect(self.update_tempo)
        self.preview_btn.clicked.connect(self.toggle_live_preview)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.save_preset_btn.clicked.connect(self.save_preset)
        self.load_preset_btn.clicked.connect(self.load_preset)
        self.preset_combo.currentIndexChanged.connect(self.load_selected_preset)
        self.random_preset_btn.clicked.connect(self.generate_random_preset)

    def update_tempo(self, value):
        self.tempo = value
        self.tempo_label.setText(f"Tempo: {value} BPM")

    def toggle_live_preview(self):
        if self.preview_btn.isChecked():
            self.audio_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, dtype='float32')
            self.audio_stream.start()
            self.timer.start(DURATION_CHUNK*1000)
            self.stream_chunk()
        else:
            self.timer.stop()
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

    def toggle_recording(self):
        if self.record_btn.isChecked():
            self.recording_buffer = []
        else:
            if self.recording_buffer:
                full_audio = np.concatenate(self.recording_buffer)
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Save Recording", "", "MP3 Files (*.mp3);;WAV Files (*.wav)"
                )
                if filename:
                    if not filename.endswith(".wav") and not filename.endswith(".mp3"):
                        filename += ".wav"
                    write(filename, SAMPLE_RATE, (full_audio*32767).astype(np.int16))
                    AudioSegment.from_wav(filename).export(filename.replace('.wav','.mp3'), format='mp3')
                    print(f"Recording saved to {filename}")
                self.recording_buffer = []

    def stream_chunk(self):
        dt = DURATION_CHUNK
        self.time_accumulator += dt
        chunk = generate_procedural_chunk(
            DURATION_CHUNK, self.tempo,
            self.scale_combo.currentText(),
            self.inst_combo.currentText(),
            use_arpeggio=self.arpeggio_toggle.isChecked()
        )

        # Apply layer LFOs
        for lfo in self.layer_lfos:
            vol_mod, pan_mod = lfo.step(dt)
            mono = np.mean(chunk, axis=1)*(1+vol_mod)
            chunk = apply_pan(mono, pan_mod)

        # Apply global effect LFOs + new effects
        chunk = process_effects(
            chunk,
            reverb_amount=self.reverb_slider.value()/100*(1+self.lfo_reverb.step(dt)),
            delay_amount=self.delay_slider.value()/100*(1+self.lfo_delay.step(dt)),
            lowpass_cutoff=self.lowpass_slider.value(),
            highpass_cutoff=self.highpass_slider.value(),
            chorus_amount=self.chorus_slider.value()/100,
            phaser_amount=self.phaser_slider.value()/100,
            stereo_widen=self.stereo_slider.value()/100
        )

        if self.audio_stream:
            self.audio_stream.write(chunk)
        if self.record_btn.isChecked():
            self.recording_buffer.append(chunk)

    def save_preset(self):
        preset = {
            "tempo": self.tempo_slider.value(),
            "scale": self.scale_combo.currentText(),
            "instrument": self.inst_combo.currentText(),
            "use_arpeggio": self.arpeggio_toggle.isChecked(),
            "reverb": self.reverb_slider.value(),
            "delay": self.delay_slider.value(),
            "chorus": self.chorus_slider.value(),
            "phaser": self.phaser_slider.value(),
            "stereo_widen": self.stereo_slider.value(),
            "lowpass": self.lowpass_slider.value(),
            "highpass": self.highpass_slider.value()
        }
        filename, _ = QFileDialog.getSaveFileName(self, "Save Preset", "", "JSON Files (*.json)")
        if filename:
            if not filename.endswith(".json"):
                filename += ".json"
            with open(filename, 'w') as f:
                json.dump(preset, f, indent=4)
            print(f"Preset saved to {filename}")

    def load_preset(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON Files (*.json)")
        if filename and os.path.exists(filename):
            with open(filename, 'r') as f:
                preset = json.load(f)
            # Apply preset values
            self.tempo_slider.setValue(preset.get("tempo", 60))
            self.scale_combo.setCurrentText(preset.get("scale", "minor"))
            self.inst_combo.setCurrentText(preset.get("instrument", "sine"))
            self.arpeggio_toggle.setChecked(preset.get("use_arpeggio", True))
            self.reverb_slider.setValue(preset.get("reverb",30))
            self.delay_slider.setValue(preset.get("delay",30))
            self.chorus_slider.setValue(preset.get("chorus",0))
            self.phaser_slider.setValue(preset.get("phaser",0))
            self.stereo_slider.setValue(preset.get("stereo_widen",0))
            self.lowpass_slider.setValue(preset.get("lowpass",15000))
            self.highpass_slider.setValue(preset.get("highpass",20))
            print(f"Preset loaded from {filename}")

    def refresh_presets(self):
        """Refresh preset dropdown from folder"""
        self.preset_combo.clear()
        presets = [f for f in os.listdir(PRESET_FOLDER) if f.endswith(".json")]
        self.preset_combo.addItems(presets)

    def save_preset(self):
        preset_name, ok = QFileDialog.getSaveFileName(self, "Save Preset", PRESET_FOLDER, "JSON Files (*.json)")
        if preset_name:
            if not preset_name.endswith(".json"):
                preset_name += ".json"
            preset = {
                "tempo": self.tempo_slider.value(),
                "scale": self.scale_combo.currentText(),
                "instrument": self.inst_combo.currentText(),
                "use_arpeggio": self.arpeggio_toggle.isChecked(),
                "reverb": self.reverb_slider.value(),
                "delay": self.delay_slider.value(),
                "chorus": self.chorus_slider.value(),
                "phaser": self.phaser_slider.value(),
                "stereo_widen": self.stereo_slider.value(),
                "lowpass": self.lowpass_slider.value(),
                "highpass": self.highpass_slider.value()
            }
            with open(preset_name, 'w') as f:
                json.dump(preset, f, indent=4)
            print(f"Preset saved to {preset_name}")
            self.refresh_presets()

    def load_preset_file(self, filename):
        if filename and os.path.exists(filename):
            with open(filename, 'r') as f:
                preset = json.load(f)
            self.tempo_slider.setValue(preset.get("tempo", 60))
            self.scale_combo.setCurrentText(preset.get("scale", "minor"))
            self.inst_combo.setCurrentText(preset.get("instrument", "sine"))
            self.arpeggio_toggle.setChecked(preset.get("use_arpeggio", True))
            self.reverb_slider.setValue(preset.get("reverb",30))
            self.delay_slider.setValue(preset.get("delay",30))
            self.chorus_slider.setValue(preset.get("chorus",0))
            self.phaser_slider.setValue(preset.get("phaser",0))
            self.stereo_slider.setValue(preset.get("stereo_widen",0))
            self.lowpass_slider.setValue(preset.get("lowpass",15000))
            self.highpass_slider.setValue(preset.get("highpass",20))
            print(f"Preset loaded from {filename}")

    def generate_random_preset(self):
      """Randomly set all parameters to create a new preset"""
      tempo = random.randint(40, 160)
      self.tempo_slider.setValue(tempo)

      scale = random.choice(list(SCALES.keys()))
      self.scale_combo.setCurrentText(scale)

      instrument = random.choice(INSTRUMENTS)
      self.inst_combo.setCurrentText(instrument)

      self.arpeggio_toggle.setChecked(random.choice([True, False]))

      self.reverb_slider.setValue(random.randint(0, 100))
      self.delay_slider.setValue(random.randint(0, 100))
      self.chorus_slider.setValue(random.randint(0, 100))
      self.phaser_slider.setValue(random.randint(0, 100))
      self.stereo_slider.setValue(random.randint(0, 100))

      self.lowpass_slider.setValue(random.randint(5000, 20000))
      self.highpass_slider.setValue(random.randint(20, 5000))

      print("Random preset generated")

def load_selected_preset(self):
    preset_name = self.preset_combo.currentText()
    if preset_name:
        self.load_preset_file(os.path.join(PRESET_FOLDER, preset_name))

if __name__=="__main__":
    app=QApplication(sys.argv)
    window=ProceduralMusicApp()
    window.show()
    sys.exit(app.exec())
