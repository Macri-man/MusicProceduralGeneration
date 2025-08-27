import sys
import numpy as np
import os
import random
import json
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

        self.scenes = []
        self.random_scene_enabled = False
        self.current_scene_index = 0
        self.scene_duration = 30  
        self.scene_timer = 0.0
        self.auto_scene_enabled = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.stream_chunk)

    def init_ui(self):
        
        self.random_scene_toggle = QCheckBox("Enable Procedural Random Scenes")
        self.random_scene_toggle.setChecked(False)
        self.layout.addWidget(self.random_scene_toggle)

        
        self.auto_scene_toggle = QCheckBox("Enable Automatic Scene Switching")
        self.layout.addWidget(self.auto_scene_toggle)

        self.scene_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.scene_duration_slider.setRange(5, 300)  # 5s to 5min per scene
        self.scene_duration_slider.setValue(self.scene_duration)
        self.layout.addWidget(QLabel("Scene Duration (s)"))
        self.layout.addWidget(self.scene_duration_slider)

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

        self.evolving_toggle = QCheckBox("Enable Evolving Preset")
        self.evolving_toggle.setChecked(False)
        self.layout.addWidget(self.evolving_toggle)

        self.export_btn = QPushButton("Export Full Session")
        self.layout.addWidget(self.export_btn)
        self.session_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.session_duration_slider.setRange(10, 3600)  # 10s to 1h
        self.session_duration_slider.setValue(180)  # default 3 min
        self.layout.addWidget(QLabel("Session Duration (s)"))
        self.layout.addWidget(self.session_duration_slider)

        self.setLayout(self.layout)

    def init_lfos(self):
        # LFOs for evolving preset parameters
        self.lfo_tempo = LFO(rate=0.005, amplitude=20)  # BPM modulation
        self.lfo_reverb = LFO(rate=0.002, amplitude=0.5)  # Reverb modulation
        self.lfo_delay = LFO(rate=0.002, amplitude=0.5)   # Delay modulation
        self.lfo_chorus = LFO(rate=0.001, amplitude=0.5)
        self.lfo_phaser = LFO(rate=0.001, amplitude=0.5)
        self.lfo_stereo = LFO(rate=0.001, amplitude=0.5)
        # LFOs per instrument layer
        self.layer_lfos = []
        for i in range(4):  # 4 layers: drone, chords, melody, noise
            lfo = {
                "volume": LFO(rate=0.001 + 0.001*i, amplitude=0.3),  # Different speeds
                "pan": LFO(rate=0.0005 + 0.0005*i, amplitude=0.5),
                "timbre": LFO(rate=0.0007 + 0.0003*i, amplitude=0.5)  # For FM/Noise modulation
            }
            self.layer_lfos.append(lfo)

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
        self.auto_scene_toggle.stateChanged.connect(self.toggle_auto_scene)
        self.scene_duration_slider.valueChanged.connect(self.update_scene_duration)
        self.random_scene_toggle.stateChanged.connect(self.toggle_random_scene)
        self.export_btn.clicked.connect(self.export_full_session)

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

        # --- Scene Switching ---
        if self.auto_scene_enabled:
            self.scene_timer += dt
            if self.scene_timer >= self.scene_duration:
                self.scene_timer = 0.0
                self.advance_scene()

        # --- Tempo modulation ---
        if self.evolving_toggle.isChecked():
            mod_tempo = int(self.tempo + self.lfo_tempo.step(dt))
            mod_tempo = max(mod_tempo, 20)
        else:
            mod_tempo = self.tempo

        # --- Generate procedural layers ---
        # Returns a list of 2D numpy arrays: [layer0, layer1, layer2, layer3]
        layers = generate_procedural_chunk(
            DURATION_CHUNK,
            mod_tempo,
            self.scale_combo.currentText(),
            self.inst_combo.currentText(),
            use_arpeggio=self.arpeggio_toggle.isChecked(),
            return_layers=True  # <-- make sure generator supports multiple layers
        )

        # --- Apply Layer LFOs ---
        processed_layers = []
        for i, layer in enumerate(layers):
            lfo = self.layer_lfos[i % len(self.layer_lfos)]  # safety
            vol_mod = lfo["volume"].step(dt)
            pan_mod = lfo["pan"].step(dt)
            timbre_mod = lfo["timbre"].step(dt)

            # Mono mix, volume modulation
            mono = np.mean(layer, axis=1) * (1 + vol_mod)

            # Pan modulation
            stereo = apply_pan(mono, pan_mod)

            # Timbre modulation for FM/Noise
            if self.inst_combo.currentText() in ["fm_sine", "noise_pad"]:
                stereo *= (1 + 0.2 * timbre_mod)

            processed_layers.append(stereo)

        # --- Mix all layers ---
        chunk = np.sum(processed_layers, axis=0)
        chunk = np.clip(chunk, -1, 1)

        # --- Apply global evolving effects ---
        if self.evolving_toggle.isChecked():
            reverb_amount = min(max(self.reverb_slider.value()/100 + self.lfo_reverb.step(dt),0),1)
            delay_amount = min(max(self.delay_slider.value()/100 + self.lfo_delay.step(dt),0),1)
            chorus_amount = min(max(self.chorus_slider.value()/100 + self.lfo_chorus.step(dt),0),1)
            phaser_amount = min(max(self.phaser_slider.value()/100 + self.lfo_phaser.step(dt),0),1)
            stereo_widen = min(max(self.stereo_slider.value()/100 + self.lfo_stereo.step(dt),0),1)
        else:
            reverb_amount = self.reverb_slider.value()/100
            delay_amount = self.delay_slider.value()/100
            chorus_amount = self.chorus_slider.value()/100
            phaser_amount = self.phaser_slider.value()/100
            stereo_widen = self.stereo_slider.value()/100

        chunk = process_effects(
            chunk,
            reverb_amount=reverb_amount,
            delay_amount=delay_amount,
            lowpass_cutoff=self.lowpass_slider.value(),
            highpass_cutoff=self.highpass_slider.value(),
            chorus_amount=chorus_amount,
            phaser_amount=phaser_amount,
            stereo_widen=stereo_widen
        )

        # --- Playback and Recording ---
        if self.audio_stream:
            self.audio_stream.write(chunk.astype(np.float32))
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

    def toggle_auto_scene(self):
        self.auto_scene_enabled = self.auto_scene_toggle.isChecked()
        self.load_scene_list()

    def update_scene_duration(self, value):
        self.scene_duration = value

    def load_scene_list(self):
        """Load all presets in the folder as scenes"""
        self.scenes = [os.path.join(PRESET_FOLDER, f) for f in os.listdir(PRESET_FOLDER) if f.endswith(".json")]
        self.current_scene_index = 0

    def advance_scene(self):
        if self.random_scene_enabled:
            self.generate_random_scene()
        else:
            if not self.scenes:
                return
            self.current_scene_index = (self.current_scene_index + 1) % len(self.scenes)
            self.load_preset_file(self.scenes[self.current_scene_index])
            print(f"Switched to scene: {os.path.basename(self.scenes[self.current_scene_index])}")

    def toggle_random_scene(self):
        self.random_scene_enabled = self.random_scene_toggle.isChecked()
        if self.random_scene_enabled:
            self.auto_scene_enabled = True
            self.auto_scene_toggle.setChecked(True)

    def generate_random_scene(self):
        """Generate a fully random preset scene and apply it"""
        self.generate_random_preset()  # Reuse existing random preset generator
        # Optional: randomize LFOs for more dynamic evolution
        for i, lfo in enumerate(self.layer_lfos):
            lfo["volume"].rate = random.uniform(0.0005, 0.005)
            lfo["volume"].amplitude = random.uniform(0.1, 0.5)
            lfo["pan"].rate = random.uniform(0.0002, 0.003)
            lfo["pan"].amplitude = random.uniform(0.3, 0.7)
            lfo["timbre"].rate = random.uniform(0.0001, 0.003)
            lfo["timbre"].amplitude = random.uniform(0.1, 0.5)

        # Randomize global LFOs
        self.lfo_tempo.rate = random.uniform(0.002, 0.01)
        self.lfo_tempo.amplitude = random.uniform(5, 25)
        self.lfo_reverb.rate = random.uniform(0.001, 0.005)
        self.lfo_reverb.amplitude = random.uniform(0.2, 0.6)
        self.lfo_delay.rate = random.uniform(0.001, 0.005)
        self.lfo_delay.amplitude = random.uniform(0.2, 0.6)
        self.lfo_chorus.rate = random.uniform(0.0005, 0.003)
        self.lfo_chorus.amplitude = random.uniform(0.1, 0.5)
        self.lfo_phaser.rate = random.uniform(0.0005, 0.003)
        self.lfo_phaser.amplitude = random.uniform(0.1, 0.5)
        self.lfo_stereo.rate = random.uniform(0.0005, 0.003)
        self.lfo_stereo.amplitude = random.uniform(0.1, 0.5)

        print("Procedural random scene generated")

    def export_full_session(self):
        total_duration = self.session_duration_slider.value()
        chunk_duration = DURATION_CHUNK
        num_chunks = int(total_duration / chunk_duration)
        session_audio = []

        print(f"Exporting full session: {total_duration}s ({num_chunks} chunks)")

        # Save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Full Session", "", "MP3 Files (*.mp3);;WAV Files (*.wav)"
        )
        if not filename:
            return
        if not (filename.endswith(".wav") or filename.endswith(".mp3")):
            filename += ".wav"

        # Reset scene timer and index
        self.scene_timer = 0.0
        current_scene_idx = self.current_scene_index

        for i in range(num_chunks):
            # --- Scene switching ---
            if self.auto_scene_enabled and (i * chunk_duration) % self.scene_duration == 0:
                self.scene_timer = 0.0
                self.advance_scene()

            # --- Tempo modulation for export ---
            if self.evolving_toggle.isChecked():
                mod_tempo = int(self.tempo + self.lfo_tempo.step(chunk_duration))
                mod_tempo = max(mod_tempo, 20)
            else:
                mod_tempo = self.tempo

            # --- Generate procedural layers ---
            layers = generate_procedural_chunk(
                chunk_duration,
                mod_tempo,
                self.scale_combo.currentText(),
                self.inst_combo.currentText(),
                use_arpeggio=self.arpeggio_toggle.isChecked(),
                return_layers=True
            )

            # --- Apply Layer LFOs ---
            processed_layers = []
            for j, layer in enumerate(layers):
                lfo = self.layer_lfos[j % len(self.layer_lfos)]
                vol_mod = lfo["volume"].step(chunk_duration)
                pan_mod = lfo["pan"].step(chunk_duration)
                timbre_mod = lfo["timbre"].step(chunk_duration)

                mono = np.mean(layer, axis=1) * (1 + vol_mod)
                stereo = apply_pan(mono, pan_mod)

                if self.inst_combo.currentText() in ["fm_sine", "noise_pad"]:
                    stereo *= (1 + 0.2 * timbre_mod)

                processed_layers.append(stereo)

            # --- Mix layers ---
            chunk = np.sum(processed_layers, axis=0)
            chunk = np.clip(chunk, -1, 1)

            # --- Apply global evolving effects ---
            if self.evolving_toggle.isChecked():
                reverb_amount = min(max(self.reverb_slider.value()/100 + self.lfo_reverb.step(chunk_duration),0),1)
                delay_amount = min(max(self.delay_slider.value()/100 + self.lfo_delay.step(chunk_duration),0),1)
                chorus_amount = min(max(self.chorus_slider.value()/100 + self.lfo_chorus.step(chunk_duration),0),1)
                phaser_amount = min(max(self.phaser_slider.value()/100 + self.lfo_phaser.step(chunk_duration),0),1)
                stereo_widen = min(max(self.stereo_slider.value()/100 + self.lfo_stereo.step(chunk_duration),0),1)
            else:
                reverb_amount = self.reverb_slider.value()/100
                delay_amount = self.delay_slider.value()/100
                chorus_amount = self.chorus_slider.value()/100
                phaser_amount = self.phaser_slider.value()/100
                stereo_widen = self.stereo_slider.value()/100

            chunk = process_effects(
                chunk,
                reverb_amount=reverb_amount,
                delay_amount=delay_amount,
                lowpass_cutoff=self.lowpass_slider.value(),
                highpass_cutoff=self.highpass_slider.value(),
                chorus_amount=chorus_amount,
                phaser_amount=phaser_amount,
                stereo_widen=stereo_widen
            )

            session_audio.append(chunk)

        # --- Concatenate all chunks ---
        full_audio = np.concatenate(session_audio)
        full_audio = np.clip(full_audio, -1, 1)

        # --- Save WAV ---
        write(filename, SAMPLE_RATE, (full_audio*32767).astype(np.int16))
        print(f"Session exported as {filename}")

        # --- Optional: convert to MP3 ---
        if filename.endswith(".wav"):
            try:
                AudioSegment.from_wav(filename).export(filename.replace(".wav", ".mp3"), format="mp3")
                print(f"MP3 version saved: {filename.replace('.wav','.mp3')}")
            except:
                print("pydub not installed or failed, MP3 conversion skipped")


if __name__=="__main__":
    app=QApplication(sys.argv)
    window=ProceduralMusicApp()
    window.show()
    sys.exit(app.exec())
