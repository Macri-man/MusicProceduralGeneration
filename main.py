import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QComboBox
)
from PyQt6.QtCore import Qt, QTimer
import sounddevice as sd

# =====================
# SETTINGS
# =====================
SAMPLE_RATE = 44100
DURATION_CHUNK = 5  # seconds per generated chunk
TEMPO = 60
SCALE = 'minor'
INSTRUMENTS = ['sine', 'square', 'triangle', 'sawtooth']

SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11, 12],
    'minor': [0, 2, 3, 5, 7, 8, 10, 12],
    'pentatonic': [0, 2, 4, 7, 9, 12],
    'chromatic': list(range(13))
}

# =====================
# SOUND UTILS
# =====================
def midi_to_freq(midi_note):
    return 440 * 2 ** ((midi_note - 69) / 12)

def generate_tone(frequency, duration, instrument='sine', volume=0.2):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    if instrument == 'sine':
        wave = np.sin(frequency * t * 2 * np.pi)
    elif instrument == 'square':
        wave = np.sign(np.sin(frequency * t * 2 * np.pi))
    elif instrument == 'triangle':
        wave = 2 * np.arcsin(np.sin(2 * np.pi * frequency * t)) / np.pi
    elif instrument == 'sawtooth':
        wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
    return (wave * volume).astype(np.float32)

def generate_noise(duration, volume=0.05):
    n_samples = int(duration * SAMPLE_RATE)
    return np.random.normal(0, 1, (n_samples,)).astype(np.float32) * volume

def apply_envelope(signal, attack=0.1, decay=0.5):
    n_samples = len(signal)
    env = np.ones(n_samples)
    attack_samples = int(attack * n_samples)
    decay_samples = int(decay * n_samples)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        env[-decay_samples:] = np.linspace(1, 0, decay_samples)
    return signal * env

def apply_pan(signal, pan=0.0):
    left = signal * (1 - pan) / 2
    right = signal * (1 + pan) / 2
    stereo = np.stack([left, right], axis=1)
    return stereo

# =====================
# PROCEDURAL MUSIC GENERATION
# =====================
def generate_procedural_chunk(duration, tempo, scale, instrument):
    beats = int(duration / 60 * tempo)
    audio = np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)
    scale_notes = SCALES[scale]

    # Drone layer
    for i in range(beats):
        root = 48 + np.random.choice(scale_notes)
        freq = midi_to_freq(root)
        start_idx = int(i * (SAMPLE_RATE * 60 / tempo))
        end_idx = start_idx + int(SAMPLE_RATE * 60 / tempo)
        tone = generate_tone(freq, 60 / tempo, instrument, 0.08)
        tone = apply_envelope(tone, attack=0.3, decay=0.7)
        audio[start_idx:end_idx] += tone[:len(audio[start_idx:end_idx])]

    # Chord layer
    for i in range(beats // 2):
        root = 60 + np.random.choice(scale_notes)
        chord = [root, root + scale_notes[2], root + scale_notes[4]]
        start_idx = int(i * 2 * (SAMPLE_RATE * 60 / tempo))
        end_idx = start_idx + int(2 * (SAMPLE_RATE * 60 / tempo))
        for note in chord:
            freq = midi_to_freq(note)
            tone = generate_tone(freq, 2 * 60 / tempo, instrument, 0.05)
            tone = apply_envelope(tone, attack=0.5, decay=0.5)
            audio[start_idx:end_idx] += tone[:len(audio[start_idx:end_idx])]

    # Melody layer
    for i in range(beats):
        if np.random.rand() < 0.2:
            note = 60 + np.random.choice(scale_notes)
            freq = midi_to_freq(note)
            start_idx = int(i * (SAMPLE_RATE * 60 / tempo))
            end_idx = start_idx + int((SAMPLE_RATE * 60 / tempo) / 2)
            tone = generate_tone(freq, 60 / tempo / 2, instrument, 0.07)
            tone = apply_envelope(tone, attack=0.05, decay=0.5)
            audio[start_idx:end_idx] += tone[:len(audio[start_idx:end_idx])]

    # Noise layer
    audio += generate_noise(duration, 0.02)
    audio = np.clip(audio, -1, 1)
    pan = np.random.uniform(-0.5, 0.5)
    stereo = apply_pan(audio, pan)
    return stereo

# =====================
# GUI
# =====================
class ProceduralMusicApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Looping Procedural Music")
        self.layout = QVBoxLayout()

        self.tempo_label = QLabel(f"Tempo: {TEMPO} BPM")
        self.tempo_slider = QSlider(Qt.Orientation.Horizontal)
        self.tempo_slider.setRange(30, 200)
        self.tempo_slider.setValue(TEMPO)
        self.tempo_slider.valueChanged.connect(self.update_tempo)
        self.layout.addWidget(self.tempo_label)
        self.layout.addWidget(self.tempo_slider)

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(SCALES.keys())
        self.scale_combo.setCurrentText(SCALE)
        self.layout.addWidget(QLabel("Scale"))
        self.layout.addWidget(self.scale_combo)

        self.inst_combo = QComboBox()
        self.inst_combo.addItems(INSTRUMENTS)
        self.layout.addWidget(QLabel("Instrument"))
        self.layout.addWidget(self.inst_combo)

        self.preview_btn = QPushButton("Start/Stop Live Preview")
        self.preview_btn.setCheckable(True)
        self.preview_btn.clicked.connect(self.toggle_live_preview)
        self.layout.addWidget(self.preview_btn)

        self.setLayout(self.layout)

        self.tempo = TEMPO
        self.audio_stream = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.stream_chunk)

    def update_tempo(self, value):
        self.tempo = value
        self.tempo_label.setText(f"Tempo: {value} BPM")

    def toggle_live_preview(self):
        if self.preview_btn.isChecked():
            self.audio_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=2, dtype='float32')
            self.audio_stream.start()
            self.timer.start(DURATION_CHUNK * 1000)  # Generate new chunk every DURATION_CHUNK seconds
            self.stream_chunk()  # Start immediately
        else:
            self.timer.stop()
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

    def stream_chunk(self):
        chunk = generate_procedural_chunk(DURATION_CHUNK, self.tempo,
                                          self.scale_combo.currentText(),
                                          self.inst_combo.currentText())
        if self.audio_stream:
            self.audio_stream.write(chunk)

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProceduralMusicApp()
    window.show()
    sys.exit(app.exec())
