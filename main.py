import sys
import numpy as np
from scipy.io.wavfile import write
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QComboBox, QFileDialog
)
from PyQt6.QtCore import Qt
from pydub import AudioSegment
import sounddevice as sd  # For audio preview

# =====================
# SETTINGS
# =====================
SAMPLE_RATE = 44100
DURATION = 60
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
# SOUND UTILITIES
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
    return (wave * volume * 32767).astype(np.int16)

def apply_reverb(signal, decay=0.3, delay_time=0.03):
    delay_samples = int(SAMPLE_RATE * delay_time)
    reverb_signal = np.copy(signal)
    for i in range(delay_samples, len(signal)):
        reverb_signal[i] += int(signal[i - delay_samples] * decay)
    return np.clip(reverb_signal, -32768, 32767)

def apply_pan(signal, pan=0.0):
    left = signal * (1 - pan) / 2
    right = signal * (1 + pan) / 2
    stereo = np.stack([left, right], axis=1).astype(np.int16)
    return stereo

def generate_noise(duration, volume=0.05):
    n_samples = int(duration * SAMPLE_RATE)
    noise = np.random.normal(0, 1, n_samples) * volume * 32767
    return np.clip(noise, -32768, 32767).astype(np.int16)

def apply_envelope(signal, attack=0.1, decay=0.5):
    n_samples = len(signal)
    env = np.ones(n_samples)
    attack_samples = int(attack * n_samples)
    decay_samples = int(decay * n_samples)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        env[-decay_samples:] = np.linspace(1, 0, decay_samples)
    return (signal * env).astype(np.int16)

def generate_chord(root, scale_notes, chord_type='triad'):
    if chord_type == 'triad':
        return [root, root + scale_notes[2], root + scale_notes[4]]
    else:
        return [root]

# =====================
# PROCEDURAL MUSIC
# =====================
def generate_procedural_music(duration=DURATION, tempo=TEMPO, scale='minor', instrument='sine'):
    beats = int(duration / 60 * tempo)
    audio = np.zeros(int(SAMPLE_RATE * duration), dtype=np.int16)
    scale_notes = SCALES[scale]
    layers = []

    # Drone
    drone_audio = np.zeros_like(audio)
    for i in range(beats):
        root_note = 48 + np.random.choice(scale_notes)
        freq = midi_to_freq(root_note)
        start_idx = int(i * (SAMPLE_RATE * 60 / tempo))
        end_idx = start_idx + int(SAMPLE_RATE * 60 / tempo)
        tone = generate_tone(freq, 60 / tempo, instrument, volume=0.08)
        tone = apply_envelope(tone, attack=0.3, decay=0.7)
        drone_audio[start_idx:end_idx] += tone[:len(drone_audio[start_idx:end_idx])]
    drone_audio = apply_reverb(drone_audio, decay=0.3)
    layers.append(drone_audio)

    # Chord layer
    chord_audio = np.zeros_like(audio)
    for i in range(beats // 2):
        root_note = 60 + np.random.choice(scale_notes)
        chord_notes = generate_chord(root_note, scale_notes)
        start_idx = int(i * 2 * (SAMPLE_RATE * 60 / tempo))
        end_idx = start_idx + int(2 * (SAMPLE_RATE * 60 / tempo))
        for note in chord_notes:
            freq = midi_to_freq(note)
            tone = generate_tone(freq, 2 * 60 / tempo, instrument, volume=0.05)
            tone = apply_envelope(tone, attack=0.5, decay=0.5)
            chord_audio[start_idx:end_idx] += tone[:len(chord_audio[start_idx:end_idx])]
    chord_audio = apply_reverb(chord_audio, decay=0.25)
    layers.append(chord_audio)

    # Melody layer
    melody_audio = np.zeros_like(audio)
    for i in range(beats):
        if np.random.rand() < 0.2:
            note = 60 + np.random.choice(scale_notes)
            freq = midi_to_freq(note)
            start_idx = int(i * (SAMPLE_RATE * 60 / tempo))
            end_idx = start_idx + int((SAMPLE_RATE * 60 / tempo) / 2)
            tone = generate_tone(freq, 60 / tempo / 2, instrument, volume=0.07)
            tone = apply_envelope(tone, attack=0.05, decay=0.5)
            melody_audio[start_idx:end_idx] += tone[:len(melody_audio[start_idx:end_idx])]
    melody_audio = apply_reverb(melody_audio, decay=0.2)
    layers.append(melody_audio)

    # Noise layer
    noise_audio = generate_noise(duration, volume=0.02)
    layers.append(noise_audio)

    # Mix layers
    for layer in layers:
        audio += layer
    audio = np.clip(audio, -32768, 32767)
    pan = np.random.uniform(-0.5, 0.5)
    stereo_audio = apply_pan(audio, pan=pan)
    return stereo_audio

def save_audio(audio_data, filename='output.wav'):
    if audio_data.ndim == 2:
        write(filename, SAMPLE_RATE, audio_data)
    else:
        write(filename, SAMPLE_RATE, audio_data)
    AudioSegment.from_wav(filename).export(filename.replace('.wav', '.mp3'), format='mp3')

def play_audio(audio_data):
    """Preview audio using sounddevice"""
    # Convert to float32 between -1 and 1
    if audio_data.ndim == 2:
        audio_data_f = audio_data.astype(np.float32) / 32768
    else:
        audio_data_f = np.stack([audio_data, audio_data], axis=1).astype(np.float32) / 32768
    sd.play(audio_data_f, samplerate=SAMPLE_RATE)
    sd.wait()

# =====================
# GUI
# =====================
class ProceduralMusicApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cinematic Ambient Procedural Music Generator")
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

        self.duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.duration_slider.setRange(5, 300)
        self.duration_slider.setValue(DURATION)
        self.duration_slider.valueChanged.connect(self.update_duration)
        self.duration_label = QLabel(f"Duration: {DURATION} sec")
        self.layout.addWidget(self.duration_label)
        self.layout.addWidget(self.duration_slider)

        self.preview_btn = QPushButton("Preview Music")
        self.preview_btn.clicked.connect(self.preview_music)
        self.layout.addWidget(self.preview_btn)

        self.generate_btn = QPushButton("Generate & Save Music")
        self.generate_btn.clicked.connect(self.generate_music)
        self.layout.addWidget(self.generate_btn)

        self.setLayout(self.layout)
        self.tempo = TEMPO
        self.duration = DURATION
        self.last_audio = None

    def update_tempo(self, value):
        self.tempo = value
        self.tempo_label.setText(f"Tempo: {value} BPM")

    def update_duration(self, value):
        self.duration = value
        self.duration_label.setText(f"Duration: {value} sec")

    def preview_music(self):
        self.last_audio = generate_procedural_music(
            duration=self.duration,
            tempo=self.tempo,
            scale=self.scale_combo.currentText(),
            instrument=self.inst_combo.currentText()
        )
        play_audio(self.last_audio)

    def generate_music(self):
        if self.last_audio is None:
            self.last_audio = generate_procedural_music(
                duration=self.duration,
                tempo=self.tempo,
                scale=self.scale_combo.currentText(),
                instrument=self.inst_combo.currentText()
            )
        filename, _ = QFileDialog.getSaveFileName(self, "Save Music", "", "MP3 Files (*.mp3);;WAV Files (*.wav)")
        if filename:
            if not filename.endswith(".wav") and not filename.endswith(".mp3"):
                filename += ".wav"
            save_audio(self.last_audio, filename)
            print(f"Music saved to {filename}")

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProceduralMusicApp()
    window.show()
    sys.exit(app.exec())
