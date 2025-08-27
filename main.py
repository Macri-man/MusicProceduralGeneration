import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QComboBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from scipy.signal import lfilter

# =====================
# SETTINGS
# =====================
SAMPLE_RATE = 44100
DURATION_CHUNK = 5  # seconds per chunk
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
# AUDIO UTILS
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
    return np.random.normal(0, 1, n_samples).astype(np.float32) * volume

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
# EFFECTS
# =====================
def apply_reverb(signal, decay=0.3, delay_time=0.03):
    delay_samples = int(SAMPLE_RATE * delay_time)
    output = np.copy(signal)
    for i in range(delay_samples, len(signal)):
        output[i] += signal[i - delay_samples] * decay
    return np.clip(output, -1, 1)

def apply_delay(signal, delay_time=0.2, feedback=0.3):
    delay_samples = int(SAMPLE_RATE * delay_time)
    output = np.copy(signal)
    for i in range(delay_samples, len(signal)):
        output[i] += output[i - delay_samples] * feedback
    return np.clip(output, -1, 1)

def apply_filter(signal, filter_type='low', cutoff=1000):
    omega = 2 * np.pi * cutoff / SAMPLE_RATE
    if filter_type == 'low':
        b = [omega / (omega + 1)]
        a = [1, (omega - 1) / (omega + 1)]
    else:
        b = [1 / (omega + 1), -1 / (omega + 1)]
        a = [1, (omega - 1) / (omega + 1)]
    return lfilter(b, a, signal, axis=0)

def process_effects(chunk, reverb_amount, delay_amount, lowpass_cutoff, highpass_cutoff):
    if reverb_amount > 0:
        chunk[:, 0] = apply_reverb(chunk[:, 0], decay=reverb_amount)
        chunk[:, 1] = apply_reverb(chunk[:, 1], decay=reverb_amount)
    if delay_amount > 0:
        chunk[:, 0] = apply_delay(chunk[:, 0], feedback=delay_amount)
        chunk[:, 1] = apply_delay(chunk[:, 1], feedback=delay_amount)
    if lowpass_cutoff > 20:
        chunk = apply_filter(chunk, 'low', lowpass_cutoff)
    if highpass_cutoff > 20:
        chunk = apply_filter(chunk, 'high', highpass_cutoff)
    return np.clip(chunk, -1, 1)

# =====================
# LFOs
# =====================
class LFO:
    def __init__(self, freq=0.05, depth=0.2, waveform='sine'):
        self.freq = freq
        self.depth = depth
        self.waveform = waveform
        self.phase = 0.0
    def step(self, dt):
        self.phase += 2 * np.pi * self.freq * dt
        if self.waveform == 'sine':
            return np.sin(self.phase) * self.depth
        elif self.waveform == 'triangle':
            return (2 / np.pi) * np.arcsin(np.sin(self.phase)) * self.depth
        elif self.waveform == 'square':
            return np.sign(np.sin(self.phase)) * self.depth
        elif self.waveform == 'sawtooth':
            return ((self.phase % (2*np.pi)) / np.pi - 1) * self.depth
        else:
            return 0

# =====================
# LAYER LFO
# =====================
class LayerLFO:
    def __init__(self, freq_vol=0.05, depth_vol=0.2, freq_pan=0.03, depth_pan=0.3):
        self.lfo_vol = LFO(freq=freq_vol, depth=depth_vol)
        self.lfo_pan = LFO(freq=freq_pan, depth=depth_pan)
    def step(self, dt):
        return self.lfo_vol.step(dt), self.lfo_pan.step(dt)

# =====================
# GENERATE CHUNK
# =====================
def generate_procedural_chunk(duration, tempo, scale, instrument):
    beats = int(duration / 60 * tempo)
    audio = np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)
    scale_notes = SCALES[scale]

    # Drone
    for i in range(beats):
        if np.random.rand() < 0.8:
            root = 48 + np.random.choice(scale_notes)
            freq = midi_to_freq(root)
            start_idx = int(i * (SAMPLE_RATE * 60 / tempo))
            end_idx = start_idx + int(SAMPLE_RATE * 60 / tempo)
            tone = generate_tone(freq, 60 / tempo, instrument, volume=0.08)
            tone = apply_envelope(tone, attack=0.3, decay=0.7)
            audio[start_idx:end_idx] += tone[:len(audio[start_idx:end_idx])]

    # Chords
    for i in range(beats // 2):
        if np.random.rand() < 0.7:
            root = 60 + np.random.choice(scale_notes)
            chord = [root, root + scale_notes[2], root + scale_notes[4]]
            start_idx = int(i * 2 * (SAMPLE_RATE * 60 / tempo))
            end_idx = start_idx + int(2 * (SAMPLE_RATE * 60 / tempo))
            for note in chord:
                freq = midi_to_freq(note)
                tone = generate_tone(freq, 2 * 60 / tempo, instrument, volume=0.05)
                tone = apply_envelope(tone, 0.5, 0.5)
                audio[start_idx:end_idx] += tone[:len(audio[start_idx:end_idx])]

    # Melody
    for i in range(beats):
        if np.random.rand() < 0.3:
            note = 60 + np.random.choice(scale_notes)
            freq = midi_to_freq(note)
            start_idx = int(i * (SAMPLE_RATE * 60 / tempo))
            dur_note = (60 / tempo) * np.random.choice([0.5,1,1.5])
            end_idx = start_idx + int(dur_note * SAMPLE_RATE)
            tone = generate_tone(freq, dur_note, instrument, 0.07)
            tone = apply_envelope(tone, 0.05, 0.5)
            audio[start_idx:end_idx] += tone[:len(audio[start_idx:end_idx])]

    # Noise
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
        self.setWindowTitle("Advanced Procedural Ambient DAW")
        self.layout = QVBoxLayout()

        # Controls
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

        # Preview & Record
        self.preview_btn = QPushButton("Start/Stop Live Preview")
        self.preview_btn.setCheckable(True)
        self.preview_btn.clicked.connect(self.toggle_live_preview)
        self.layout.addWidget(self.preview_btn)

        self.record_btn = QPushButton("Start/Stop Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.layout.addWidget(self.record_btn)

        self.setLayout(self.layout)

        # State
        self.tempo = TEMPO
        self.audio_stream = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.stream_chunk)
        self.recording_buffer = []
        self.time_accumulator = 0.0

        # Layer LFOs
        self.layer_lfos = [LayerLFO() for _ in range(4)]
        self.lfo_reverb = LFO(0.02, 0.3)
        self.lfo_delay = LFO(0.01, 0.2)

    def update_tempo(self, value):
        self.tempo = value
        self.tempo_label.setText(f"Tempo: {value} BPM")

    def toggle_live_preview(self):
        if self.preview_btn.isChecked():
            self.audio_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=2, dtype='float32')
            self.audio_stream.start()
            self.timer.start(DURATION_CHUNK * 1000)
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
                    self, "Save Recording", "", "MP3 Files (*.mp3);;WAV Files (*.wav)")
                if filename:
                    if not filename.endswith(".wav") and not filename.endswith(".mp3"):
                        filename += ".wav"
                    write(filename, SAMPLE_RATE, (full_audio * 32767).astype(np.int16))
                    AudioSegment.from_wav(filename).export(filename.replace('.wav','.mp3'), format='mp3')
                    print(f"Recording saved to {filename}")
                self.recording_buffer = []

    def stream_chunk(self):
        dt = DURATION_CHUNK
        self.time_accumulator += dt
        chunk = generate_procedural_chunk(DURATION_CHUNK, self.tempo,
                                          self.scale_combo.currentText(),
                                          self.inst_combo.currentText())

        # Apply layer LFOs
        for lfo in self.layer_lfos:
            vol_mod, pan_mod = lfo.step(dt)
            mono = np.mean(chunk, axis=1) * (1 + vol_mod)
            chunk = apply_pan(mono, pan_mod)

        # Apply global effect LFOs
        chunk = process_effects(
            chunk,
            reverb_amount=(self.reverb_slider.value()/100) * (1 + self.lfo_reverb.step(dt)),
            delay_amount=(self.delay_slider.value()/100) * (1 + self.lfo_delay.step(dt)),
            lowpass_cutoff=self.lowpass_slider.value(),
            highpass_cutoff=self.highpass_slider.value()
        )

        if self.audio_stream:
            self.audio_stream.write(chunk)
        if self.record_btn.isChecked():
            self.recording_buffer.append(chunk)

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProceduralMusicApp()
    window.show()
    sys.exit(app.exec())
