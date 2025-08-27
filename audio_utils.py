import numpy as np
from scipy.signal import lfilter

SAMPLE_RATE = 44100

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
    return np.stack([left, right], axis=1)

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

def process_effects(chunk, reverb_amount=0.3, delay_amount=0.3, lowpass_cutoff=15000, highpass_cutoff=20):
    if reverb_amount > 0:
        chunk[:,0] = apply_reverb(chunk[:,0], decay=reverb_amount)
        chunk[:,1] = apply_reverb(chunk[:,1], decay=reverb_amount)
    if delay_amount > 0:
        chunk[:,0] = apply_delay(chunk[:,0], feedback=delay_amount)
        chunk[:,1] = apply_delay(chunk[:,1], feedback=delay_amount)
    if lowpass_cutoff > 20:
        chunk = apply_filter(chunk, 'low', lowpass_cutoff)
    if highpass_cutoff > 20:
        chunk = apply_filter(chunk, 'high', highpass_cutoff)
    return np.clip(chunk, -1, 1)
