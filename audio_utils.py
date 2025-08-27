import numpy as np
from scipy.signal import lfilter

SAMPLE_RATE = 44100

def midi_to_freq(midi_note):
    return 440 * 2 ** ((midi_note - 69) / 12)

def generate_tone(frequency, duration, instrument='sine', volume=0.2):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    if instrument == 'sine':
        wave = np.sin(2 * np.pi * frequency * t)
    elif instrument == 'square':
        wave = np.sign(np.sin(2 * np.pi * frequency * t))
    elif instrument == 'triangle':
        wave = 2 * np.arcsin(np.sin(2 * np.pi * frequency * t)) / np.pi
    elif instrument == 'sawtooth':
        wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
    elif instrument == 'fm_sine':
        mod_freq = frequency * 2
        mod_index = 2.0
        wave = np.sin(2*np.pi*frequency*t + mod_index*np.sin(2*np.pi*mod_freq*t))
    elif instrument == 'noise_pad':
        wave = np.random.normal(0, 1, len(t)).astype(np.float32)
        wave = apply_envelope(wave, attack=0.5, decay=0.7)
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

def apply_chorus(signal, depth=0.003, rate=0.25):
    n_samples = len(signal)
    output = np.copy(signal)
    delay_samples = int(depth * SAMPLE_RATE)
    for i in range(delay_samples, n_samples):
        mod = int(delay_samples * np.sin(2*np.pi*rate*i/SAMPLE_RATE))
        output[i] += 0.5 * signal[i-mod]
    return np.clip(output, -1, 1)

def apply_phaser(signal, rate=0.2, depth=0.02):
    n_samples = len(signal)
    output = np.copy(signal)
    for i in range(n_samples):
        shift = int(depth * SAMPLE_RATE * np.sin(2*np.pi*rate*i/SAMPLE_RATE))
        if i-shift >=0:
            output[i] += signal[i-shift]
    return np.clip(output, -1, 1)

def apply_stereo_widen(signal, amount=0.3):
    left = signal[:,0]
    right = signal[:,1]
    mid = (left+right)/2
    side = (left-right)/2*(1+amount)
    new_left = mid + side
    new_right = mid - side
    return np.stack([new_left,new_right],axis=1)

def apply_filter(signal, filter_type='low', cutoff=1000):
    omega = 2 * np.pi * cutoff / SAMPLE_RATE
    if filter_type == 'low':
        b = [omega / (omega + 1)]
        a = [1, (omega - 1) / (omega + 1)]
    else:
        b = [1 / (omega + 1), -1 / (omega + 1)]
        a = [1, (omega - 1) / (omega + 1)]
    return lfilter(b,a,signal,axis=0)

def process_effects(chunk, reverb_amount=0.3, delay_amount=0.3, lowpass_cutoff=15000, highpass_cutoff=20,
                    chorus_amount=0.0, phaser_amount=0.0, stereo_widen=0.0):
    if reverb_amount>0:
        chunk[:,0]=apply_reverb(chunk[:,0],decay=reverb_amount)
        chunk[:,1]=apply_reverb(chunk[:,1],decay=reverb_amount)
    if delay_amount>0:
        chunk[:,0]=apply_delay(chunk[:,0],feedback=delay_amount)
        chunk[:,1]=apply_delay(chunk[:,1],feedback=delay_amount)
    if chorus_amount>0:
        chunk[:,0]=apply_chorus(chunk[:,0],depth=0.003,rate=0.25*chorus_amount)
        chunk[:,1]=apply_chorus(chunk[:,1],depth=0.003,rate=0.25*chorus_amount)
    if phaser_amount>0:
        chunk[:,0]=apply_phaser(chunk[:,0],rate=0.2*phaser_amount,depth=0.02*phaser_amount)
        chunk[:,1]=apply_phaser(chunk[:,1],rate=0.2*phaser_amount,depth=0.02*phaser_amount)
    if stereo_widen>0:
        chunk=apply_stereo_widen(chunk,stereo_widen)
    if lowpass_cutoff>20:
        chunk=apply_filter(chunk,'low',lowpass_cutoff)
    if highpass_cutoff>20:
        chunk=apply_filter(chunk,'high',highpass_cutoff)
    return np.clip(chunk,-1,1)
