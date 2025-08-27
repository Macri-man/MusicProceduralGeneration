import numpy as np
from audio_utils import generate_tone, generate_noise, apply_envelope, apply_pan, midi_to_freq

SCALES = {
    'major':[0,2,4,5,7,9,11,12],
    'minor':[0,2,3,5,7,8,10,12],
    'pentatonic':[0,2,4,7,9,12],
    'chromatic':list(range(13))
}

def generate_procedural_chunk(duration, tempo, scale='minor', instrument='sine'):
    beats = int(duration/60*tempo)
    audio = np.zeros(int(duration*44100), dtype=np.float32)
    scale_notes = SCALES[scale]

    # Drone layer
    for i in range(beats):
        if np.random.rand()<0.8:
            root = 48 + np.random.choice(scale_notes)
            freq = midi_to_freq(root)
            start_idx = int(i*44100*60/tempo)
            end_idx = start_idx + int(44100*60/tempo)
            tone = generate_tone(freq, 60/tempo, instrument, 0.08)
            tone = apply_envelope(tone,0.3,0.7)
            audio[start_idx:end_idx] += tone[:len(audio[start_idx:end_idx])]

    # Chord layer
    for i in range(beats//2):
        if np.random.rand()<0.7:
            root = 60+np.random.choice(scale_notes)
            chord = [root, root+scale_notes[2], root+scale_notes[4]]
            start_idx = int(i*2*44100*60/tempo)
            end_idx = start_idx + int(2*44100*60/tempo)
            for note in chord:
                freq = midi_to_freq(note)
                tone = generate_tone(freq,2*60/tempo, instrument,0.05)
                tone = apply_envelope(tone,0.5,0.5)
                audio[start_idx:end_idx] += tone[:len(audio[start_idx:end_idx])]

    # Melody layer
    for i in range(beats):
        if np.random.rand()<0.3:
            note = 60+np.random.choice(scale_notes)
            freq = midi_to_freq(note)
            start_idx = int(i*44100*60/tempo)
            dur_note = 60/tempo*np.random.choice([0.5,1,1.5])
            end_idx = start_idx + int(dur_note*44100)
            tone = generate_tone(freq,dur_note,instrument,0.07)
            tone = apply_envelope(tone,0.05,0.5)
            audio[start_idx:end_idx] += tone[:len(audio[start_idx:end_idx])]

    # Noise layer
    audio += generate_noise(duration,0.02)
    audio = np.clip(audio,-1,1)
    pan = np.random.uniform(-0.5,0.5)
    return apply_pan(audio, pan)
