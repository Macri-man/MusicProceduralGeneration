import numpy as np

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
            return (2/np.pi) * np.arcsin(np.sin(self.phase)) * self.depth
        elif self.waveform == 'square':
            return np.sign(np.sin(self.phase)) * self.depth
        elif self.waveform == 'sawtooth':
            return ((self.phase % (2*np.pi)) / np.pi -1) * self.depth
        return 0

class LayerLFO:
    def __init__(self, freq_vol=0.05, depth_vol=0.2, freq_pan=0.03, depth_pan=0.3):
        self.lfo_vol = LFO(freq=freq_vol, depth=depth_vol)
        self.lfo_pan = LFO(freq=freq_pan, depth=depth_pan)

    def step(self, dt):
        return self.lfo_vol.step(dt), self.lfo_pan.step(dt)
