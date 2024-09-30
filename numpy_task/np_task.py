import numpy as np
from scipy.io import wavfile

class SoundWaveFactory:
    SAMPLING_RATE = 44100
    MAX_AMPLITUDE = 2 ** 13
    NOTES = {
    '0': 0, 'e0': 20.60172, 'f0': 21.82676, 'f#0': 23.12465, 'g0': 24.49971, 'g#0': 25.95654, 'a0': 27.50000, 'a#0': 29.13524,
    'b0': 30.86771, 'c0': 32.70320, 'c#0': 34.64783, 'd0': 36.70810, 'd#0': 38.89087,
    'e1': 41.20344, 'f1': 43.65353, 'f#1': 46.24930, 'g1': 48.99943, 'g#1': 51.91309, 'a1': 55.00000, 'a#1': 58.27047,
    'b1': 61.73541, 'c1': 65.40639, 'c#1': 69.29566, 'd1': 73.41619, 'd#1': 77.78175,
    'e2': 82.40689, 'f2': 87.30706, 'f#2': 92.49861, 'g2': 97.99886, 'g#2': 103.8262, 'a2': 110.0000, 'a#2': 116.5409,
    'b2': 123.4708, 'c2': 130.8128, 'c#2': 138.5913, 'd2': 146.8324, 'd#2': 155.5635,
    'e3': 164.8138, 'f3': 174.6141, 'f#3': 184.9972, 'g3': 195.9977, 'g#3': 207.6523, 'a3': 220.0000, 'a#3': 233.0819,
    'b3': 246.9417, 'c3': 261.6256, 'c#3': 277.1826, 'd3': 293.6648, 'd#3': 311.1270,
    'e4': 329.6276, 'f4': 349.2282, 'f#4': 369.9944, 'g4': 391.9954, 'g#4': 415.3047, 'a4': 440.0000, 'a#4': 466.1638,
    'b4': 493.8833, 'c4': 523.2511, 'c#4': 554.3653, 'd4': 587.3295, 'd#4': 622.2540,
    'e5': 659.2551, 'f5': 698.4565, 'f#5': 739.9888, 'g5': 783.9909, 'g#5': 830.6094, 'a5': 880.0000, 'a#5': 932.3275,
    'b5': 987.7666, 'c5': 1046.502, 'c#5': 1108.731, 'd5': 1174.659, 'd#5': 1244.508,
    'e6': 1318.510, 'f6': 1396.913, 'f#6': 1479.978, 'g6': 1567.982, 'g#6': 1661.219, 'a6': 1760.000, 'a#6': 1864.655,
    'b6': 1975.533, 'c6': 2093.005, 'c#6': 2217.461, 'd6': 2349.318, 'd#6': 2489.016,
    'e7': 2637.020, 'f7': 2793.826, 'f#7': 2959.955, 'g7': 3135.963, 'g#7': 3322.438, 'a7': 3520.000, 'a#7': 3729.310,
    'b7': 3951.066, 'c7': 4186.009, 'c#7': 4434.922, 'd7': 4698.636, 'd#7': 4978.032,
}
    
    def __init__(self, duration_seconds: float = 5):
        self.duration_seconds = duration_seconds
        self.sound_array_len = int(self.SAMPLING_RATE * self.duration_seconds)
        self.common_timeline = np.linspace(0, self.duration_seconds, num=self.sound_array_len)

    def get_normed_sin(self, frequency: float):
        return self.MAX_AMPLITUDE * np.sin(2 * np.pi * frequency * self.common_timeline)
    
    def get_soundwave(self, note: str):
        return self.get_normed_sin(self.NOTES[note])
    
    def create_note(self, note: str="a4"):
        sound_wave = self.get_soundwave(note).astype(np.int16)
        return sound_wave
    
    #Method to read a wave from .txt file
    def read_wave_from_txt(self, text_file: str):
        return np.loadtxt(text_file)
    
    #Method to print any details you think will be important about the wave
    def wave_details(self, wave: np.ndarray):
        print(f"Wave shape: {wave.shape[0]}")
        print(f"Wave dtype: {wave.dtype}")
        print(f"Min value: {np.min(wave)}")
        print(f"Max value: {np.max(wave)}")
        print(f"Standard deviation: {np.std(wave):.2f}")

    #Method to normalize_sound_waves several waves: in both length (to the shortest file) and amplitude (according to the amplitude attribute)
    def normalize_sound_waves(self, waves:list):
        min_length = min(wave.shape[0] for wave in waves)
        normalized_waves = [wave[:min_length] for wave in waves]
        max_amplitude = max(np.max(np.abs(wave)) for wave in normalized_waves)
        return [wave * (self.MAX_AMPLITUDE / max_amplitude) for wave in normalized_waves]

    #Method to save wave into np.array txt by default and into WAV file if parameter "type='WAV'" is provided
    def save_wave(self, sound_wave: np.ndarray, file_name: str, type: str = 'TXT'):
        if type=='WAV':
            wavfile.write(file_name, self.SAMPLING_RATE, sound_wave)
        else:
            np.savetxt(file_name, sound_wave)

    #Extra: a method to switch sin waves into triangular waves;
    def get_triangle_wave(self, frequency: float):
        t = self.common_timeline * frequency - 0.25
        return (2 * self.MAX_AMPLITUDE / np.pi) * np.arcsin(np.sin(2 * np.pi * t))
    
    
    #Extra: a method to switch sin waves into square waves;
    def get_square_wave(self, frequency: float):
        return self.MAX_AMPLITUDE * np.sign(np.sin(2 * np.pi * frequency * self.common_timeline))
