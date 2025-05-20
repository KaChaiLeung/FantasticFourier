import librosa
import matplotlib.pyplot as plt
import numpy as np

class MelSpec:
    def __init__(self, file_path: str):
        self.file = file_path

        self.time_amp, self.sample_rate = librosa.load(f'bass_acoustic/{self.file}', sr=None)
        self.freq_amp = librosa.stft(self.time_amp, n_fft=2048, hop_length=256)
    

    def create_mel_spec(self):
        self.mel_spectrogram = librosa.feature.melspectrogram(S=np.abs(self.freq_amp), sr=self.sample_rate, n_mels=128)
        self.mel_spectrogram_dB = librosa.power_to_db(self.mel_spectrogram, ref=np.min)
    

    def plot_mel_spec(self):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.mel_spectrogram_dB, sr=self.sample_rate, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(self.file)
        plt.tight_layout()
        plt.show()