import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


FFT_SIZE = 4096
HOP_LENGTH = 256
WIN_LENGTH = 2048


class Spectrogram:


    def __init__(self, file_path):
        self.fft_size = FFT_SIZE
        self.hop_length = HOP_LENGTH
        self.win_length = WIN_LENGTH

        self.file_name = file_path.stem
        self.file_path = file_path
        self.signal, self.sample_rate = librosa.load(file_path, sr=22050)

        self.make_spec()
    

    def make_spec(self):
        self.stft = librosa.stft(self.signal,
                                 n_fft=self.fft_size,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 center=False)
        self.spectrogram = np.abs(self.stft)
        self.spectrogram_dB = librosa.amplitude_to_db(self.spectrogram, ref=np.max)
    
        self.img = librosa.display.specshow(self.spectrogram_dB,
                                            y_axis='mel',
                                            x_axis='time',
                                            sr=self.sample_rate,
                                            hop_length=self.hop_length,
                                            win_length=self.win_length,
                                            cmap='inferno')


    def plot_spec(self):
        plt.colorbar(self.img, format='%+2.f dB')
        plt.title(self.file_name)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.show()
    

    def save_spec(self, save_dir):
        plt.savefig(f'{save_dir}/{self.file_path.stem.split('_')[0]}/{self.file_name}.png', 
                    dpi=300, 
                    format='png',
                    bbox_inches='tight', 
                    pad_inches=0.1, 
                    facecolor='white')
        plt.close()

if __name__ == '__main__':
    path_1 = Path('data/train_audio/guitar/guitar_synthetic_012-120-075.wav')
    path_2 = Path('data/train_images/guitar/')
