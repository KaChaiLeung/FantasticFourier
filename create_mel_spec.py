import os
from mel_spec_class import MelSpec

WAV_FILES = os.listdir('bass_acoustic')

wav_dict = {}

for file in WAV_FILES:
    wav_dict[file] = MelSpec(f'{file}')
    wav_dict[file].create_mel_spec()
    wav_dict[file].plot_mel_spec()
