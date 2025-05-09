import numpy as np
from scipy.fft import fft, fftfreq

# Need to input y-data, sample rate and duration of clip.
def ft(data, sample_rate, duration):

    samples = sample_rate * duration # Number of samples
    
    ftdata = fft(data) # Fourier transform on y-data
    ftdom = fftfreq(samples, 1 / sample_rate) # Converting to frequency domain; number of samples & duration of sample

    return ftdom, ftdata # Returning data to be plotted.