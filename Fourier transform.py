import numpy as np
import matplotlib.pyplot as plt
import pyaudio as pa
import struct

# Defining constants
CHUNK = 1024*2
FORMAT = pa.paInt16
CHANNELS = 1
RATE = 44100 # Hz

# Preparing for audio input
p = pa.PyAudio()

stream = p.open(
    format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    output = True,
    frames_per_buffer = CHUNK
)

# Reading data from mic
data = stream.read(CHUNK)
