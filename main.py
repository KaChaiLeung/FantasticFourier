import pyaudio as pa
import struct
import numpy as np
import matplotlib.pyplot as plt

# use this backend to display in separate Tk window
%matplotlib tk

# constants
CHUNK = 1024 * 2             # samples per frame
FORMAT = pa.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second

# create matplotlib figure and axes
fig, ax = plt.subplots(1, figsize=(15, 7))

# pyaudio class instance
p = pa.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# Initialising plots
fig, ax = plt.subplots()
x = np.arange(0, 2*CHUNK, 2)
line, = ax.plot(x, np.random.rand(CHUNK), 'r')
ax.set_ylim(-10000, 10000) 
ax.set_xlim(0, CHUNK)
fig.show()


# Loop to take in data from mic
while True:
    data = stream.read(CHUNK)
    data_int = struct.unpack(str(CHUNK) + 'h', data)
    line.set_ydata(data_int)
    fig.canvas.draw()
    fig.canvas.flush_events()