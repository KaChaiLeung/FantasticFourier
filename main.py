import pyaudio as pa
import struct
import numpy as np
import matplotlib
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import time

matplotlib.use('TkAgg') # To open another window

# Constants
CHUNK = 1024 * 4 # samples per frame
FORMAT = pa.paInt16 # audio format
CHANNELS = 1 # single channel
RATE = 44100 # samples per second

# PyAudio instance
p = pa.PyAudio()

# Open stream
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# Initialize plot
fig, (ax1, ax2) = plt.subplots(2)
x = np.arange(0, 2 * CHUNK, 2)
x_fft = np.linspace(0, RATE, CHUNK)
line, = ax1.plot(x, np.zeros(CHUNK), 'r', lw=1)
line_fft, = ax2.plot(x_fft, np.zeros(CHUNK), 'g', lw=1)
ax1.set_ylim(-10000, 10000)
ax1.set_xlim(0, CHUNK)
ax2.set_ylim(0,10)
ax2.set_xlim(0, 1000)
fig.show()

# Flag to stop loop
running = True

# Function to handle key press
def on_key(event):
    global running
    if event.key == 'q': # q to exit
        print("Stopping program.")
        running = False

# Connect key press event
fig.canvas.mpl_connect('key_press_event', on_key)

frame_count = 0
start_time = time.time()

# Audio loop
try:
    while running:
        data = stream.read(CHUNK, exception_on_overflow=False)
        data_int = struct.unpack(str(CHUNK) + 'h', data)
        line.set_ydata(data_int)

        y_fft = fft(data_int)
        line_fft.set_ydata(np.abs(y_fft[0:CHUNK]) * 2 / (256 * CHUNK))

        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1

except Exception as e:
    print("Error:", e)

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print(frame_count / (time.time() - start_time))