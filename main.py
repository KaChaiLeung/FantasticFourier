import pyaudio as pa
import struct
import numpy as np
import matplotlib.pyplot as plt

# constants
CHUNK = 1024 * 2
FORMAT = pa.paInt16
CHANNELS = 1
RATE = 44100

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
fig, ax = plt.subplots()
x = np.arange(0, 2 * CHUNK, 2)
line, = ax.plot(x, np.zeros(CHUNK), 'r')
ax.set_ylim(-10000, 10000)
ax.set_xlim(0, CHUNK)
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

# Audio loop
try:
    while running:
        data = stream.read(CHUNK, exception_on_overflow=False)
        data_int = struct.unpack(str(CHUNK) + 'h', data)
        line.set_ydata(data_int)
        fig.canvas.draw()
        fig.canvas.flush_events()

except Exception as e:
    print("Error:", e)

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()