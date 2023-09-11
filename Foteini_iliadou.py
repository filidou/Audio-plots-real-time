import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import librosa
import librosa.display

# Enable interactive plots
plt.ion()

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    print(f"Device index: {i}, Device name: {device_info['name']}")
    d = p.get_device_info_by_index(i)

mic_device_index = 1
WINDOW_SIZE = 2048
CHANNELS = 1
RATE = 44100
FFT_FRAMES_IN_SPEC = 20
global_blocks = np.zeros((FFT_FRAMES_IN_SPEC, WINDOW_SIZE))
fft_frame = np.array(WINDOW_SIZE // 2)
win = np.hamming(WINDOW_SIZE)
spec_img = np.zeros((WINDOW_SIZE // 2, FFT_FRAMES_IN_SPEC))
user_terminated = False
s = np.zeros(WINDOW_SIZE * FFT_FRAMES_IN_SPEC)  # Initialize s outside the loop

def callback(in_data, frame_count, time_info, status):
    global global_blocks, fft_frame, win, spec_img, s
    
    numpy_block_from_bytes = np.frombuffer(in_data, dtype='int16')
    block_for_speakers = np.zeros((numpy_block_from_bytes.size, CHANNELS), dtype='int16')
    block_for_speakers[:, 0] = numpy_block_from_bytes

    if len(win) == len(numpy_block_from_bytes):
        frame_fft = np.fft.fft(win * numpy_block_from_bytes)
        p = np.abs(frame_fft) * 2 / np.sum(win)
        fft_frame = 20 * np.log10(p[:WINDOW_SIZE // 2] / 32678)
        spec_img = np.roll(spec_img, -1, axis=1)
        spec_img[:, -1] = fft_frame[::-1]
        global_blocks = np.roll(global_blocks, -1, axis=0)
        global_blocks[-1, :] = numpy_block_from_bytes
        
        s = np.reshape(global_blocks, WINDOW_SIZE * FFT_FRAMES_IN_SPEC)
    
    return (block_for_speakers, pyaudio.paContinue)

def user_input_function():
    k = input('press "s" to terminate (then press "Enter"): ')
    print('pressed:', k)
    if k == 's' or k == 'S':
        global user_terminated
        user_terminated = True
        print('user_terminated:', user_terminated)

output = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=mic_device_index,
    frames_per_buffer=WINDOW_SIZE,
    stream_callback=callback,
    start=False
)

output.start_stream()

threaded_input = Thread(target=user_input_function)
threaded_input.start()

while output.is_active() and not user_terminated:
    plt.clf()
    
    plt.subplot(2, 1, 2)
    mfccs = librosa.feature.mfcc(y=s, sr=RATE, n_mfcc=12, n_fft=WINDOW_SIZE, hop_length=WINDOW_SIZE)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.title('MFCCs')
    plt.draw()
    plt.pause(0.01)

print('stopping audio')
output.stop_stream()