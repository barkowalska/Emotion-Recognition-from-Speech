from RBM import GB_RBM
import pandas as pd
import numpy as np
import os
import librosa
from DBN import DBN
from DBN_Signal import DBN_Signal


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#main d
def extract_frames(signal, sample_rate):
    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Hamming Window
    frames *= np.hamming(frame_length)
    return frames

def reduce_dimensionality(frames, n_components=2):
    # Apply PCA
    #print(frames.shape)
    pca = PCA(n_components=n_components)
    frames_reduced = pca.fit_transform(frame)

    #print(frames_reduced.shape)
    return frames_reduced

# Funkcja dopełniająca sygnał zerami do określonej długości
def pad_frames(signal, length):
    return np.pad(signal, (0, length - len(signal)), 'constant')


frames = []

directory = "glosowe\wav"
for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        filepath = os.path.join(directory, filename)
        signal, sr = librosa.load(filepath, sr=None)
        frame = extract_frames(signal, sr)
        #frame_reduced = reduce_dimensionality(frame, n_components=40)
        frames.append(frame)


X_train = np.concatenate(frames)

print(X_train.shape)

stddev = np.mean(np.std(X_train, axis=1))

model = DBN_Signal(X_train.shape[1], (200, 100, 10), 6, 1)
model.train(X_train, 100, 32, 0.01, 0.1)

statistics = model.process_signal(frames[0])
print(statistics)