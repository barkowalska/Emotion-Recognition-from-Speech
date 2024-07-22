from RBM import GB_RBM
import pandas as pd
import numpy as np
import os
import librosa
from DBN import DBN
import torch
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_frames(signal, sample_rate):
   
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

   
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
  
    frames *= np.hamming(frame_length)
    return frames

def reduce_dimensionality(frames, n_components=2):
    
    pca = PCA(n_components=n_components)
    frames_reduced = pca.fit_transform(frame)

    return frames_reduced

def pad_frames(signal, length):
    return np.pad(signal, (0, length - len(signal)), 'constant')


frames = []

directory = "C:\\Users\\jkowa\\Desktop\\glosowe\\data1"
for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        filepath = os.path.join(directory, filename)
        signal, sr = librosa.load(filepath, sr=None)
        frame = extract_frames(signal, sr)
        frames.append(frame)


X = np.concatenate(frames)

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

print(X_train.shape)

stddev = np.mean(np.std(X_train, axis=1))

print(X_train.shape[1])
model = DBN(X_train.shape[1], (200, 150, 100), 5, 1)

model.train(X_train, 100, 32, 0.01, 0.1)

print("Model parameters before saving:")
for i, layer in enumerate(model.layer_parameters):
    print(f"Layer {i}: W: {layer['W'].shape}, bp: {layer['bp'].shape}, bn: {layer['bn'].shape}, stddev: {layer['stddev'].shape if layer['stddev'] is not None else 'None'}")



current_directory = os.getcwd()
model_file_path = os.path.join(current_directory, "dbn_model.pth")
model.save_model(model_file_path)

loaded_model = DBN(X_train.shape[1], (200, 100, 10), 6, 1)
loaded_model.load_model(model_file_path)


print("Loaded model parameters:")
for i, layer in enumerate(loaded_model.layer_parameters):
    print(f"Layer {i}: W: {layer['W'].shape}, bp: {layer['bp'].shape}, bn: {layer['bn'].shape}, stddev: {layer['stddev'].shape if layer['stddev'] is not None else 'None'}")

test_frame = torch.tensor(X_test[0], dtype=torch.float32)


hidden_activations = loaded_model.generate_input_for_layer(len(loaded_model.layer_parameters), test_frame)

print("Hidden Layer Activations from loaded model: ", hidden_activations)


evaluation_error_before_saving = model.evaluate(torch.tensor(X_test, dtype=torch.float32))
print(f"Evaluation error before saving: {evaluation_error_before_saving}")

