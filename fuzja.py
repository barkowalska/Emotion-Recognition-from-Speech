import torch
import numpy as np
from DBN import DBN
from data import load_data  # Ensure this function signature matches the required parameters
from RBF import RBF  # Correctly import the RBF class
from sklearn.model_selection import train_test_split
import librosa
from data import parse_filename_cremad, parse_filename_emodb
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


directory = 'C:\\Users\\jkowa\\Desktop\\glosowe\\data1'  
dataset_type = 'emodb'  
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

def compute_features(dbn, data):
    features = []
    for row in data:
        frame_features = []
        for frame in row:
            frame = torch.tensor(frame, dtype=torch.float32)
            feature = dbn.generate_input_for_layer(len(dbn.layer_parameters), frame)
            frame_features.append(feature.numpy())
        frame_features = np.array(frame_features)
        features.append(np.mean(frame_features, axis=0))
    return np.array(features)

X = []  
y = []  

for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        filepath = os.path.join(directory, filename)
        signal, sr = librosa.load(filepath, sr=None)
        frame = extract_frames(signal, sr)
        X.append(frame)
        
        if dataset_type == 'emodb':
            speaker_number, emotion = parse_filename_emodb(filename)
        elif dataset_type == 'cremad':
            speaker_number, emotion = parse_filename_cremad(filename)
        y.append(emotion)

if len(X) != len(y):
    raise ValueError("Mismatch between the number of feature sets and labels")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

dbn = DBN(400, (200, 10, 100), 1, 0.1) 
dbn.load_model('dbn_model.pth')

print("Loaded model parameters:")
for i, layer in enumerate(dbn.layer_parameters):
    print(f"Layer {i}: W: {layer['W'].shape}, bp: {layer['bp'].shape}, bn: {layer['bn'].shape}, stddev: {layer['stddev'].shape if layer['stddev'] is not None else 'None'}")

X = compute_features(dbn, X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

input_dim = int(X_train.shape[1])
output_dim = int(len(np.unique(y_train)))
num_centers = 200

def train_evaluate_rbf(X_train, y_train, X_val, y_val, num_centers, beta):
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    rbf = RBF(input_dim, num_centers, output_dim)
    rbf.beta = beta
    
    Y_train = np.eye(output_dim)[y_train]
    rbf.train(X_train, Y_train)
    
    y_val_pred = rbf.predict(X_val)
    y_val_pred_labels = np.argmax(y_val_pred, axis=1)
    
    val_accuracy = accuracy_score(y_val, y_val_pred_labels)
    return val_accuracy

a=train_evaluate_rbf(X_train, y_train, X_val, y_val, num_centers, beta=0.5)

print(f"Validation Accuracy of DBN with RBF: {a * 100:.2f}%")