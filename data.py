import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from features import extract_features
from sklearn.preprocessing import MinMaxScaler

# Define emotion mapping based on the naming convention for EmoDB
emotion_mapping_emodb = {
    'W': 'anger',
    'L': 'boredom',
    'A': 'anxiety',
    'F': 'happiness',
    'T': 'sadness',
    'E': 'disgust',
    'N': 'neutral'
}



# Define emotion mapping for RAVDESS
emotion_mapping_ravdess = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def parse_filename_emodb(filename):
    speaker_number = filename[:2]
    emotion_code = filename[5]
    emotion = emotion_mapping_emodb.get(emotion_code, 'unknown')
    return speaker_number, emotion



def parse_filename_ravdess(filename):
    parts = filename.split('-')
    modality = parts[0]
    vocal_channel = parts[1]
    emotion_code = parts[2]
    emotion = emotion_mapping_ravdess.get(emotion_code, 'unknown')
    return modality, vocal_channel, emotion

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

def pad_signals(signals):
    if not signals:
        raise ValueError("No signals to pad. Ensure the directory contains .wav files.")
    max_length = max(len(signal) for signal in signals)
    padded_signals = np.array([np.pad(signal, (0, max_length - len(signal)), 'constant') for signal in signals])
    return padded_signals

def load_signals(directory):
    signals = []
    print(f"Loading signals from directory: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            print(f"Loading file: {filepath}")
            signal, sr = librosa.load(filepath, sr=None)
            signals.append(signal)
    if not signals:
        print("No .wav files found in the directory.")
    else:
        print(f"Loaded {len(signals)} files.")
    signals = pad_signals(signals)
    return signals

def load_labels(directories, dataset_type):
    labels = []
    for filename in os.listdir(directories):
        if filename.endswith('.wav'):
            filepath = os.path.join(directories, filename)
            if dataset_type == 'emodb':
                speaker_number, emotion = parse_filename_emodb(filename)
            elif dataset_type == 'ravdess':
                modality, vocal_channel, emotion = parse_filename_ravdess(filename)
            labels.append(emotion)
    labels = np.array(labels)
    return labels

def load_data(directories, dataset_type):
    features = []
    labels = []
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                signal, sr = librosa.load(filepath, sr=None)
                mfcc_features = extract_features(signal, sr)
                features.append(mfcc_features)
                if dataset_type == 'emodb':
                    speaker_number, emotion = parse_filename_emodb(filename)
                elif dataset_type == 'ravdess':
                    modality, vocal_channel, emotion = parse_filename_ravdess(filename)
                labels.append(emotion)
    if not features:
        raise ValueError("No valid audio files found")
    labels = np.array(labels)
    return features, labels

def preprocess_emodb_data(emodb_dir):
    features, labels = load_data([emodb_dir], 'emodb')
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(features, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


def preprocess_ravdess_data(ravdess_dir):
    features, labels = load_data([ravdess_dir], 'ravdess')
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(features, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


'''
directory = "C:/Users/jkowa/Desktop/glosowe/speach"

X_train, X_val, X_test, y_train, y_val, y_test, label_encoder=preprocess_ravdess_data(directory)

    


X_train = np.array(X_train)
#X_temp = np.array(X_temp)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
#y_temp = np.array(y_temp)
y_val = np.array(y_val)
y_test = np.array(y_test)

def print_dataset_description(X_train, y_train, X_val, y_val, X_test, y_test):
    print("### Description of data division no. 1\n")
    print(f"The training set consists of {X_train.shape[0]} samples, each with {X_train.shape[1]} features, and their corresponding {y_train.shape[0]} labels.")
    print(f"The validation set contains {X_val.shape[0]} samples, also with {X_val.shape[1]} features each, and their corresponding {y_val.shape[0]} labels.")
    print(f"The test set consists of {X_test.shape[0]} samples, each with {X_test.shape[1]} features, and {y_test.shape[0]} labels.")
    print("The training set is the largest, which is typical as it is used for training the model, while the validation and test sets are smaller and are used for hyperparameter tuning and final model performance evaluation, respectively.")

# Wypisywanie liczności zbiorów
#print_dataset_description(X_train, y_train, X_val, y_val, X_test, y_test)
'''