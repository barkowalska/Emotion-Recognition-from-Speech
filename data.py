import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from features import extract_features
from sklearn.preprocessing import MinMaxScaler

emotion_mapping_emodb = {
    'W': 'anger',
    'L': 'boredom',
    'A': 'anxiety',
    'F': 'happiness',
    'T': 'sadness',
    'E': 'disgust',
    'N': 'neutral'
}

emotion_mapping_cremad = {
    'SAD': 'sadness',
    'ANG': 'anger',
    'DIS': 'disgust',
    'FEA': 'anxiety',
    'HAP': 'happiness',
    'NEU': 'neutral'
}


def parse_filename_emodb(filename):
    speaker_number = filename[:2]
    emotion_code = filename[5]
    emotion = emotion_mapping_emodb.get(emotion_code, 'unknown')
    return speaker_number, emotion

def parse_filename_cremad(filename):
    parts = filename.split('_')
    speaker_number = parts[0]
    emotion_code = parts[2]
    emotion = emotion_mapping_cremad.get(emotion_code, 'unknown')
    return speaker_number, emotion



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
    max_length = max(len(signal) for signal in signals)
    padded_signals = np.array([np.pad(signal, (0, max_length - len(signal)), 'constant') for signal in signals])
    return padded_signals

def load_signals(directory):
    #frames = []
    signals=[]
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            signal, sr = librosa.load(filepath, sr=None)
            #frame = extract_frames(signal, sr)
            #frames.append(frame)
            signals.append(signal)

    signals=pad_signals(signals)
    #return frames
    return signals
   

def load_labels(directories, dataset_type):
    labels = []
    
    
    
    for filename in os.listdir(directories):
        if filename.endswith('.wav'):
            filepath = os.path.join(directories, filename)
                
                                    
            if dataset_type == 'emodb':
                        speaker_number, emotion = parse_filename_emodb(filename)
            elif dataset_type == 'cremad':
                        speaker_number, emotion = parse_filename_cremad(filename)
            labels.append(emotion)
               
    labels = np.array(labels)

    return  labels

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
                elif dataset_type == 'cremad':
                            speaker_number, emotion = parse_filename_cremad(filename)
                labels.append(emotion)
               
    
    if not features:
        raise ValueError("No valid audio files found")
    
    labels = np.array(labels)

    return features, labels,


def preprocess_frames(emodb_dir):
    frames=load_frames(emodb_dir)
    
    X_train, X_test = train_test_split(frames, test_size=0.3, random_state=42)


    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    
    return X_train, X_test

def preprocess_emodb_data(emodb_dir):
    features, labels = load_data([emodb_dir], 'emodb')
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_temp, y_train, y_temp = train_test_split(features, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    '''
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

def preprocess_cremad_data(cremad_dir):
    features, labels= load_data([cremad_dir], 'cremad')
    
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_temp, y_train, y_temp = train_test_split(features, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(features, X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

