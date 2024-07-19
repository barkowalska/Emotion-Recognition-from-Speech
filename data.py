import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from features import extract_features

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

# Define emotion mapping based on the naming convention for CREMA-D
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

def preprocess_emodb_data(emodb_dir):
    features, labels = load_data([emodb_dir], 'emodb')
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_temp, y_train, y_temp = train_test_split(features, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

def preprocess_cremad_data(cremad_dir):
    features, labels= load_data([cremad_dir], 'cremad')
    
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_temp, y_train, y_temp = train_test_split(features, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(features, X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

