import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from features import extract_features

# Define emotion mapping based on the naming convention
emotion_mapping = {
    'W': 'anger',
    'L': 'boredom',
    'A': 'anxiety',
    'F': 'happiness',
    'T': 'sadness',
    'E': 'disgust',
    'N': 'neutral'
}

def parse_filename(filename):
    speaker_number = filename[:2]
    emotion_code = filename[5]
    emotion = emotion_mapping.get(emotion_code, 'unknown')
    return speaker_number, emotion

def load_data(directory):
    features = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            signal, sr = librosa.load(filepath, sr=None)
            mfcc_features = extract_features(signal, sr)
            features.append(mfcc_features)
            speaker_number, emotion = parse_filename(filename)
            labels.append(emotion)
    
    max_length = max([len(mfcc) for mfcc in features])
    padded_features = pad_sequences(features, maxlen=max_length, padding='post', dtype='float32')

    features = np.array(padded_features)
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

# Directory containing audio files
directory = 'glosowe/data1'

# Load data
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_data(directory)

# Print shapes of the datasets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")
