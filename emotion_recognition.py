import os
import numpy as np
import librosa
import scipy.signal
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

emotion_mapping = {
    'W': 'anger',       # Wut (Anger)
    'L': 'boredom',     # Langeweile (Boredom)
    'A': 'anxiety',     # Angst (Anxiety/Fear)
    'F': 'happiness',   # Freude (Happiness)
    'T': 'sadness',     # Traurigkeit (Sadness)
    'E': 'disgust',     # Ekel (Disgust)
    'N': 'neutral'      # Neutral (Neutrality)
}

def compute_lpc(signal, order):
    lpc_coeffs = librosa.lpc(signal, order=order)
    return lpc_coeffs

#statistics features(min, max, mean, var)
def calc_statistics(feature):
    return np.min(feature), np.max(feature), np.mean(feature), np.var(feature)

def extract_features(signal, sample_rate):
    # Pre-emphasis
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

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) +  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Hamming Window
    frames *= np.hamming(frame_length)

    # Fourier Transform and Power Spectrum
    NFFT = 512

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Power Spectrum
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Mel Filter Bank

    nfilt = 40
    mel_filter_bank = librosa.filters.mel(sr=sample_rate, n_fft=NFFT, n_mels=nfilt)
    mel_spectrum = np.dot(pow_frames, mel_filter_bank.T)
    mel_spectrum = np.where(mel_spectrum == 0, np.finfo(float).eps, mel_spectrum)
    mel_spectrum = 20 * np.log10(mel_spectrum)  # dB

    # Mel-Frequency Cepstral Coefficients (MFCCs)
    num_ceps = 12
    mfcc = dct(mel_spectrum, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]  # Keep 2-13

    # Short-term Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=frame_step).T

    # Short-term Energy
    energy = np.array([sum(abs(signal[i:i+frame_length]**2)) for i in range(0, len(signal), frame_step)]).T

    # Pitch
    pitches, magnitudes = librosa.core.piptrack(y=signal, sr=sample_rate, n_fft=NFFT, hop_length=frame_step, fmin=75.0, fmax=600.0, win_length=frame_length)
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch.append(pitches[index, t])

    # Formant (using Linear Predictive Coding)
    lpc_order = 2 + sample_rate // 1000
    lpc_coeffs = compute_lpc(signal, lpc_order)
    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) >= 0]
    angles = np.angle(roots)
    frequencies = angles * (sample_rate / (2 * np.pi))
    formants = frequencies[frequencies > 90]

    mfcc_min, mfcc_max, mfcc_mean, mfcc_var = calc_statistics(mfcc)
    pitch_min, pitch_max, pitch_mean, pitch_var = calc_statistics(pitch)
    formant_min, formant_max, formant_mean, formant_var = calc_statistics(formants[:4])


# Combine all features
    features = np.hstack([
        mfcc_min, mfcc_max, mfcc_mean, mfcc_var,
        zcr.flatten(), energy.flatten(),
        pitch_min, pitch_max, pitch_mean, pitch_var,
        formant_min, formant_max, formant_mean, formant_var
    ])

    return features

def parse_filename(filename):
    speaker_number = filename[:2]
    emotion_code = filename[5]
    emotion = emotion_mapping.get(emotion_code, 'unknown')
    return speaker_number, emotion

# Directory containing audio files
directory = '/content/drive/MyDrive/glosowe/wav'


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
        print(f"{filename}: {emotion}")


max_length = max([len(mfcc) for mfcc in features])

# Pad all feature arrays to the maximum length
padded_features = pad_sequences(features, maxlen=max_length, padding='post', dtype='float32')

features = np.array(padded_features)
labels = np.array(labels)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print the shapes of the resulting datasets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

