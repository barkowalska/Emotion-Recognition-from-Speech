import numpy as np
import librosa
from scipy.fftpack import dct
from sklearn.decomposition import PCA


def compute_lpc(signal, order):
    return librosa.lpc(signal, order=order)

def calc_statistics(feature):
    return np.min(feature), np.max(feature), np.mean(feature), np.var(feature)

def reduce_dimensionality(frames, n_components=2):
    # Apply PCA
    pca = PCA(n_components=n_components)
    frames_reduced = pca.fit_transform(frames)
    return frames_reduced

def extract_features(signal, sample_rate):
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
    
    # Fourier Transform and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Mel Filter Bank
    nfilt = 40
    mel_filter_bank = librosa.filters.mel(sr=sample_rate, n_fft=NFFT, n_mels=nfilt)
    mel_spectrum = np.dot(pow_frames, mel_filter_bank.T)
    mel_spectrum = np.where(mel_spectrum == 0, np.finfo(float).eps, mel_spectrum)  # Numerical Stability
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
    formants = frequencies[frequencies > 90]  # Consider frequencies above 90Hz as formants

    mfcc_min, mfcc_max, mfcc_mean, mfcc_var = calc_statistics(mfcc)
    pitch_min, pitch_max, pitch_mean, pitch_var = calc_statistics(pitch)
    formant_min, formant_max, formant_mean, formant_var = calc_statistics(formants[:4])
    zcr_min, zcr_max, zcr_mean, zcr_var = calc_statistics(zcr)
    energy_min, energy_max, energy_mean, energy_var = calc_statistics(energy)

    # Combine all features
    features = np.hstack([
        mfcc_min, mfcc_max, mfcc_mean, mfcc_var,
        zcr.flatten(), energy.flatten(),
        pitch_min, pitch_max, pitch_mean, pitch_var,
        formant_min, formant_max, formant_mean, formant_var
    ])

    return features
