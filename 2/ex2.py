import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile


def read_audio(file_path):
    """
    Reads a WAV file, handles stereo-to-mono conversion, and normalizes to float.
    """
    sample_rate, audio_signal = wavfile.read(file_path)
    original_dtype = audio_signal.dtype
    if audio_signal.ndim > 1:
        audio_signal = audio_signal[:, 0]
    if "int" in str(original_dtype):
        max_val = np.iinfo(original_dtype).max
        audio_signal = audio_signal.astype(np.float64) / max_val
    else:
        max_val = 1.0

    return sample_rate, audio_signal, original_dtype, max_val


def apply_watermark_fft(audio_signal, sample_rate, target_freq, boost_val):
    """
    Applies a frequency domain watermark by boosting a specific frequency.
    """
    n_samples = len(audio_signal)
    fft_spectrum = np.fft.fft(audio_signal)
    fft_frequencies = np.fft.fftfreq(n_samples, d=1 / sample_rate)
    positive_idx = np.argmin(np.abs(fft_frequencies - target_freq))
    negative_idx = np.argmin(np.abs(fft_frequencies - (-target_freq)))
    fft_spectrum[positive_idx] = boost_val
    fft_spectrum[negative_idx] = boost_val
    new_audio_complex = np.fft.ifft(fft_spectrum)
    return np.real(new_audio_complex)


def save_audio(file_path, audio_signal, sample_rate, original_dtype, max_val):
    """
    Converts the audio back to its original format and saves it.
    """
    if "int" in str(original_dtype):
        current_max = np.max(np.abs(audio_signal))
        if current_max > 1.0:
            audio_signal = audio_signal / current_max
        audio_data_int = (audio_signal * max_val).astype(original_dtype)
    else:
        audio_data_int = audio_signal.astype(original_dtype)

    wavfile.write(file_path, sample_rate, audio_data_int)
    print("Done.")


def plot_wav_spectrum(file_path, save: bool = False):
    """
    Calculates and plots the FFT spectrum of a given WAV file.
    Useful for visually identifying watermarks.
    """
    fs, data, _, _ = read_audio(file_path)
    n_samples = len(data)
    fft_spectrum = np.fft.fft(data)
    fft_frequencies = np.fft.fftfreq(n_samples, d=1 / fs)
    positive_indices = np.where(fft_frequencies >= 0)
    frequencies = fft_frequencies[positive_indices]
    magnitudes = np.abs(fft_spectrum)[positive_indices]
    plt.figure(figsize=(12, 5))
    plt.plot(frequencies, magnitudes)
    file_name = os.path.basename(file_path)
    plt.title(f"Frequency Spectrum: {file_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    if save:
        plt.savefig(f"plots{os.sep}spectrum {file_name}.png")
    plt.show()


def plot_spectrogram(file_path, save: bool = False):
    """
    Plots the spectrogram and detects the dominant high-frequency watermark.
    """
    sample_rate, audio_signal = wavfile.read(file_path)
    if "int" in str(audio_signal.dtype):
        audio_signal = audio_signal.astype(np.float64)
    plt.figure(figsize=(12, 6))
    Pxx, freqs, bins, im = plt.specgram(audio_signal, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='inferno')
    file_name = os.path.basename(file_path)
    plt.title(f"Spectrogram: {file_name}")
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (Seconds)')
    cbar = plt.colorbar(im)
    cbar.set_label('Intensity (dB)')
    plt.axvline(x=5, color="black", linestyle='--', alpha=0.5)
    plt.axvline(x=10, color="black", linestyle='--', alpha=0.5)
    plt.axvline(x=15, color="black", linestyle='--', alpha=0.5)
    plt.axvline(x=20, color="black", linestyle='--', alpha=0.5)
    plt.axvline(x=25, color="black", linestyle='--', alpha=0.5)
    plt.axvline(x=30, color="black", linestyle='--', alpha=0.5)
    plt.axvline(x=35, color="black", linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    if save:
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig(f"plots{os.sep}spectrogram_{file_name}.png")

    plt.show()


TASK1_FOLDER = "Task 1" + os.path.sep
TASK2_FOLDER = "Task 2" + os.path.sep
TASK3_FOLDER = "Task 3" + os.path.sep

if __name__ == "__main__":
    # 1. Configuration

    # bad_audio = apply_watermark_fft(data, fs, target_freq=1000, boost_val=50000)
    # save_audio("Task1_BadWatermark.wav", bad_audio, fs, dtype, max_v)
    # good_audio = apply_watermark_fft(data, fs, target_freq=30000, boost_val=70000)
    # save_audio("Task1_GoodWatermark.wav", good_audio, fs, dtype, max_v)
    # folder = TASK3_FOLDER
    # for file in os.listdir(folder):
    #     plot_spectrogram(folder + file, False)
    sample_rate, audio_signal, original_dtype, max_val = read_audio(TASK1_FOLDER + "task1.wav")
    watermarked_audio = apply_watermark_fft(audio_signal, sample_rate, target_freq=20000, boost_val=70000)




