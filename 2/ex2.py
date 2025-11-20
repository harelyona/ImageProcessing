import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# --- 1. Define File Path ---
file_path = "Task 1/Task1.wav"

# --- 2. Read WAV File ---
sample_rate, audio_signal = wavfile.read(file_path)

# Store original data type for saving later
original_dtype = audio_signal.dtype

# Handle stereo files by taking only the first channel
if audio_signal.ndim > 1:
    audio_signal = audio_signal[:, 0]

# Normalize to floating point (-1.0 to 1.0) for processing
# This is good practice to avoid overflow issues
if "int" in str(original_dtype):
    max_val = np.iinfo(original_dtype).max
    audio_signal = audio_signal.astype(np.float64) / max_val


# --- 3. Calculate FFT ---
fft_spectrum = np.fft.fft(audio_signal)
n_samples = len(audio_signal)
fft_frequencies = np.fft.fftfreq(n_samples, d=1 / sample_rate)
fft_magnitude = np.abs(fft_spectrum)
positive_freq_indices = np.where(fft_frequencies >= 0)
frequencies_to_plot = fft_frequencies[positive_freq_indices]
magnitude_to_plot = fft_magnitude[positive_freq_indices]

# --- 6. NEW: Modify Spectrum and Perform Inverse Transform ---

# Make a copy to modify. This is important!
modified_spectrum = np.copy(fft_spectrum)

# --- Define watermark parameters ---
target_frequency = 10000  # The frequency to boost (e.g., 20,000 Hz)
# The magnitude of the complex number at the target frequency
# We will set it to a high value to make it detectable
boost_value = 50000

# --- Find the index for the positive target frequency ---
# We find the index of the frequency bin closest to our target
positive_index = np.argmin(np.abs(fft_frequencies - target_frequency))

# --- Find the index for the corresponding negative frequency ---
# The FFT spectrum is symmetric. For a real signal,
# S[k] must be the complex conjugate of S[-k].
# np.fft.fftfreq handily provides negative freqs for the second half
negative_index = np.argmin(np.abs(fft_frequencies - (-target_frequency)))
# Set the magnitude at the positive and negative frequencies
# We set both to the same complex value (with phase 0) for simplicity.
# This ensures the complex conjugate property is maintained.
modified_spectrum[positive_index] = boost_value
modified_spectrum[negative_index] = boost_value  # For a real signal, this should be the conjugate.
# Setting both to the same real value is a simple way
# to add a real sine wave.

# --- Perform Inverse FFT ---
# np.fft.ifft() converts the modified spectrum back to the time domain
new_audio_signal_complex = np.fft.ifft(modified_spectrum)
new_audio_signal = np.real(new_audio_signal_complex)
if "int" in str(original_dtype):
    max_new_val = np.max(np.abs(new_audio_signal))
    if max_new_val > 1.0:
        new_audio_signal = new_audio_signal / max_new_val
    new_audio_signal_int = (new_audio_signal * max_val).astype(original_dtype)
else:
    new_audio_signal_int = new_audio_signal.astype(original_dtype)
output_file_path = "Task1_watermarked.wav"
wavfile.write(output_file_path, sample_rate, new_audio_signal_int)