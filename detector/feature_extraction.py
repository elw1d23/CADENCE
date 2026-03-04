"""
Feature extraction module with PCEN

"""

import numpy as np
from scipy.signal import lfilter


class FeatureExtractor:
    """Extracts features matching training preprocessing"""
    
    def __init__(self, config):
        self.window_size = config['window_size']
        self.overlap = config['overlap']
        self.freq_range = config['freq_range']
        self.freq_bin_size = config['freq_bin_size']
        self.verbose = config.get('verbose', True)
        
        # PCEN parameters
        self.pcen_alpha = config['pcen']['alpha']
        self.pcen_delta = config['pcen']['delta']
        self.pcen_r = config['pcen']['r']
        self.pcen_epsilon = config['pcen']['epsilon']
        self.pcen_s = config['pcen']['s']
        
        # Calculate number of frequency bins
        self.n_freq_bins = int((self.freq_range[1] - self.freq_range[0]) / self.freq_bin_size)
    
    def extract_features(self, audio_data, sample_rate):
        """
        Extract features exactly matching MATLAB implementation
        
        Args:
            audio_data: Audio samples (1D numpy array)
            sample_rate: Sample rate in Hz
            
        Returns:
            features: Feature matrix [n_freq_bins, n_windows]
            time_vector: Time point for each window (1D array)
        """
        # Calculate window parameters (match MATLAB exactly)
        window_samples = int(round(self.window_size * sample_rate))
        hop_samples = int(round(window_samples * (1 - self.overlap)))
        nfft = 2 ** int(np.ceil(np.log2(window_samples)))
        
        # Calculate number of windows
        n_windows = int(np.floor((len(audio_data) - window_samples) / hop_samples)) + 1
        
        # Frequency vector
        freq_vector = np.arange(0, nfft//2 + 1) * sample_rate / nfft
        
        # Initialize outputs
        features = np.zeros((self.n_freq_bins, n_windows), dtype=np.float32)
        time_vector = np.zeros(n_windows, dtype=np.float32)
        
        # Window function (match MATLAB's hamming)
        window_func = np.hamming(window_samples)
        
        if self.verbose:
            print(f"  Extracting features: {self.n_freq_bins} freq bins × {n_windows} windows")
        
        # Extract features for each window
        for win_idx in range(n_windows):
            start_sample = win_idx * hop_samples
            end_sample = start_sample + window_samples
            
            if end_sample > len(audio_data):
                break
            
            # Apply window and zero-pad (match MATLAB exactly)
            windowed_signal = audio_data[start_sample:end_sample] * window_func
            padded_signal = np.concatenate([windowed_signal, 
                                           np.zeros(nfft - window_samples)])
            
            # Compute FFT
            fft_result = np.fft.fft(padded_signal, n=nfft)
            magnitude_spectrum = np.abs(fft_result[:nfft//2 + 1])
            
            # Extract energy in each frequency bin (match MATLAB method exactly)
            for bin_idx in range(self.n_freq_bins):
                bin_start_freq = self.freq_range[0] + bin_idx * self.freq_bin_size
                bin_end_freq = bin_start_freq + self.freq_bin_size
                
                # Find FFT bins within this frequency range (match MATLAB)
                bin_start_idx = np.searchsorted(freq_vector, bin_start_freq, side='left')
                bin_end_idx = np.searchsorted(freq_vector, bin_end_freq, side='right') - 1
                
                if bin_start_idx < len(magnitude_spectrum) and bin_end_idx < len(magnitude_spectrum):
                    # Sum energy and normalize by bin count (match MATLAB exactly)
                    n_bins = bin_end_idx - bin_start_idx + 1
                    features[bin_idx, win_idx] = np.sum(magnitude_spectrum[bin_start_idx:bin_end_idx+1]**2) / n_bins
            
            # Time at window CENTER (match MATLAB)
            time_vector[win_idx] = (start_sample + window_samples/2) / sample_rate
        
        if self.verbose:
            print(f"  Features extracted: {features.shape[0]} × {features.shape[1]}")
            print(f"  Time range: [{time_vector[0]:.3f} - {time_vector[-1]:.3f}] s")
        
        return features, time_vector
    
    def apply_pcen(self, features):
        """
        Apply PCEN (Per-Channel Energy Normalization)
        
        Args:
            features: Feature matrix [n_freq_bins, n_windows]
            
        Returns:
            features_pcen: PCEN-normalized features [n_freq_bins, n_windows]
        """
        if self.verbose:
            print("  Applying PCEN...")
        
        features_pcen = np.zeros_like(features, dtype=np.float32)
        
        # Apply PCEN to each frequency bin independently
        for freq_idx in range(features.shape[0]):
            # Temporal integration (smooth background estimate)
            # MATLAB: M = filter([1-s], [1, -s], features(freq_idx, :), features(freq_idx, 1))
            b = np.array([1 - self.pcen_s])
            a = np.array([1, -self.pcen_s])
            zi = features[freq_idx, 0] * (1 - self.pcen_s)
            M, _ = lfilter(b, a, features[freq_idx, :], zi=[zi])
            
            # Adaptive gain normalization
            # MATLAB: ((features ./ (M + epsilon)^alpha) + delta)^r - delta^r
            features_pcen[freq_idx, :] = (
                ((features[freq_idx, :] / (M + self.pcen_epsilon)**self.pcen_alpha) 
                 + self.pcen_delta)**self.pcen_r 
                - self.pcen_delta**self.pcen_r
            )
        
        if self.verbose:
            print(f"  PCEN applied. Range: [{np.min(features_pcen):.4f}, {np.max(features_pcen):.4f}]")
        
        return features_pcen