"""
Audio loading and preprocessing module

"""

import numpy as np
import soundfile as sf
from pathlib import Path


class AudioProcessor:
    """Handle audio loading and preprocessing"""
    
    def __init__(self, config):
        self.gain_factor = config['gain_factor']
        self.verbose = config.get('verbose', True)
    
    def load_audio(self, audio_path):
        """
        Load audio file and apply preprocessing
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            audio_data: Preprocessed audio samples (1D numpy array)
            sample_rate: Sample rate in Hz
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.verbose:
            print(f"Loading: {audio_path.name}")
        
        # Load audio file
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
            if self.verbose:
                print("  Converted stereo to mono")
        
        # Remove DC offset (match MATLAB exactly)
        audio_data = audio_data - np.mean(audio_data)
        
        # Apply gain (match MATLAB exactly)
        # MATLAB: audio_data = audio_data * 20
        audio_data = audio_data * self.gain_factor
        
        if self.verbose:
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Duration: {len(audio_data)/sample_rate:.2f} s")
            print(f"  Applied gain: {self.gain_factor} dB")
        
        return audio_data, sample_rate