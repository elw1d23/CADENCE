"""
Feature normalization module
Uses saved training normalization parameters from MATLAB

Note - if model is retrained the normalisation parameters must be updated!!!! If there is a mismatch between training and inference the pipeline will fail
"""

import numpy as np
import scipy.io
from pathlib import Path


class FeatureNormalizer:
    """Apply normalization using training parameters"""
    
    def __init__(self, normalization_params_path, config):
        self.verbose = config.get('verbose', True)
        self.load_normalization_params(normalization_params_path)
    
    def load_normalization_params(self, params_path):
        """
        Load normalization parameters from MAT file
        
        Args:
            params_path: Path to normalization_params MAT file
        """
        params_path = Path(params_path)
        
        if not params_path.exists():
            raise FileNotFoundError(
                f"Normalization parameters file not found: {params_path}\n"
                f"This file is CRITICAL for correct inference!"
            )
        
        if self.verbose:
            print(f"Loading normalization parameters: {params_path.name}")
        
        # Load MAT file
        mat_data = scipy.io.loadmat(str(params_path))
        
        if 'normalization_params' not in mat_data:
            raise ValueError(
                f"No 'normalization_params' found in {params_path}"
            )
        
        # Extract parameters (handle MATLAB struct format)
        params = mat_data['normalization_params']
        
        # Get method
        method = params['method'][0, 0][0]
        
        if method != 'percentile_normalization':
            raise ValueError(
                f"Unknown normalization method: {method}\n"
                f"Expected: 'percentile_normalization'"
            )
        
        # Get p5 and p95 values
        self.p5 = float(params['p5_value'][0, 0][0, 0])
        self.p95 = float(params['p95_value'][0, 0][0, 0])
        self.method = method
        
        # Verify PCEN was applied during training
        applied_pcen = int(params['applied_pcen'][0, 0][0, 0])
        if applied_pcen != 1:
            print("  WARNING: PCEN may not have been applied during training!")
        
        if self.verbose:
            print(f"  Method: {self.method}")
            print(f"  p5: {self.p5:.6e}")
            print(f"  p95: {self.p95:.6f}")
    
    def normalize(self, features):
        """
        Apply percentile normalisation using training parameters
        Matches MATLAB implementation exactly
        
        Args:
            features: Feature matrix [n_freq_bins, n_windows]
            
        Returns:
            features_norm: Normalised features [n_freq_bins, n_windows]
        """
        if self.verbose:
            print(f"  Applying normalisation: {self.method}")
            print(f"    Using p5={self.p5:.4e}, p95={self.p95:.4f} from training")
        
        # Apply same normalization as training (match MATLAB exactly)
        # MATLAB: features_norm = (features - p5) / (p95 - p5)
        features_norm = (features - self.p5) / (self.p95 - self.p5)
        
        # Clip to [0, 1.2] (match MATLAB)
        features_norm = np.clip(features_norm, 0.0, 1.2)
        
        # Scale to [0, 1] (match MATLAB)
        features_norm = features_norm / 1.2
        
        if self.verbose:
            print(f"  Final feature range: [{np.min(features_norm):.4f}, {np.max(features_norm):.4f}]")
        
        return features_norm