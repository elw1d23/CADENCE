"""
TCN model inference using ONNX Runtime
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path


class TCNInference:
    """Handle ONNX model loading and inference"""
    
    def __init__(self, model_path, config):
        self.verbose = config.get('verbose', True)
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load ONNX model
        
        Args:
            model_path: Path to ONNX model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if self.verbose:
            print(f"Loading model: {model_path.name}")
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get expected input shape
        input_shape = self.session.get_inputs()[0].shape
        
        if self.verbose:
            print(f"  Input name: {self.input_name}")
            print(f"  Expected input shape: {input_shape}")
            print(f"  Output name: {self.output_name}")
    
    def predict(self, features):
        """
        Run inference on features
        
        Args:
            features: Feature matrix [n_freq_bins, n_windows] (should be 100 × n_windows)
            
        Returns:
            predictions: Frame-level predictions [n_windows]
        """
        # Validate input dimensions
        if features.shape[0] != 100:
            raise ValueError(
                f"Feature dimension mismatch. Expected 100 features, got {features.shape[0]}"
            )
        
        if self.verbose:
            print(f"  Running TCN inference on {features.shape[1]} windows...")
        
        # Prepare input for ONNX model
        # MATLAB format: [100, n_time_steps, 1] with 'UUU' format
        # ONNX expects: [batch_size, features, time_steps] or [batch_size, time_steps, features]
        # We'll try the most common format: [1, 100, n_time_steps]
        
        input_data = features.reshape(1, features.shape[0], features.shape[1]).astype(np.float32)
        
        if self.verbose:
            print(f"  Input shape to model: {input_data.shape}")
        
        try:
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            
            predictions = outputs[0]
            
        except Exception as e:
            # Try alternative format if first attempt fails
            
            # Try [1, n_time_steps, 100]
            input_data = features.T.reshape(1, features.shape[1], features.shape[0]).astype(np.float32)
            
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            
            predictions = outputs[0]
        
        # Ensure predictions are 1D array
        predictions = np.squeeze(predictions)
        
        # Ensure predictions are in [0, 1] range (sigmoid should already be applied)
        predictions = np.clip(predictions, 0.0, 1.0)
        
        if self.verbose:
            print(f"  Inference complete. Output shape: {predictions.shape}")
            print(f"  Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
        
        return predictions