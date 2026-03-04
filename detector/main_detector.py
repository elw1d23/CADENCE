#!/usr/bin/env python3
"""
TCN Harbor Porpoise Click Detector
Main detection pipeline - processes audio files and outputs detection events

Usage:
    python detector.py config.yaml
"""

import sys
import yaml
import time
from pathlib import Path
from glob import glob
from itertools import chain

from audio_processor import AudioProcessor
from feature_extraction import FeatureExtractor
from normalisation import FeatureNormalizer
from tcn_inference import TCNInference
from post_processing import DetectionPostProcessor
from detection_writer import DetectionWriter


class PorpoiseDetector:
    """Main detector class that orchestrates pipeline"""
    
    def __init__(self, config_path):
        """
        Initialize detector with configuration
        
        Check congig.yaml for settings if you need to change anything 
                
        Args:
            config_path: Path to YAML configuration file
        """
        print("=" * 70)
        print("TCN Harbor Porpoise Click Detector")
        print("=" * 70)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize all components
        print("\n=== Initializing Components ===")
        self.audio_processor = AudioProcessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.normalizer = FeatureNormalizer(
            self.config['normalization_params'], 
            self.config
        )
        self.tcn_model = TCNInference(self.config['model_path'], self.config)
        self.post_processor = DetectionPostProcessor(self.config)
        self.detection_writer = DetectionWriter(self.config)
        
        print("=== Initialization Complete ===\n")
    
    def load_config(self, config_path):
        """Load YAML configuration file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        print(f"\nLoading configuration: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = [
            'model_path', 'normalization_params', 
            'input_directory', 'output_directory'
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        print(f"  Input directory: {config['input_directory']}")
        print(f"  Output directory: {config['output_directory']}")
        print(f"  Model: {Path(config['model_path']).name}")
        
        return config
    
    def process_file(self, audio_path):
        """
        Process a single audio file through the complete pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            results: Dictionary containing all processing results
        """
        audio_path = Path(audio_path)
        
        print("\n" + "=" * 70)
        print(f"Processing: {audio_path.name}")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Load and preprocess audio
            audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
            
            # Step 2: Extract features
            features, time_vector = self.feature_extractor.extract_features(
                audio_data, sample_rate
            )
            
            # Step 3: Apply PCEN
            features_pcen = self.feature_extractor.apply_pcen(features)
            
            # Step 4: Normalize features
            features_norm = self.normalizer.normalize(features_pcen)
            
            # Step 5: Run TCN inference
            predictions = self.tcn_model.predict(features_norm)
            
            # Step 6: Post-process detections
            click_events, grouped_events, binary_detections = \
                self.post_processor.process(predictions, time_vector)
            
            # Step 7: Write output files
            if self.config.get('save_click_events', True):
                click_path = self.detection_writer.write_click_events(
                    str(audio_path), sample_rate, click_events
                )
            
            if self.config.get('save_porpoise_events', True):
                porpoise_path = self.detection_writer.write_porpoise_events(
                    str(audio_path), sample_rate, grouped_events
                )
                
            if self.config.get('save_audacity_labels', True):
                audacity_path = self.detection_writer.write_audacity_labels(
                    str(audio_path), click_events, event_type = 'clicks'
                )
            
            # Calculate processing time
            elapsed_time = time.time() - start_time
            
            # Print summary
            print("\n=== Processing Summary ===")
            print(f"  Audio duration: {len(audio_data)/sample_rate:.2f} s")
            print(f"  Processing time: {elapsed_time:.2f} s")
            print(f"  Total click events: {len(click_events)}")
            print(f"  Total porpoise events: {len(grouped_events)}")
            print("=" * 70 + "\n")
            
            return {
                'filename': audio_path.name,
                'sample_rate': sample_rate,
                'duration': len(audio_data) / sample_rate,
                'n_click_events': len(click_events),
                'n_porpoise_events': len(grouped_events),
                'processing_time': elapsed_time,
                'success': True
            }
            
        except Exception as e:
            print(f"\nERROR processing {audio_path.name}: {str(e)}")
            print("=" * 70 + "\n")
            
            return {
                'filename': audio_path.name,
                'success': False,
                'error': str(e)
            }
    
    def process_directory(self):
        """
        Process all audio files in the input directory
        
        Returns:
            List of results for each processed file
        """
        input_dir = Path(self.config['input_directory'])
        
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all WAV files 
        
        extensions = ['*.wav', '*.WAV', '*.flac', '*.FLAC', '*.mp3', '*.MP3'] 
        audio_files = sorted(chain.from_iterable(input_dir.glob(ext) for ext in extensions))
        
        if len(audio_files) == 0:
            print(f"\nWARNING: No .wav files found in {input_dir}")
            return []
        
        print(f"\nFound {len(audio_files)} audio files to process")
        
        results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]")
            result = self.process_file(audio_file)
            results.append(result)
        
        # Print overall summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """Print summary of all processed files"""
        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 70)
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"Total files: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if len(successful) > 0:
            total_clicks = sum(r['n_click_events'] for r in successful)
            total_porpoise = sum(r['n_porpoise_events'] for r in successful)
            total_time = sum(r['processing_time'] for r in successful)
            
            print(f"\nTotal click events detected: {total_clicks}")
            print(f"Total porpoise events detected: {total_porpoise}")
            print(f"Total processing time: {total_time:.2f} s")
        
        if len(failed) > 0:
            print("\nFailed files:")
            for r in failed:
                print(f"  - {r['filename']}: {r['error']}")
        
        print("=" * 70 + "\n")


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python detector.py <config.yaml>")
        print("\nExample:")
        print("  python detector.py config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        # Create detector
        detector = PorpoiseDetector(config_path)
        
        # Process all files in directory
        detector.process_directory()
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()