"""
Detection file writer
Outputs .det files for click events and porpoise events
"""

import os
from pathlib import Path


class DetectionWriter:
    """Write detection results to .det files"""
    
    def __init__(self, config):
        self.output_dir = Path(config['output_directory'])
        self.hydrophone_id = config['hydrophone_id']
        self.threshold = config['threshold']
        self.window_size = config['window_size']
        self.min_duration = config['min_duration']
        self.max_gap = config['max_gap']
        self.remove_isolated = config.get('remove_isolated', False)
        self.verbose = config.get('verbose', True)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_click_events(self, audio_filename, sample_rate, click_events):
        """
        Write raw click detection events to .det file
        
        Args:
            audio_filename: Original audio filename
            sample_rate: Audio sample rate
            click_events: List of click event dicts
        """
        # Generate output filename
        base_name = Path(audio_filename).stem
        output_path = self.output_dir / f"{base_name}_clickEvents.det"
        
        if self.verbose:
            print(f"Writing click events: {output_path.name}")
        
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"# Hydrophone: {self.hydrophone_id}\n")
            f.write(f"# Filename: {os.path.basename(audio_filename)}\n")
            f.write(f"# Sample Rate: {sample_rate} Hz\n")
            f.write(f"# Detection Threshold: {self.threshold}\n")
            f.write(f"# Window Size: {self.window_size} s\n")
            
            # Write column headers
            f.write("event_id,start_time_s,end_time_s,confidence\n")
            
            # Write each click event
            for event_id, event in enumerate(click_events, start=1):
                f.write(f"{event_id},"
                       f"{event['start_time']:.5f},"
                       f"{event['end_time']:.5f},"
                       f"{event['confidence']:.4f}\n")
        
        if self.verbose:
            print(f"  Wrote {len(click_events)} click events")
        
        return output_path
    
    def write_porpoise_events(self, audio_filename, sample_rate, grouped_events):
        """
        Write grouped porpoise events to .det file
        
        Args:
            audio_filename: Original audio filename
            sample_rate: Audio sample rate
            grouped_events: List of grouped event dicts
        """
        # Generate output filename
        base_name = Path(audio_filename).stem
        output_path = self.output_dir / f"{base_name}_porpoiseEvents.det"
        
        if self.verbose:
            print(f"Writing porpoise events: {output_path.name}")
        
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"# Hydrophone: {self.hydrophone_id}\n")
            f.write(f"# Filename: {os.path.basename(audio_filename)}\n")
            f.write(f"# Sample Rate: {sample_rate} Hz\n")
            f.write(f"# Detection Threshold: {self.threshold}\n")
            f.write(f"# Min Duration: {self.min_duration} s\n")
            f.write(f"# Max Gap: {self.max_gap} s\n")
            f.write(f"# Remove Isolated: {self.remove_isolated}\n")
            
            # Write column headers
            f.write("event_id,start_time_s,end_time_s,duration_s,n_clicks,"
                   "mean_confidence,max_confidence\n")
            
            # Write each grouped event
            for event_id, event in enumerate(grouped_events, start=1):
                f.write(f"{event_id},"
                       f"{event['start_time']:.5f},"
                       f"{event['end_time']:.5f},"
                       f"{event['duration']:.5f},"
                       f"{event['n_clicks']},"
                       f"{event['mean_confidence']:.4f},"
                       f"{event['max_confidence']:.4f}\n")
        
        if self.verbose:
            print(f"  Wrote {len(grouped_events)} porpoise events")
        
        return output_path
    
    def write_audacity_labels(self, audio_filename, click_events, event_type='clicks'):
        """
        Write detections to Audacity label file format
        
        Args:
            audio_filename: Original audio filename
            click_events: List of click event dicts
            event_type: 'clicks' or 'porpoise' for different label styles
        """
        # Generate output filename
        base_name = Path(audio_filename).stem
        output_path = self.output_dir / f"{base_name}_audacity.txt"
        
        if self.verbose:
            print(f"Writing Audacity labels: {output_path.name}")
        
        with open(output_path, 'w') as f:
            for i, event in enumerate(click_events, start=1):
                # Tab-separated: start_time, end_time, label
                if event_type == 'clicks':
                    label = f"Click_{i}"
                else:
                    # For grouped events, show number of clicks
                    n_clicks = event.get('n_clicks', 1)
                    label = f"Event_{i} ({n_clicks} clicks)"
                
                f.write(f"{event['start_time']:.5f}\t{event['end_time']:.5f}\t{label}\n")
        
        if self.verbose:
            print(f"  Wrote {len(click_events)} labels")
        
        return output_path