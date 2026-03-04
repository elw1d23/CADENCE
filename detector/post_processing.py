# Creates the detection output file which is a .det file containing timestamps and metadata of the detections
"""
Detection post-processing module
Handles threshold application, event grouping, and merging
"""

import numpy as np


class DetectionPostProcessor:
    """Post-process predictions into detection events"""
    
    def __init__(self, config):
        self.threshold = config['threshold']
        self.min_duration = config['min_duration']
        self.max_gap = config['max_gap']
        self.remove_isolated = config.get('remove_isolated', False)
        self.window_size = config['window_size']
        self.verbose = config.get('verbose', True)
    
    def process(self, predictions, time_vector):
        """
        Convert predictions to detection events
        
        Args:
            predictions: Network predictions [n_windows]
            time_vector: Time points [n_windows]
            
        Returns:
            click_events: List of raw click detections (dicts)
            grouped_events: List of grouped porpoise events (dicts)
            binary_detections: Binary detection array [n_windows]
        """
        # Step 1: Apply threshold
        binary_detections = predictions > self.threshold
        n_detections = np.sum(binary_detections)
        
        if self.verbose:
            print(f"\n=== Detection Post-Processing ===")
            print(f"  Threshold: {self.threshold}")
            print(f"  Detected windows: {n_detections}/{len(predictions)} ({100*n_detections/len(predictions):.2f}%)")
        
        # Step 2: Extract raw click events (for clickEvents.det)
        click_events = self._extract_click_events(predictions, time_vector, binary_detections)
        
        if self.verbose:
            print(f"  Raw click events: {len(click_events)}")
        
        # Step 3: Group and merge events (for porpoiseEvents.det)
        grouped_events = self._group_events(click_events, time_vector)
        
        if self.verbose:
            print(f"  Grouped events: {len(grouped_events)}")
            print("=================================\n")
        
        return click_events, grouped_events, binary_detections
    
    def _extract_click_events(self, predictions, time_vector, binary_detections):
        """
        Extract individual click events (5ms windows above threshold)
        
        Returns:
            List of dicts with keys: start_time, end_time, confidence, window_idx
        """
        click_events = []
        
        # Find all windows above threshold
        detection_indices = np.where(binary_detections)[0]
        
        for idx in detection_indices:
            # Calculate window start and end times
            center_time = time_vector[idx]
            start_time = center_time - self.window_size / 2
            end_time = center_time + self.window_size / 2
            
            click_events.append({
                'start_time': start_time,
                'end_time': end_time,
                'confidence': predictions[idx],
                'window_idx': idx
            })
        
        return click_events
    
    def _group_events(self, click_events, time_vector):
        """
        Group nearby click events into porpoise encounter events
        
        Args:
            click_events: List of individual click events
            time_vector: Time points array
            
        Returns:
            List of grouped events (dicts)
        """
        if len(click_events) == 0:
            return []
        
        # Step 1: Find continuous regions
        events = self._find_continuous_regions(click_events)
        
        if len(events) == 0:
            return []
        
        # Step 2: Filter by minimum duration
        events = self._filter_by_duration(events)
        
        # Step 3: Merge nearby events
        if self.max_gap > 0 and len(events) > 1:
            events = self._merge_nearby_events(events)
        
        return events
    
    def _find_continuous_regions(self, click_events):
        """
        Group consecutive click events
        
        Returns:
            List of event dicts with aggregated statistics
        """
        if len(click_events) == 0:
            return []
        
        grouped_events = []
        current_clicks = [click_events[0]]
        
        for i in range(1, len(click_events)):
            prev_click = click_events[i-1]
            curr_click = click_events[i]
            
            # Check if clicks are consecutive (window indices differ by 1)
            if curr_click['window_idx'] == prev_click['window_idx'] + 1:
                current_clicks.append(curr_click)
            else:
                # Save current group and start new one
                if len(current_clicks) > 0:
                    grouped_events.append(self._create_grouped_event(current_clicks))
                current_clicks = [curr_click]
        
        # Add the last group
        if len(current_clicks) > 0:
            grouped_events.append(self._create_grouped_event(current_clicks))
        
        return grouped_events
    
    def _create_grouped_event(self, clicks):
        """Create a grouped event from a list of clicks"""
        confidences = [c['confidence'] for c in clicks]
        
        return {
            'start_time': clicks[0]['start_time'],
            'end_time': clicks[-1]['end_time'],
            'duration': clicks[-1]['end_time'] - clicks[0]['start_time'],
            'n_clicks': len(clicks),
            'mean_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'clicks': clicks  # Keep reference to constituent clicks
        }
    
    def _filter_by_duration(self, events):
        """Filter events by minimum duration"""
        if self.min_duration <= 0:
            return events
        
        filtered = [e for e in events if e['duration'] >= self.min_duration]
        
        n_removed = len(events) - len(filtered)
        if self.verbose and n_removed > 0:
            print(f"  Removed {n_removed} events shorter than {self.min_duration}s")
        
        return filtered
    
    def _merge_nearby_events(self, events):
        """
        Merge events that are within max_gap of each other
        
        Args:
            events: List of event dicts
            
        Returns:
            List of merged events
        """
        if len(events) <= 1:
            return events
        
        merged_events = []
        current_event = events[0].copy()
        
        for i in range(1, len(events)):
            gap = events[i]['start_time'] - current_event['end_time']
            
            if gap <= self.max_gap:
                # Merge events
                # Combine clicks from both events
                combined_clicks = current_event['clicks'] + events[i]['clicks']
                confidences = [c['confidence'] for c in combined_clicks]
                
                current_event = {
                    'start_time': current_event['start_time'],
                    'end_time': events[i]['end_time'],
                    'duration': events[i]['end_time'] - current_event['start_time'],
                    'n_clicks': len(combined_clicks),
                    'mean_confidence': np.mean(confidences),
                    'max_confidence': np.max(confidences),
                    'clicks': combined_clicks
                }
            else:
                # Save current event and start new one
                merged_events.append(current_event)
                current_event = events[i].copy()
        
        # Add the last event
        merged_events.append(current_event)
        
        n_merged = len(events) - len(merged_events)
        if self.verbose and n_merged > 0:
            print(f"  Merged {n_merged} events within {self.max_gap}s gap")
        
        return merged_events