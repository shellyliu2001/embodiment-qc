import csv
import numpy as np


# Calculate the percentage of frames where at least one hand is detected
def hands_in_frame_pct(outputs_per_frame):
    print(f"Calculating hands in frame percentage from {len(outputs_per_frame)} frames...")
    
    if len(outputs_per_frame) == 0:
        print("Warning: No frames provided")
        return 0.0
    
    frames_with_hands = 0
    for frame_idx, frame_outputs in enumerate(outputs_per_frame):
        num_objects = len(frame_outputs)
        if num_objects > 0:
            frames_with_hands += 1
    
    percentage = (frames_with_hands / len(outputs_per_frame)) * 100.0
    print(f"Found {frames_with_hands}/{len(outputs_per_frame)} frames with at least one hand ({percentage:.2f}%)")
    return percentage


# Calculate the percentage of frames where both hands are detected (2 or more objects)
def both_hands_in_frame_pct(outputs_per_frame):
    print(f"Calculating both hands in frame percentage from {len(outputs_per_frame)} frames...")
    
    if len(outputs_per_frame) == 0:
        print("Warning: No frames provided")
        return 0.0
    
    frames_with_both_hands = 0
    for frame_idx, frame_outputs in enumerate(outputs_per_frame):
        num_objects = len(frame_outputs)
        if num_objects >= 2:
            frames_with_both_hands += 1
    
    percentage = (frames_with_both_hands / len(outputs_per_frame)) * 100.0
    print(f"Found {frames_with_both_hands}/{len(outputs_per_frame)} frames with both hands ({percentage:.2f}%)")
    return percentage


# Calculate both metrics at once for convenience
def calculate_hand_metrics(outputs_per_frame):
    print("\n=== Calculating Hand Detection Metrics ===")
    hands_pct = hands_in_frame_pct(outputs_per_frame)
    both_hands_pct = both_hands_in_frame_pct(outputs_per_frame)
    
    results = {
        'hands_in_frame_pct': hands_pct,
        'both_hands_in_frame_pct': both_hands_pct
    }
    
    print(f"\nResults:")
    print(f"  Hands in frame: {hands_pct:.2f}%")
    print(f"  Both hands in frame: {both_hands_pct:.2f}%")
    print("=== Metrics Calculation Complete ===\n")
    
    return results


# Calculate hand metrics for multiple videos
def calculate_hand_metrics_for_videos(all_outputs, metrics_file_path=None):
    print(f"\n{'='*60}")
    print(f"Calculating Hand Detection Metrics for {len(all_outputs)} Video(s)")
    print(f"{'='*60}\n")
    
    all_results = {}
    
    for i, (video_path, outputs_per_frame) in enumerate(all_outputs.items(), 1):
        print(f"\n--- Video {i}/{len(all_outputs)}: {video_path} ---")
        metrics = calculate_hand_metrics(outputs_per_frame)
        all_results[video_path] = metrics
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - Hand Detection Metrics Across All Videos")
    print(f"{'='*60}")
    print(f"{'Video Path':<50} {'Hands %':<12} {'Both Hands %':<12}")
    print("-" * 60)
    
    if metrics_file_path is not None:
        with open(metrics_file_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['video_path', 'hands_in_frame_pct', 'both_hands_in_frame_pct'])
            writer.writeheader()
            for video_path, metrics in all_results.items():
                writer.writerow({'video_path': video_path, 'hands_in_frame_pct': metrics['hands_in_frame_pct'], 'both_hands_in_frame_pct': metrics['both_hands_in_frame_pct']})
        print(f"Metrics saved to {metrics_file_path}")
    
    return all_results
