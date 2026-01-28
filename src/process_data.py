import cv2
import numpy as np
import os
import tempfile
import glob

# Extract frames from a video file with optional stride and maximum frame limit, converting to RGB format
def convert_video_to_frames(video_path, max_frames=200, stride=4):
    """Read frames from video with stride; returns list of RGB frames (H,W,3)."""
    print(f"Reading frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    n = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if n % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        n += 1
    cap.release()
    print(f"Extracted {len(frames)} frames from video")
    return frames

# Write RGB frames to a temporary directory as JPEG images, converting back to BGR for OpenCV
def write_video_frames(frames):
    """Write frames to a directory."""
    print(f"Writing {len(frames)} frames to directory...")
    frames_dir = tempfile.mkdtemp(prefix="sam3_frames_")
    for i, fr in enumerate(frames):
        path = os.path.join(frames_dir, f"{i:05d}.jpg")
        cv2.imwrite(path, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    print(f"Wrote {len(frames)} frames to {frames_dir}")
    return frames_dir

# Process a single video: extract frames and save them to a temporary directory
def process_video(video_path, max_frames=200, stride=4):
    print(f"Processing video: {video_path}")
    frames = convert_video_to_frames(video_path, max_frames, stride)
    frames_dir = write_video_frames(frames)
    del frames # free up memory
    print(f"Completed processing: {video_path}")
    return frames_dir

# Process all MP4 videos in a folder: extract frames from each video and save to temporary directories
def process_videos(video_folder, max_frames=200, stride=4):
    print(f"Processing videos from folder: {video_folder}")
    video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))
    print(f"Found {len(video_paths)} video(s) to process")
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n--- Processing video {i}/{len(video_paths)} ---")
        process_video(video_path, max_frames, stride)
    print(f"\nCompleted processing all {len(video_paths)} video(s)")
