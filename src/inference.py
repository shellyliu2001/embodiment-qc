import glob
import os


# Propagate segmentation masks from the initial frame through the entire video
def propagate_in_video(predictor, session_id):
    print(f"Starting propagation in video for session_id: {session_id}")
    outputs_per_frame = {}
    frame_count = 0
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_idx = response["frame_index"]
        outputs_per_frame[frame_idx] = response["outputs"]
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    print(f"Completed propagation: processed {frame_count} frames total")
    return outputs_per_frame

# Run SAM3 inference on a video: start a session, add a text prompt, and propagate through all frames
def run_inference_on_video(video_path, predictor, gpus_to_use, prompt_text_str = "hands"):
    print(f"\n=== Starting inference on video: {video_path} ===")
    print(f"Using prompt text: '{prompt_text_str}'")
    
    print("Starting new session...")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    print(f"Session started with session_id: {session_id}")

    frame_idx = 0
    print(f"Adding text prompt on frame {frame_idx}...")
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )
    print("Prompt added successfully")
    
    outputs_per_frame = propagate_in_video(predictor, session_id)
    print(f"=== Inference completed: {len(outputs_per_frame)} frames processed ===\n")
    return outputs_per_frame

# Run inference on multiple videos: globs for subfolders in a parent directory, each subfolder contains frames from a single video
def run_inference_on_videos(video_folder, gpus_to_use, predictor, prompt_text_str = "hands"):
    """
    Run inference on multiple videos.
    """
    print(f"Searching for video subfolders in: {video_folder}")
    # Glob for all subdirectories in the video folder
    video_paths = [d for d in glob.glob(os.path.join(video_folder, "*")) if os.path.isdir(d)]
    video_paths.sort()  # Sort for consistent processing order
    
    print(f"Found {len(video_paths)} video subfolder(s) to process")
    all_outputs = {}
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n--- Processing video {i}/{len(video_paths)}: {video_path} ---")
        outputs_per_frame = run_inference_on_video(video_path, predictor, gpus_to_use, prompt_text_str)
        all_outputs[video_path] = outputs_per_frame
        print(f"Inference completed for video: {video_path}")
    
    print(f"\n=== Completed inference on all {len(video_paths)} video(s) ===")
    return all_outputs