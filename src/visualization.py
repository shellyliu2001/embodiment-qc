import glob
import os
import torch
import numpy as np
from sam3.visualization_utils import load_frame, prepare_masks_for_visualization, visualize_formatted_frame_output, save_masklet_video, normalize_bbox
from torchvision.ops import masks_to_boxes

# Load video frames from a directory for visualization purposes (frames are not used by the model)
def load_video_frames_for_vis(video_path):
    print(f"Loading video frames for visualization from: {video_path}")
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    print(f"Found {len(video_frames_for_vis)} frame(s)")
    try:
        # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
        print("Sorted frames using integer-based sorting")
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        print(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()
    return video_frames_for_vis

# Convert formatted outputs per frame to masklet format required by save_masklet_video
def formatted_outputs_to_masklet(video_frames_for_vis, outputs_per_frame, default_prob=1.0):
    img0 = load_frame(video_frames_for_vis[0])
    H, W = img0.shape[:2]

    masklet = {}

    for frame_idx, objmask_dict in outputs_per_frame.items():
        out_obj_ids = []
        out_binary_masks = []
        out_boxes_xywh = []
        out_probs = []

        for obj_id, mask in objmask_dict.items():
            m = mask.detach().cpu() if isinstance(mask, torch.Tensor) else torch.tensor(mask)
            m = m.squeeze()
            m = (m > 0).to(torch.uint8)

            if int(m.sum().item()) == 0:
                continue

            # bbox from mask: XYXY absolute -> normalize -> XYWH normalized
            box_xyxy = masks_to_boxes(m.unsqueeze(0)).squeeze(0)   # abs [x1,y1,x2,y2]
            box_xyxy = normalize_bbox(box_xyxy, W, H)             # normalized

            x1, y1, x2, y2 = box_xyxy.tolist()
            box_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

            out_obj_ids.append(int(obj_id))
            out_binary_masks.append(m.numpy())
            out_boxes_xywh.append(box_xywh)
            out_probs.append(float(default_prob))

        masklet[frame_idx] = {
            "out_boxes_xywh": np.array(out_boxes_xywh, dtype=np.float32),
            "out_probs": np.array(out_probs, dtype=np.float32),
            "out_obj_ids": np.array(out_obj_ids, dtype=np.int32),
            "out_binary_masks": (np.stack(out_binary_masks, axis=0).astype(np.uint8)
                                 if len(out_binary_masks) > 0 else np.zeros((0, H, W), dtype=np.uint8)),
        }

    return masklet

# Save visualization video with segmentation masks overlaid on frames
def save_video(video_frames_for_vis, outputs_per_frame, out_path):
    masklet_outputs = formatted_outputs_to_masklet(video_frames_for_vis, outputs_per_frame)
    save_masklet_video(
        video_frames=video_frames_for_vis,
        outputs=masklet_outputs,
        out_path=out_path,
    )

# Convert inference outputs format to visualization format
def convert_outputs_for_visualization(outputs_per_frame, video_frames_for_vis=None):
    """
    Convert outputs_per_frame from inference format to format expected by prepare_masks_for_visualization.
    
    Args:
        outputs_per_frame: Dictionary mapping frame_index to dict of {object_id: mask_array}
        video_frames_for_vis: Optional list of frame paths to determine mask shape
    
    Returns:
        Dictionary mapping frame_index to dict with 'out_obj_ids' and 'out_binary_masks'
    """
    # Get mask shape from first non-empty frame or from video frames
    mask_shape = None
    if video_frames_for_vis:
        try:
            img0 = load_frame(video_frames_for_vis[0])
            mask_shape = img0.shape[:2]  # (H, W)
        except:
            pass
    
    # If we couldn't get shape from video, try from first non-empty mask
    if mask_shape is None:
        for frame_idx, objmask_dict in outputs_per_frame.items():
            if len(objmask_dict) > 0:
                sample_mask = list(objmask_dict.values())[0]
                if isinstance(sample_mask, torch.Tensor):
                    mask_shape = tuple(sample_mask.shape)
                else:
                    mask_shape = np.array(sample_mask).shape
                break
        
        if mask_shape is None:
            mask_shape = (1080, 1920)  # Default shape
    
    converted = {}
    for frame_idx, objmask_dict in outputs_per_frame.items():
        out_obj_ids = []
        out_binary_masks = []
        
        for obj_id, mask in objmask_dict.items():
            # Convert mask to numpy array if needed
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # Ensure boolean mask
            mask_bool = mask_np.astype(bool)
            
            # Skip empty masks
            if not mask_bool.any():
                continue
            
            out_obj_ids.append(int(obj_id))
            out_binary_masks.append(mask_bool.astype(np.uint8))
        
        if len(out_obj_ids) > 0:
            converted[frame_idx] = {
                "out_obj_ids": np.array(out_obj_ids, dtype=np.int32),
                "out_binary_masks": np.stack(out_binary_masks, axis=0).astype(np.uint8)
            }
        else:
            # Empty frame - use determined shape
            converted[frame_idx] = {
                "out_obj_ids": np.array([], dtype=np.int32),
                "out_binary_masks": np.zeros((0, *mask_shape), dtype=np.uint8)
            }
    
    return converted

# Visualize segmentation outputs per frame with optional stride and save to video file
def visualize_outputs_per_frame(outputs_per_frame, video_path, stride=20, out_path=None):
    video_frames_for_vis = load_video_frames_for_vis(video_path)
    # Convert to format expected by prepare_masks_for_visualization
    converted_outputs = convert_outputs_for_visualization(outputs_per_frame, video_frames_for_vis)
    outputs_per_frame = prepare_masks_for_visualization(converted_outputs)
    assert len(outputs_per_frame) == len(video_frames_for_vis), "Number of frames must match"

    for frame_idx in range(0, len(outputs_per_frame), stride):
        visualize_formatted_frame_output(
            frame_idx,
            video_frames_for_vis,
            outputs_list=[outputs_per_frame],
            titles=["SAM 3 Dense Tracking outputs"],
            figsize=(6, 4),
        )
    
    if out_path is not None:
        save_video(video_frames_for_vis, outputs_per_frame, out_path)