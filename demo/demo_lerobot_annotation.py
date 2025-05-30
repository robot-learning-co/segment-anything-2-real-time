import os
import torch
import numpy as np
import cv2
from pathlib import Path

# Use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time

# Initialize SAM-2 predictor
sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# Load the video
video_path = "/home/ubuntu/.cache/huggingface/lerobot/trlc/full_data_set/videos/chunk-000/observation.images.cam_head/episode_000008.mp4"
print(f"Loading video from {video_path}")

# Open video and read first frame
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read first frame from video.")

# Prepare for interactive click
def on_mouse(event, x, y, flags, param):
    global click_point
    # Only set the click point on left-button click
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

click_point = None
window_name = "Select Prompt Point (Left-click)"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, on_mouse)

# Display until click or 'q'
while True:
    cv2.imshow(window_name, first_frame)
    key = cv2.waitKey(1) & 0xFF
    if click_point is not None:
        break
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

# Clean up UI
cv2.destroyAllWindows()

# Initialize predictor on the first frame (RGB)
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
predictor.load_first_frame(first_frame_rgb)

# Add prompt based on user click
ann_frame_idx = 0
obj_id = 1
points = np.array([click_point], dtype=np.float32)
labels = np.array([1], dtype=np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
    frame_idx=ann_frame_idx,
    obj_id=obj_id,
    points=points,
    labels=labels
)

# Process subsequent frames and display red mask overlay
masked_frames = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

    h, w = frame.shape[:2]
    mask_acc = np.zeros((h, w), dtype=np.uint8)
    for logit in out_mask_logits:
        m = (logit > 0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).squeeze() * 255
        mask_acc = cv2.bitwise_or(mask_acc, m)

    alpha = 0.9  # how strongly red you want the tint: 0.0 = no tint, 1.0 = full red

    red_img = np.zeros_like(frame)
    red_img[..., 2] = 255  # BGR → red channel

    blended = cv2.addWeighted(frame, 1.0 - alpha, red_img, alpha, 0)
    mask_bool = mask_acc.astype(bool)               # shape (h, w), True where mask
    mask_3c  = np.repeat(mask_bool[:, :, None], 3, axis=2)

    overlay = frame.copy()
    overlay[mask_3c] = blended[mask_3c]
    masked_frames.append(overlay)
    # 4. Show
    cv2.imshow("Tracking", overlay)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print("Press 'y' to confirm and save the video, or any other key to discard.")
key = cv2.waitKey(0)
if key == ord('y'):
    print("Video confirmed. Proceeding to save...")
    # You can call encode_masked_frames or any saving logic here
else:
    raise ValueError("Video discarded.")



cap.release()
cv2.destroyAllWindows()

import logging
import av
from pathlib import Path
from typing import List, Union

def encode_masked_frames(
    frames: List[np.ndarray],
    video_path: Union[Path, str],
    fps: int = 30,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    video_path = Path(video_path)
    if video_path.exists() and not overwrite:
        raise FileExistsError(f"{video_path} already exists (set overwrite=True to replace)")
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # codec support check
    if vcodec not in ("h264","hevc","libsvtav1"):
        raise ValueError(f"Unsupported codec {vcodec}")

    # fix incompatible pix_fmt
    if vcodec in ("hevc","libsvtav1") and pix_fmt == "yuv444p":
        logging.warning(f"Auto-downgrading pix_fmt to yuv420p for {vcodec}")
        pix_fmt = "yuv420p"

    # build ffmpeg options dict
    opts: dict[str,str] = {}
    if g is not None:       opts["g"]  = str(g)
    if crf is not None:     opts["crf"] = str(crf)
    if fast_decode:
        if vcodec=="libsvtav1":
            opts["svtav1-params"] = f"fast-decode={fast_decode}"
        else:
            opts["tune"] = "fastdecode"

    # set logging
    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    # open & write
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream(vcodec, rate=fps, options=opts)
        stream.pix_fmt = pix_fmt
        h, w = frames[0].shape[:2]
        stream.width = w
        stream.height = h

        for img in frames:
            # frame is BGR uint8 numpy
            frame_av = av.VideoFrame.from_ndarray(img, format="bgr24")
            for pkt in stream.encode(frame_av):
                container.mux(pkt)

        # flush
        for pkt in stream.encode():
            container.mux(pkt)

    # restore default logging
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Encoding failed, no file at {video_path}")

# — then at the bottom of your main script —
encode_masked_frames(
    masked_frames,
    video_path=video_path,
    fps=30,
    vcodec="libsvtav1",
    pix_fmt="yuv420p",
    g=2,
    crf=18,
    fast_decode=1,
    log_level=av.logging.ERROR,
    overwrite=True,
)