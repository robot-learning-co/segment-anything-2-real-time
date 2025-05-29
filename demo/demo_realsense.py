import cv2, numpy as np, torch, pyrealsense2 as rs
from sam2.build_sam import build_sam2_camera_predictor

# ───────── CONFIG ────────────────────────────────────────────────
MODEL_CFG       = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_small.pt"
SIZE            = (640, 480)
WIN             = "frame"

# ───────── CAMERA ────────────────────────────────────────────────
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, *SIZE, rs.format.rgb8, 30)
pipe.start(cfg)

# ───────── TORCH (bfloat16, TF32) ────────────────────────────────
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True

def new_predictor():
    return build_sam2_camera_predictor(MODEL_CFG, SAM2_CHECKPOINT)

predictor          = new_predictor()
frame0             = None            # frozen first frame
mask_preview       = None            # latest mask‑image for frame0
tracking           = False
obj_id             = 1

# ───────── mouse ────────────────────────────────────────────────
def mouse_cb(event, x, y, flags, userdata):
    global mask_preview, tracking
    if tracking:                                      # ignore clicks after C
        return
    if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        return
    lbl  = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
    pts  = np.array([[x, y]], np.float32)
    lbls = np.array([lbl],  np.int32)

    # send just this click → get updated mask logits
    _, _, logits = predictor.add_new_prompt(
                        frame_idx=0, obj_id=obj_id,
                        points=pts, labels=lbls)

    mask_preview = (logits[0] > 0.0).permute(1,2,0).cpu().numpy().astype(np.uint8)*255

    # visual dot so user sees the click location
    col = (0,255,0) if lbl else (0,0,255)
    cv2.circle(frame0, (x,y), 4, col, -1)

cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, mouse_cb)

# ───────── helpers ───────────────────────────────────────────────
def reset():
    global frame0, mask_preview, tracking, predictor
    predictor     = new_predictor()
    frame0        = None
    mask_preview  = None
    tracking      = False
    print("[INFO] Reset – waiting for new first frame")

def composite(img, mask):
    """overlay mask + border onto BGR img → BGR"""
    if mask is None:
        return img
    h,w = mask.shape[:2]
    border = np.zeros((h,w), np.uint8)
    cntrs,_ = cv2.findContours(mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(border, cntrs, -1, 255, 2)
    out = cv2.addWeighted(img, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.5, 0)
    out[border==255] = [255,0,0]
    return out

# ───────── main loop ────────────────────────────────────────────
try:
    while True:
        frame_rs = pipe.wait_for_frames().get_color_frame()
        frame    = np.asanyarray(frame_rs.get_data())

        # ── freeze first frame ──
        if frame0 is None:
            frame0 = frame.copy()
            predictor.load_first_frame(frame0)
            cv2.imshow(WIN, cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            continue

        if not tracking:
            # still collecting clicks: show frame0 + preview mask
            disp = composite(frame0.copy(), mask_preview)
            cv2.imshow(WIN, cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and mask_preview is not None:   # commit
                tracking = True
                print("[INFO] Prompt committed – live tracking started")
            elif key == ord('r'):
                reset()
            elif key == ord('q'):
                break
            continue

        # ── tracking branch ──
        obj_ids, logits = predictor.track(frame)
        h,w             = frame.shape[:2]
        all_mask        = np.zeros((h,w,1), np.uint8)
        border          = np.zeros((h,w),   np.uint8)
        for lg in logits:
            m = (lg>0).permute(1,2,0).cpu().numpy().astype(np.uint8)*255
            all_mask = cv2.bitwise_or(all_mask, m)
            cntrs,_ = cv2.findContours(m.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(border, cntrs, -1, 255, 4)
        live = frame.copy()
        #live = cv2.addWeighted(frame, 1, cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB), 0.5, 0)
        live[border==255] = [255,0,0]
        cv2.imshow(WIN, cv2.cvtColor(live, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            reset()
        elif key == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()