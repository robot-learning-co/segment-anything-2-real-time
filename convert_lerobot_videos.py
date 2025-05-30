import os
import subprocess

input_dir = '/home/ubuntu/.cache/huggingface/lerobot/trlc/full_data_set/videos/chunk-000/observation.images.cam_head'
for fname in os.listdir(input_dir):
    if not fname.lower().endswith('.mp4'):
        continue

    in_path = os.path.join(input_dir, fname)
    name, _ = os.path.splitext(fname)
    tmp_path = os.path.join(input_dir, f"{name}.tmp.mp4")

    cmd = [
        'ffmpeg',
        '-y',                     # allow overwriting any existing tmp file
        '-i', in_path,
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-pix_fmt', 'yuv420p',
        '-crf', '22',
        '-preset', 'medium',
        '-movflags', '+faststart',
        tmp_path
    ]

    try:
        print(f"Converting {fname} → {name}.tmp.mp4…")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # atomically replace the old file with the new one
        os.replace(tmp_path, in_path)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {fname}:")
        print(e.stderr.decode())
        # clean up partial tmp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        print(f"✅ Overwrote {fname}")

print("All done!")