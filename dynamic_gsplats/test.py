from pathlib import Path
import glob
import torch
import numpy as np
from file_ops import extract_single_frame, load_and_preprocess_images_square
from scene_init import init_model, model_inference, bundle_adjustment
from scene_init import DEVICE, DTYPE
from gsplat_train import train_gsplat_from_colmap
import hashlib

videos_path = Path(__file__).parent.parent / "examples" / "time_varying" / "cut_roasted_beef"
output_dir = Path(__file__).parent.parent / "output" / hashlib.md5(str(videos_path).encode()).hexdigest()
output_dir.mkdir(parents=True, exist_ok=True)

videos = sorted(glob.glob(str(videos_path / "*.mp4")))
img_dir = Path(output_dir / "images")
img_dir.mkdir(parents=True, exist_ok=True)
img_paths = []
for video in videos:
    img_path = extract_single_frame(Path(video), 0, img_dir)
    img_paths.append(img_path)

vggt_fixed_resolution = 518
img_load_resolution = 1024

images, original_coords = load_and_preprocess_images_square(img_paths, img_load_resolution)
images = images.to(DEVICE)
original_coords = original_coords.to(DEVICE)
print(f"Loaded {len(images)} images from {img_dir}")

vggt_model = init_model()
print("Running model inference...")
extrinsic, intrinsic, depth_map, depth_conf, points_3d = model_inference(vggt_model, images)

print("Running bundle adjustment...")
bundle_adjustment(img_paths, original_coords, points_3d, extrinsic, intrinsic, depth_conf, 
    images, img_load_resolution, vggt_fixed_resolution, output_dir)

train_gsplat_from_colmap(str(output_dir))


