from pathlib import Path
import glob
import torch
import numpy as np
from file_ops import extract_single_frame_save_to_disc, load_and_preprocess_images_square, extract_frames_parallel, get_total_frames
from scene_init import init_model, model_inference, bundle_adjustment
from scene_init import DEVICE, DTYPE
from gsplat_train import check_colmap_done, train_gsplat_timestep_0, find_latest_checkpoint, train_gsplat_timestep_n
import hashlib
import imageio.v3 as iio

vggt_fixed_resolution = 518

def preprocessing(all_video_paths, output_dir):

    print("Preprocessing videos")
    img_dir = Path(output_dir / "images")
    img_dir.mkdir(parents=True, exist_ok=True)
    img_paths = []
    for video in all_video_paths:
        img_path = extract_single_frame_save_to_disc(Path(video), 0, img_dir)
        img_paths.append(img_path)

    images, original_coords, img_load_resolution = load_and_preprocess_images_square(img_paths)
    images = images.to(DEVICE)
    original_coords = original_coords.to(DEVICE)
    print(f"Loaded {len(images)} images from {img_dir} with mas resolution {img_load_resolution}")

    return img_paths, original_coords, images, img_load_resolution

def ts0_preprocess(img_paths, images, original_coords, img_load_resolution, output_dir):
    
    print("Initializing VGGT")
    vggt_model = init_model()
    print("Running model inference...")
    extrinsic, intrinsic, depth_map, depth_conf, points_3d = model_inference(vggt_model, images, output_dir=output_dir)
    print(f"Finished VGGT inference. Extrinsics: {extrinsic.shape}, intrinsic {intrinsic.shape}, depth_map {depth_map.shape}, depth_conf {depth_conf.shape}, points_3d {points_3d.shape}")

    print("Running bundle adjustment...")
    bundle_adjustment(img_paths, original_coords, points_3d, extrinsic, intrinsic, depth_conf, 
        images, img_load_resolution, vggt_fixed_resolution, output_dir)

def main():
    videos_path = Path(__file__).parent.parent / "examples" / "time_varying" / "cut_roasted_beef"
    all_video_paths = sorted(glob.glob(str(videos_path / "*.mp4")))
    output_dir = Path(__file__).parent.parent / "output" / hashlib.md5(str(videos_path).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = get_total_frames(all_video_paths[0])
    

    start_frame = 0
    frame_skip = 10

    latest_checkpoint, max_key = find_latest_checkpoint(output_dir / "ckpts")
    if latest_checkpoint:
        start_frame = max_key[0]+frame_skip
    
    for frame in range(start_frame, total_frames, frame_skip):
        if frame == 0:
            # Check if colmap already ran
            if not check_colmap_done(output_dir):
                # Get first frame stuff for the 
                img_paths, original_coords, images, img_load_resolution = preprocessing(all_video_paths, output_dir)
                # Run first frame posing
                ts0_preprocess(img_paths, images, original_coords, img_load_resolution, output_dir)

            # train first timestep
            train_gsplat_timestep_0(str(output_dir))
        else:
            latest_checkpoint, max_key = find_latest_checkpoint(output_dir / "ckpts")
            print(f"Continuing from {latest_checkpoint}")
            previous_splats = torch.load(latest_checkpoint, map_location="cuda", weights_only=False)
            new_images = extract_frames_parallel(all_video_paths, frame)
            # i = 0
            # for im in new_images:
            #     iio.imwrite(output_dir / "images" / f"cam{i:02d}_frame_{frame:06d}.png", im)
            #     i += 1
            train_gsplat_timestep_n(str(output_dir), frame_no=frame, new_images = new_images, starting_splats=previous_splats)
        

if __name__ == "__main__":
    main()