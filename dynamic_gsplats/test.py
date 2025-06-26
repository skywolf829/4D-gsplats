from pathlib import Path
import torch
from file_ops import (
    load_and_preprocess_images_square, 
    extract_frames_at_time, load_images,
    find_videos, get_video_durations
)
from scene_init import init_model, model_inference, bundle_adjustment
from scene_init import DEVICE
from gsplat_train import check_colmap_done, train_gsplat_timestep_0, find_latest_checkpoint, train_gsplat_timestep_n
import hashlib
from video_sync import sync_videos_get_offset, compute_synced_time_range

def preprocessing(all_video_paths : list[Path], offsets, output_dir, img_folder:str):

    print("Preprocessing videos")
    img_dir = Path(output_dir / img_folder)
    img_dir.mkdir(parents=True, exist_ok=True)
    img_paths = extract_frames_at_time(all_video_paths, offsets, 0.0, img_dir)

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
        images, img_load_resolution, output_dir)

def main():
    videos_path = Path(__file__).parent.parent / "examples" / "time_varying" / "sky_livingroom"
    all_video_paths = find_videos(videos_path)
    output_dir = Path(__file__).parent.parent / "output" / hashlib.md5(str(videos_path).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # sync videos first
    durations = get_video_durations(all_video_paths)
    print(f"Video durations: {durations}")
    offsets = sync_videos_get_offset([Path(p) for p in all_video_paths])
    print(f"Video offsets: {offsets}")

    
    time_start, time_end = compute_synced_time_range(offsets, durations)
    time_end = time_end - 0.1 # some epsilon for vid issues
    timestep = time_start
    fps = 1

    latest_checkpoint, max_key = find_latest_checkpoint(output_dir / "ckpts")
    if latest_checkpoint:
        timestep = max_key[0]+(1./fps)
    
    while timestep < time_end:
        print(f"Timestep: {timestep} / {time_end}")
        img_folder = f"synced_frames_{timestep:0.02f}"
        if timestep == 0.0:
            # Check if colmap already ran
            if not check_colmap_done(output_dir):
                # Get first frame stuff for the 
                img_paths, original_coords, images, img_load_resolution = preprocessing(all_video_paths, offsets, output_dir, img_folder)
                # Run first frame posing
                ts0_preprocess(img_paths, images, original_coords, img_load_resolution, output_dir)

            # train first timestep
            train_gsplat_timestep_0(str(output_dir), img_folder)
        else:
            latest_checkpoint, max_key = find_latest_checkpoint(output_dir / "ckpts")
            if latest_checkpoint is not None:
                print(f"Continuing from {latest_checkpoint}")
                previous_splats = torch.load(latest_checkpoint, map_location="cuda", weights_only=False)                
                new_images_paths = extract_frames_at_time(all_video_paths, offsets, timestep, output_dir / img_folder)
                new_images = load_images(new_images_paths)
                train_gsplat_timestep_n(str(output_dir), img_folder, timestep=timestep, new_images = new_images, starting_splats=previous_splats)
            else:
                print(f"Error: expected a latest checkpoint but found none")
                quit()
        timestep += (1/fps)

if __name__ == "__main__":
    main()