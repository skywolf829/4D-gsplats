import torch
import torch.nn.functional as F
from vggt.models.vggt import VGGT # type: ignore
from vggt.utils.pose_enc import pose_encoding_to_extri_intri # type: ignore
from vggt.utils.geometry import unproject_depth_map_to_point_map # type: ignore
from vggt.dependency.track_predict import predict_tracks # type: ignore
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
import pycolmap
import numpy as np
import trimesh
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction

def init_model(load_url=_URL, device=DEVICE, dtype=DTYPE):
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(load_url))
    model.eval()
    model.to(device)
    return model

def model_inference(model, images, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    return extrinsic, intrinsic, depth_map, depth_conf, points_3d

def bundle_adjustment(image_paths, original_coords, points_3d, extrinsic, intrinsic, depth_conf, images, img_load_resolution, vggt_fixed_resolution, save_dir):
    image_size = np.array(images.shape[-2:])
    scale = img_load_resolution / vggt_fixed_resolution
    shared_camera = False

    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        # Predicting Tracks
        # Using VGGSfM tracker instead of VGGT tracker for efficiency
        # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
        # Will be fixed in VGGT v2

        # You can also change the pred_tracks to tracks from any other methods
        # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=512,
        )

        torch.cuda.empty_cache()

    # rescale the intrinsic matrix from 518 to 1024
    intrinsic[:, :2, :] *= scale
    track_mask = pred_vis_scores > 0.1

    # TODO: radial distortion, iterative BA, masks
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        points_3d,
        extrinsic,
        intrinsic,
        pred_tracks,
        image_size,
        masks=track_mask,
        max_reproj_error=0.5,
        shared_camera=shared_camera,
        camera_type="pinhole",
        points_rgb=points_rgb,
    )

    if reconstruction is None:
        raise ValueError("No reconstruction can be built with BA")

    # Bundle Adjustment
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    reconstruction_resolution = img_load_resolution
    
    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        image_paths,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {save_dir}/sparse")
    sparse_reconstruction_dir = save_dir / "sparse"
    sparse_reconstruction_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(save_dir / "sparse/points.ply")

    return True