import torch

from vggt.models.vggt import VGGT # type: ignore
from vggt.utils.pose_enc import pose_encoding_to_extri_intri # type: ignore
from vggt.utils.geometry import unproject_depth_map_to_point_map # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

def init_model(load_url=_URL, device=DEVICE, dtype=DTYPE):
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(load_url))
    model.eval()
    model.to(device)
    return model

def model_inference(model, images):
    # Forward pass
    with torch.no_grad():
        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


    