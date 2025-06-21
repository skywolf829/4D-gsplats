import imageio.v3 as iio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset

def load_video(video_path):
    vid = iio.imread(video_path)
    # convert to float32
    vid = vid.astype(np.float32) / 255.0
    return vid

def load_image(image_path):
    img = iio.imread(image_path)
    # convert to float32
    img = img.astype(np.float32) / 255.0
    return img

def load_images(image_paths):
    # load images in parallel and in order with multithreading
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        images = list(executor.map(load_image, image_paths))
    return images

def load_images_from_dir(image_dir):
    image_paths = sorted(list(Path(image_dir).iterdir()))
    return load_images(image_paths)

def resize_images_to_similar_square(images, final_pixels=518):
    def pad_and_resize(img):
        h, w = img.shape[:2]
        # Determine the size of the square (use the larger dimension)
        square_size = max(h, w)
        
        # Calculate padding needed
        pad_h = square_size - h
        pad_w = square_size - w
        
        # Pad to make square (center the image)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Pad with black (0 values)
        # Use numpy padding which handles different image formats better
        if len(img.shape) == 3:
            pad_width = ((top, bottom), (left, right), (0, 0))  # Don't pad channel dimension
        else:
            pad_width = ((top, bottom), (left, right))  # Grayscale
        padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
        
        # Resize to final dimensions
        resized_img = cv2.resize(padded_img, (final_pixels, final_pixels), interpolation=cv2.INTER_CUBIC)
        
        return resized_img
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        resized_images = list(executor.map(pad_and_resize, images))
    return resized_images

def extract_single_frame(video_file, timestep, target_dir_images):
    """
    Extract a single frame from a video file at the specified timestep.
    Memory efficient - only loads the specific frame needed.
    Returns the image path if successful, None otherwise.
    """
    try:
        vs = cv2.VideoCapture(str(video_file))
        vs.set(cv2.CAP_PROP_POS_FRAMES, timestep)
        ret, frame = vs.read()
        vs.release()
        
        if ret:
            image_path = target_dir_images / f"{video_file.stem}_frame_{timestep:06d}.png"
            cv2.imwrite(str(image_path), frame)
            return image_path
    except Exception as e:
        print(f"Error extracting frame from {video_file}: {e}")
    return None

def extract_frames_parallel(video_files, timestep, target_dir_images, max_workers=8):
    """
    Extract frames from multiple videos in parallel at the specified timestep.
    Returns list of successfully extracted image paths.
    """
    image_paths = []
    with ThreadPoolExecutor(max_workers=min(len(video_files), max_workers)) as executor:
        future_to_video = {
            executor.submit(extract_single_frame, video_file, timestep, target_dir_images): video_file 
            for video_file in video_files
        }
        
        for future in future_to_video:
            image_path = future.result()
            if image_path:
                image_paths.append(image_path)
    
    return sorted(image_paths)

class CameraImageDataset(Dataset):
    """
    Custom dataset for dynamic-gsplats training/validation.
    Each sample is a dict with keys:
        - 'camtoworld': (4, 4) camera extrinsics
        - 'K': (3, 3) camera intrinsics
        - 'image': (H, W, 3) image (float32, [0, 255] or [0, 1])
        - 'image_id': int (index)
        - 'mask': (H, W) (optional)
        - 'points': (M, 2) (optional, for depth supervision)
        - 'depths': (M,) (optional, for depth supervision)
    """
    def __init__(self, image_paths, extrinsics, intrinsics, mask_paths=None, points_list=None, depths_list=None):
        self.image_paths = list(image_paths)
        self.extrinsics = extrinsics
        # add extra row if the extrinsics are 3x4
        if self.extrinsics.shape[1] == 3:
            self.extrinsics = np.concatenate([self.extrinsics, np.array([[0.0, 0.0, 0.0, 1.0]])[None,:,:].repeat(len(self.extrinsics), axis=0)], axis=1)
        self.intrinsics = intrinsics
        self.mask_paths = list(mask_paths) if mask_paths is not None else None
        self.points_list = points_list
        self.depths_list = depths_list
        assert len(self.image_paths) == len(self.extrinsics) == len(self.intrinsics), "Mismatch in number of images and camera parameters"
        if self.mask_paths is not None:
            assert len(self.mask_paths) == len(self.image_paths), "Mismatch in number of masks and images"
        if self.points_list is not None:
            assert len(self.points_list) == len(self.image_paths), "Mismatch in number of points and images"
        if self.depths_list is not None:
            assert len(self.depths_list) == len(self.image_paths), "Mismatch in number of depths and images"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = load_image(self.image_paths[idx])  # (H, W, 3), float32, [0, 1]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0  # Ensure [0, 255] for compatibility
        # Camera params
        camtoworld = self.extrinsics[idx]
        K = self.intrinsics[idx]
        # Convert to torch tensors
        sample = {
            "camtoworld": torch.from_numpy(camtoworld).float(),  # (4, 4)
            "K": torch.from_numpy(K).float(),                    # (3, 3)
            "image": torch.from_numpy(img).float(),              # (H, W, 3)
            "image_id": torch.tensor(idx, dtype=torch.long),
        }
        # Optional mask
        if self.mask_paths is not None:
            mask = load_image(self.mask_paths[idx])
            if mask.ndim == 3:
                mask = mask[..., 0]  # Use first channel if mask is RGB
            sample["mask"] = torch.from_numpy(mask).bool()
        # Optional points/depths
        if self.points_list is not None:
            points = self.points_list[idx]
            sample["points"] = torch.from_numpy(points).float()
        if self.depths_list is not None:
            depths = self.depths_list[idx]
            sample["depths"] = torch.from_numpy(depths).float()
        return sample

