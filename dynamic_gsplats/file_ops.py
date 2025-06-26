import imageio.v3 as iio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms as TF
import subprocess
import tempfile

# ---- Global FFmpeg options ----
FFMPEG_OVERWRITE = True  # Enable `-y`
FFMPEG_QUIET = True      # Enable `-loglevel quiet`

FFMPEG_FLAGS = []
if FFMPEG_OVERWRITE:
    FFMPEG_FLAGS.append("-y")
if FFMPEG_QUIET:
    FFMPEG_FLAGS.extend(["-loglevel", "quiet"])

def load_video(video_path):
    vid = iio.imread(video_path)
    # convert to float32
    vid = vid.astype(np.float32) / 255.0
    return vid

def load_image(image_path):
    img = iio.imread(image_path)
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

def extract_single_frame_save_to_disc(video_file, timestep, target_dir_images):
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

def extract_single_frame(video_file, timestep):
    """
    Extract a single frame from a video file at the specified timestep.
    Memory efficient - only loads the specific frame needed.
    Returns the image path if successful, None otherwise.
    """
    try:
        vs = cv2.VideoCapture(str(video_file))
        vs.set(cv2.CAP_PROP_POS_FRAMES, timestep)
        ret, frame = vs.read()

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vs.release()
        
        if ret:
            return frame
    except Exception as e:
        print(f"Error extracting frame from {video_file}: {e}")
    return None

def extract_frames_parallel(video_files, timestep, max_workers=8):
    """
    Extract frames from multiple videos in parallel at the specified timestep.
    Returns list of successfully extracted image paths.
    """
    imgs = []
    with ThreadPoolExecutor(max_workers=min(len(video_files), max_workers)) as executor:
        future_to_video = {
            executor.submit(extract_single_frame, video_file, timestep): video_file 
            for video_file in video_files
        }
        
        for future in future_to_video:
            img = future.result()
            if img is not None:
                imgs.append(img[:,:,:3])
    
    return imgs

def extract_frames_parallel_save_to_disc(video_files, timestep, target_dir_images, max_workers=8):
    """
    Extract frames from multiple videos in parallel at the specified timestep.
    Returns list of successfully extracted image paths.
    """
    image_paths = []
    with ThreadPoolExecutor(max_workers=min(len(video_files), max_workers)) as executor:
        future_to_video = {
            executor.submit(extract_single_frame_save_to_disc, video_file, timestep, target_dir_images): video_file 
            for video_file in video_files
        }
        
        for future in future_to_video:
            image_path = future.result()
            if image_path:
                image_paths.append(image_path)
    
    return sorted(image_paths)

def get_total_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def get_video_duration(video_path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ], stdout=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def find_shortest_duration_video(video_paths: list[Path]) -> tuple[Path, float]:
    """Returns the (video_path, duration) of the shortest video."""
    durations = [(v, get_video_duration(v)) for v in video_paths]
    return min(durations, key=lambda x: x[1])

def get_video_durations(video_paths: list[Path]) -> list[float]:
    durations = [get_video_duration(v) for v in video_paths]
    return durations

def load_and_preprocess_images_square(image_path_list, target_size=None):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    # get the sizes first
    imgs = []
    max_dim = 0
    
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)
        imgs.append(img)
        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(max_dim, width, height)

    if target_size is None:
        target_size = max_dim

    for image_path, img in zip(image_path_list, imgs):
        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords, target_size

def extract_frame_at_time(video_path: str | Path, timestamp: float, output_path: str | Path | None = None) -> Path:
    """Extract a frame from the video at a specific time (in seconds)."""
    if output_path is None:
        output_path = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
    else:
        output_path = Path(output_path)

    subprocess.run([
        "ffmpeg", *FFMPEG_FLAGS,
        "-ss", f"{timestamp:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        str(output_path)
    ], check=True)
    assert output_path.exists(), f"Error extracting timestep {timestamp} from {video_path}"

    return output_path

def extract_frames_at_time(video_paths: list[Path], offsets: list[float], timestamp: float, output_folder : Path) -> list[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    output_paths = [extract_frame_at_time(video_path, timestamp - offset, 
        output_folder / f"{video_path.name}_{timestamp:0.02f}.png") for video_path, offset in zip(video_paths, offsets)]
    return output_paths

def find_videos(videos_path: Path, extensions=(".mp4", ".mov", ".MP4", ".MOV")) -> list[Path]:
    video_paths = []
    for ext in extensions:
        video_paths.extend(Path(videos_path).glob(f"*{ext}"))
    return sorted(video_paths)

def get_video_resolution(video_path: Path) -> tuple[int, int]:
    """Returns (width, height) of the video using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def get_rotation_metadata(video_path: Path) -> int:
    """Returns rotation metadata in degrees (0, 90, 180, 270), or 0 if missing."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output = result.stdout.strip()
    return int(output) if output.isdigit() else 0