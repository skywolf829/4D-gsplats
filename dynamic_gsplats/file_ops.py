import imageio.v3 as iio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
from pathlib import Path

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

