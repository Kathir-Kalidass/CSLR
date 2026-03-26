"""
Image Utilities
Image processing helper functions
"""

import cv2
import numpy as np
from typing import Tuple, Union
import base64


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image (H, W, C)
        size: (width, height)
        interpolation: Interpolation method
    
    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=interpolation)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to image
    
    Args:
        base64_string: Base64 encoded image
    
    Returns:
        Image array (H, W, C)
    """
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    img_bytes = base64.b64decode(base64_string)
    
    # Convert to numpy array
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    
    # Decode image
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        raise ValueError("Failed to decode image")


def encode_image_base64(image: np.ndarray, format: str = '.jpg') -> str:
    """
    Encode image to base64 string
    
    Args:
        image: Image array (H, W, C)
        format: Image format (.jpg, .png, etc.)
    
    Returns:
        Base64 encoded string
    """
    # Convert RGB to BGR if needed
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Encode image
    _, buffer = cv2.imencode(format, image)
    
    # Convert to base64
    base64_string = base64.b64encode(buffer).decode('utf-8')
    
    return base64_string


def crop_center(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop center of image
    
    Args:
        image: Input image
        crop_size: (width, height)
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size
    
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    
    return image[start_y:start_y+crop_h, start_x:start_x+crop_w]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1]
    
    Args:
        image: Input image (uint8)
    
    Returns:
        Normalized image (float32)
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image to [0, 255]
    
    Args:
        image: Normalized image (float32)
    
    Returns:
        Image (uint8)
    """
    return (image * 255).astype(np.uint8)


def apply_augmentation(
    image: np.ndarray,
    flip: bool = False,
    brightness: float = 0.0,
    contrast: float = 1.0
) -> np.ndarray:
    """
    Apply simple augmentations
    
    Args:
        image: Input image
        flip: Horizontal flip
        brightness: Brightness adjustment
        contrast: Contrast adjustment
    
    Returns:
        Augmented image
    """
    result = image.copy()
    
    if flip:
        result = cv2.flip(result, 1)
    
    if brightness != 0.0 or contrast != 1.0:
        result = cv2.convertScaleAbs(result, alpha=contrast, beta=brightness)
    
    return result
