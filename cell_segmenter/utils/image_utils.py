"""Utility functions for image processing and handling."""
import numpy as np
from typing import Tuple, Optional, List
from skimage import exposure, filters, util
from scipy import ndimage
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0,1] range.
    
    Args:
        image: Input image array
        
    Returns:
        numpy.ndarray: Normalized image
    """
    try:
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy array")
            
        if image.size == 0:
            raise ValueError("Input array is empty")
            
        min_val = image.min()
        max_val = image.max()
        
        if min_val == max_val:
            logger.warning("Image has no contrast (min == max)")
            return np.zeros_like(image, dtype=float)
            
        normalized = (image - min_val) / (max_val - min_val)
        logger.debug(f"Normalized image range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        return normalized
        
    except Exception as e:
        logger.error("Error normalizing image")
        logger.exception(e)
        raise

def auto_contrast(image: np.ndarray, 
                 p_min: float = 2, 
                 p_max: float = 98) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Automatically adjust image contrast using percentile values.
    
    Args:
        image: Input image array
        p_min: Lower percentile (default: 2)
        p_max: Upper percentile (default: 98)
        
    Returns:
        tuple: (Contrast adjusted image, (min_value, max_value))
    """
    try:
        v_min, v_max = np.percentile(image, (p_min, p_max))
        adjusted = exposure.rescale_intensity(image, in_range=(v_min, v_max))
        logger.debug(f"Auto contrast: range [{v_min:.2f}, {v_max:.2f}]")
        return adjusted, (v_min, v_max)
        
    except Exception as e:
        logger.error("Error adjusting contrast")
        logger.exception(e)
        raise

def enhance_edges(image: np.ndarray, 
                 sigma: float = 2.0,
                 alpha: float = 1.0) -> np.ndarray:
    """Enhance edges in the image using multiple methods.
    
    Args:
        image: Input image array
        sigma: Gaussian sigma for edge detection
        alpha: Edge enhancement strength
        
    Returns:
        numpy.ndarray: Edge-enhanced image
    """
    try:
        # Normalize input
        img_norm = normalize_image(image)
        
        # Compute edges using different methods
        edges_sobel = filters.sobel(img_norm)
        edges_log = ndimage.gaussian_laplace(img_norm, sigma=sigma)
        
        # Combine edge information
        edges_combined = np.sqrt(edges_sobel**2 + edges_log**2)
        
        # Enhance original image
        enhanced = img_norm + alpha * edges_combined
        enhanced = np.clip(enhanced, 0, 1)
        
        logger.debug(f"Enhanced edges with sigma={sigma}, alpha={alpha}")
        return enhanced
        
    except Exception as e:
        logger.error("Error enhancing edges")
        logger.exception(e)
        raise

def compute_frame_statistics(image: np.ndarray) -> dict:
    """Compute various statistics for an image frame.
    
    Args:
        image: Input image array
        
    Returns:
        dict: Dictionary containing image statistics
    """
    try:
        stats = {
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'median': float(np.median(image)),
            'size': image.shape,
            'dtype': str(image.dtype)
        }
        logger.debug(f"Computed frame statistics: {stats}")
        return stats
        
    except Exception as e:
        logger.error("Error computing frame statistics")
        logger.exception(e)
        raise

def validate_tiff_stack(stack: np.ndarray) -> bool:
    """Validate a TIFF stack format and properties.
    
    Args:
        stack: Input image stack array
        
    Returns:
        bool: True if valid, raises exception if invalid
    """
    try:
        if not isinstance(stack, np.ndarray):
            raise TypeError("Input must be a numpy array")
            
        if stack.ndim != 3:
            raise ValueError(f"Expected 3D array (t,y,x), got shape {stack.shape}")
            
        if not np.issubdtype(stack.dtype, np.number):
            raise TypeError(f"Expected numeric dtype, got {stack.dtype}")
            
        if np.any(np.isnan(stack)):
            raise ValueError("Stack contains NaN values")
            
        if np.any(np.isinf(stack)):
            raise ValueError("Stack contains infinite values")
            
        logger.debug(f"Validated TIFF stack: shape={stack.shape}, dtype={stack.dtype}")
        return True
        
    except Exception as e:
        logger.error("Error validating TIFF stack")
        logger.exception(e)
        raise

def extract_sub_stack(stack: np.ndarray,
                     time_range: Optional[Tuple[int, int]] = None,
                     roi: Optional[Tuple[slice, slice]] = None) -> np.ndarray:
    """Extract a portion of a TIFF stack.
    
    Args:
        stack: Input image stack array
        time_range: Optional tuple of (start_frame, end_frame)
        roi: Optional tuple of (y_slice, x_slice) for spatial ROI
        
    Returns:
        numpy.ndarray: Extracted sub-stack
    """
    try:
        validate_tiff_stack(stack)
        
        # Handle time range
        if time_range is not None:
            start_t, end_t = time_range
            if not (0 <= start_t < end_t <= stack.shape[0]):
                raise ValueError(f"Invalid time range: {time_range}")
            t_slice = slice(start_t, end_t)
        else:
            t_slice = slice(None)
            
        # Handle ROI
        if roi is not None:
            y_slice, x_slice = roi
            if not (isinstance(y_slice, slice) and isinstance(x_slice, slice)):
                raise TypeError("ROI must be specified as slice objects")
        else:
            y_slice = slice(None)
            x_slice = slice(None)
            
        # Extract sub-stack
        sub_stack = stack[t_slice, y_slice, x_slice]
        logger.debug(f"Extracted sub-stack with shape {sub_stack.shape}")
        return sub_stack
        
    except Exception as e:
        logger.error("Error extracting sub-stack")
        logger.exception(e)
        raise

def apply_batch_operations(stack: np.ndarray,
                         operations: List[callable]) -> np.ndarray:
    """Apply a sequence of operations to each frame in a stack.
    
    Args:
        stack: Input image stack array
        operations: List of functions to apply to each frame
        
    Returns:
        numpy.ndarray: Processed stack
    """
    try:
        validate_tiff_stack(stack)
        
        # Create output array
        processed = np.zeros_like(stack)
        
        # Process each frame
        for t in range(stack.shape[0]):
            frame = stack[t].copy()
            for op in operations:
                frame = op(frame)
            processed[t] = frame
            
        logger.debug(f"Applied {len(operations)} operations to {stack.shape[0]} frames")
        return processed
        
    except Exception as e:
        logger.error("Error applying batch operations")
        logger.exception(e)
        raise