"""Feature extraction for cell segmentation."""
import numpy as np
from skimage import filters, feature, util
from scipy import ndimage
from typing import List, Optional
from cell_segmenter.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureExtractor:
    """Extract features for cell segmentation."""

    def __init__(self):
        """Initialize feature extractor with configuration parameters."""
        self.sigmas = [1, 2, 4]
        logger.debug(f"Initialized feature extractor")

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract full feature set from image.

        Args:
            image: Input image

        Returns:
            numpy.ndarray: Stack of feature maps
        """
        try:
            logger.debug("Starting feature extraction")
            features = []

            # Basic intensity
            features.append(self._normalize_image(image))
            logger.debug("Added basic intensity feature")

            # Multi-scale Gaussian derivatives
            for sigma in self.sigmas:
                # Gaussian filtered image
                gaussian_filtered = filters.gaussian(image, sigma=sigma, mode='reflect')
                features.append(gaussian_filtered)

                # Gradient using Sobel after Gaussian smoothing
                gradient_x = filters.sobel_h(gaussian_filtered)
                gradient_y = filters.sobel_v(gaussian_filtered)
                gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                features.append(gradient_magnitude)

                # Laplacian
                laplacian = filters.laplace(gaussian_filtered)
                features.append(laplacian)

            logger.debug(f"Added multi-scale features for {len(self.sigmas)} scales")

            # Edge detection features
            edges_sobel = filters.sobel(image)
            features.append(edges_sobel)

            edges_scharr = filters.scharr(image)
            features.append(edges_scharr)

            # Texture features using various filters
            features.append(filters.roberts(image))
            features.append(filters.prewitt(image))

            # Local Binary Pattern for texture
            lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
            features.append(util.img_as_float(lbp))

            # Stack all features
            feature_stack = np.stack(features, axis=0)
            logger.debug(f"Completed feature extraction. Shape: {feature_stack.shape}")

            return feature_stack

        except Exception as e:
            logger.error("Error during feature extraction")
            logger.exception(e)
            raise

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] range.

        Args:
            image: Input image

        Returns:
            numpy.ndarray: Normalized image
        """
        try:
            img_min = image.min()
            img_max = image.max()

            if img_min == img_max:
                return np.zeros_like(image, dtype=float)

            normalized = (image - img_min) / (img_max - img_min)
            return normalized

        except Exception as e:
            logger.error("Error normalizing image")
            logger.exception(e)
            raise
