"""Enhanced feature extraction for cell segmentation."""
import numpy as np
from scipy import ndimage
from skimage import filters, feature, util, exposure
from typing import List, Optional
from cell_segmenter.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureExtractor:
    """Extract features for cell segmentation with enhanced capabilities."""

    def __init__(self):
        """Initialize feature extractor with configuration parameters."""
        self.sigmas = [1, 2, 4]
        self.hessian_sigmas = [1.0, 2.0, 4.0]
        self.gabor_frequencies = [0.1, 0.25, 0.5]
        self.gabor_orientations = [0, 45, 90, 135]
        self.entropy_radius = 5
        self.structure_sigma = 1.0
        logger.debug(f"Initialized enhanced feature extractor")

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
            features.extend(self._extract_gaussian_features(image))
            logger.debug("Added Gaussian features")

            # Edge detection features
            features.extend(self._extract_edge_features(image))
            logger.debug("Added edge features")

            # LBP features
            features.extend(self._extract_lbp_features(image))
            logger.debug("Added LBP features")

            # Hessian features
            features.extend(self._extract_hessian_features(image))
            logger.debug("Added Hessian features")

            # Gabor features
            features.extend(self._extract_gabor_features(image))
            logger.debug("Added Gabor features")

            # Additional features
            features.extend(self._extract_additional_features(image))
            logger.debug("Added additional features")

            # Stack all features
            feature_stack = np.stack(features, axis=0)
            logger.debug(f"Completed feature extraction. Shape: {feature_stack.shape}")

            return feature_stack

        except Exception as e:
            logger.error("Error during feature extraction")
            logger.exception(e)
            raise

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] range."""
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

    def _extract_gaussian_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract multi-scale Gaussian features."""
        features = []

        for sigma in self.sigmas:
            # Gaussian filtered image
            gaussian_filtered = filters.gaussian(image, sigma=sigma)
            features.append(gaussian_filtered)

            # Gradient magnitude
            gradient_x = filters.sobel_h(gaussian_filtered)
            gradient_y = filters.sobel_v(gaussian_filtered)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            features.append(gradient_magnitude)

            # Laplacian
            laplacian = filters.laplace(gaussian_filtered)
            features.append(laplacian)

        return features

    def _extract_edge_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract edge detection features."""
        return [
            filters.sobel(image),
            filters.scharr(image),
            filters.roberts(image),
            filters.prewitt(image)
        ]

    def _extract_lbp_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract Local Binary Pattern features."""
        lbp = feature.local_binary_pattern(
            image, P=8, R=1, method='uniform'
        )
        return [util.img_as_float(lbp)]

    def _extract_hessian_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract Hessian-based features."""
        features = []

        for sigma in self.hessian_sigmas:
            # Compute Hessian matrix components
            Hxx = ndimage.gaussian_filter(image, sigma, order=(2,0))
            Hyy = ndimage.gaussian_filter(image, sigma, order=(0,2))
            Hxy = ndimage.gaussian_filter(image, sigma, order=(1,1))

            # Compute determinant and trace
            det = Hxx * Hyy - Hxy**2
            trace = Hxx + Hyy

            features.extend([det, trace])

        return features

    def _extract_gabor_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract Gabor filter features."""
        features = []

        for frequency in self.gabor_frequencies:
            for theta in self.gabor_orientations:
                theta_rad = np.deg2rad(theta)
                gabor_real = filters.gabor(image, frequency, theta_rad)[0]
                features.append(gabor_real)

        return features

    def _extract_additional_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract additional features."""
        features = []

        # Local entropy
        entropy = filters.rank.entropy(
            util.img_as_ubyte(image),
            np.ones((self.entropy_radius, self.entropy_radius))
        )
        features.append(util.img_as_float(entropy))

        # Structure tensor
        Axx, Axy, Ayy = feature.structure_tensor(
            image, sigma=self.structure_sigma
        )

        # Compute orientation and coherence
        orientation = np.arctan2(2 * Axy, Ayy - Axx) / 2
        coherency = np.sqrt((Axx - Ayy)**2 + 4*Axy**2) / (Axx + Ayy + 1e-10)

        features.extend([orientation, coherency])

        return features
