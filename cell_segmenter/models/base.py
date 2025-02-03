"""Base model interface for all segmentation models."""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional
from ..utils.logger import setup_logger
from .features import FeatureExtractor

logger = setup_logger(__name__)

class SegmentationModel(ABC):
    """Abstract base class for segmentation models."""
    
    def __init__(self):
        """Initialize base model components."""
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    def prepare_training_data(self, 
                            image: np.ndarray,
                            labels_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training.
        
        Args:
            image: Input image
            labels_dict: Dictionary mapping label names to boolean masks
            
        Returns:
            tuple: (features, labels) prepared for training
        """
        try:
            logger.debug("Preparing training data")
            features = self.feature_extractor.extract_features(image)
            
            X = []
            y = []
            
            for label_idx, (label_name, mask) in enumerate(labels_dict.items()):
                if np.any(mask):
                    logger.debug(f"Processing label '{label_name}' with {np.sum(mask)} pixels")
                    # Get features for labeled pixels
                    f = features[:, mask].T
                    X.append(f)
                    y.extend([label_idx] * f.shape[0])
            
            if not X:
                raise ValueError("No labeled pixels found in any class")
            
            X_array = np.vstack(X)
            y_array = np.array(y)
            
            logger.debug(f"Prepared training data with shape: X={X_array.shape}, y={y_array.shape}")
            return X_array, y_array
            
        except Exception as e:
            logger.error("Error preparing training data")
            logger.exception(e)
            raise
    
    @abstractmethod
    def train(self, 
             image: np.ndarray,
             labels_dict: Dict[str, np.ndarray],
             **kwargs) -> None:
        """Train the model.
        
        Args:
            image: Training image
            labels_dict: Dictionary mapping label names to boolean masks
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict segmentation for new image.
        
        Args:
            image: Image to segment
            
        Returns:
            numpy.ndarray: Predicted segmentation mask
        """
        pass
    
    def save_model(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Save path
        """
        raise NotImplementedError("Save functionality not implemented for this model")
    
    def load_model(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Load path
        """
        raise NotImplementedError("Load functionality not implemented for this model")
    
    def validate_image(self, image: np.ndarray) -> None:
        """Validate input image.
        
        Args:
            image: Input image to validate
            
        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if image.ndim != 2:
            raise ValueError("Image must be 2-dimensional")
        if not np.issubdtype(image.dtype, np.number):
            raise ValueError("Image must contain numeric values")