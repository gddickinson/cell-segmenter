"""Random Forest implementation for cell segmentation."""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Optional
from ..utils.logger import setup_logger
from .. import config
from .base import SegmentationModel

logger = setup_logger(__name__)

class RandomForestModel(SegmentationModel):
    """Random Forest classifier for cell segmentation."""
    
    def __init__(self, 
                n_estimators: int = config.RF_N_ESTIMATORS,
                max_depth: int = config.RF_MAX_DEPTH):
        """Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1  # Use all available cores
        )
        logger.debug(f"Initialized RandomForestModel with {n_estimators} trees")
    
    def train(self, 
             image: np.ndarray,
             labels_dict: Dict[str, np.ndarray],
             **kwargs) -> None:
        """Train the Random Forest classifier.
        
        Args:
            image: Training image
            labels_dict: Dictionary mapping label names to boolean masks
        """
        try:
            logger.info("Starting Random Forest training")
            self.validate_image(image)
            
            # Prepare training data
            X, y = self.prepare_training_data(image, labels_dict)
            
            # Train classifier
            logger.debug("Fitting Random Forest classifier")
            self.classifier.fit(X, y)
            
            # Log feature importances
            importances = self.classifier.feature_importances_
            for idx, importance in enumerate(importances):
                logger.debug(f"Feature {idx} importance: {importance:.4f}")
            
            self.is_trained = True
            logger.info("Random Forest training completed")
            
        except Exception as e:
            logger.error("Error during Random Forest training")
            logger.exception(e)
            raise
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict segmentation using Random Forest.
        
        Args:
            image: Image to segment
            
        Returns:
            numpy.ndarray: Predicted segmentation mask
        """
        try:
            logger.debug("Starting Random Forest prediction")
            self.validate_image(image)
            
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")
            
            # Extract features
            features = self.feature_extractor.extract_features(image)
            features_flat = features.reshape(features.shape[0], -1).T
            
            # Predict
            logger.debug("Running prediction")
            pred = self.classifier.predict(features_flat)
            pred_mask = pred.reshape(image.shape)
            
            logger.debug("Prediction completed")
            return pred_mask
            
        except Exception as e:
            logger.error("Error during Random Forest prediction")
            logger.exception(e)
            raise
    
    def save_model(self, path: str) -> None:
        """Save Random Forest model to disk.
        
        Args:
            path: Save path
        """
        try:
            logger.info(f"Saving model to {path}")
            model_data = {
                'classifier': self.classifier,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, path)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model to {path}")
            logger.exception(e)
            raise
    
    def load_model(self, path: str) -> None:
        """Load Random Forest model from disk.
        
        Args:
            path: Load path
        """
        try:
            logger.info(f"Loading model from {path}")
            model_data = joblib.load(path)
            
            self.classifier = model_data['classifier']
            self.n_estimators = model_data['n_estimators']
            self.max_depth = model_data['max_depth']
            self.is_trained = model_data['is_trained']
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model from {path}")
            logger.exception(e)
            raise