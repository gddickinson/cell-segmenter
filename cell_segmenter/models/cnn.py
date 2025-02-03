"""CNN implementation for cell segmentation."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
from ..utils.logger import setup_logger
from .. import config
from .base import SegmentationModel

logger = setup_logger(__name__)

class CNNArchitecture(nn.Module):
    """CNN architecture for cell segmentation."""
    
    def __init__(self, n_input_channels: int, n_classes: int):
        """Initialize CNN architecture.
        
        Args:
            n_input_channels: Number of input feature channels
            n_classes: Number of output classes
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Initial convolution block
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            # Upsampling blocks
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final convolution
            nn.Conv2d(32, n_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CNNModel(SegmentationModel):
    """CNN-based segmentation model."""
    
    def __init__(self,
                batch_size: int = config.CNN_BATCH_SIZE,
                learning_rate: float = config.CNN_LEARNING_RATE,
                n_epochs: int = config.CNN_EPOCHS):
        """Initialize CNN model.
        
        Args:
            batch_size: Training batch size
            learning_rate: Learning rate
            n_epochs: Number of training epochs
        """
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        logger.debug(f"Initialized CNNModel (device: {self.device})")
    
    def _prepare_cnn_data(self,
                        image: np.ndarray,
                        labels_dict: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for CNN training.
        
        Args:
            image: Input image
            labels_dict: Dictionary mapping label names to boolean masks
            
        Returns:
            tuple: (input_tensor, target_tensor)
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(image)
            
            # Prepare target
            n_classes = len(labels_dict)
            target = np.zeros((n_classes, *image.shape), dtype=np.float32)
            for i, mask in enumerate(labels_dict.values()):
                target[i][mask] = 1
            
            # Convert to PyTorch tensors
            X = torch.FloatTensor(features).unsqueeze(0)
            y = torch.FloatTensor(target).unsqueeze(0)
            
            return X.to(self.device), y.to(self.device)
            
        except Exception as e:
            logger.error("Error preparing CNN data")
            logger.exception(e)
            raise
    
    def train(self,
             image: np.ndarray,
             labels_dict: Dict[str, np.ndarray],
             **kwargs) -> None:
        """Train the CNN model.
        
        Args:
            image: Training image
            labels_dict: Dictionary mapping label names to boolean masks
        """
        try:
            logger.info("Starting CNN training")
            self.validate_image(image)
            
            # Initialize model if needed
            if self.model is None:
                n_features = len(self.feature_extractor.extract_features(image))
                n_classes = len(labels_dict)
                self.model = CNNArchitecture(n_features, n_classes).to(self.device)
                logger.debug(f"Initialized CNN architecture with {n_features} input channels")
            
            # Prepare data
            X, y = self._prepare_cnn_data(image, labels_dict)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(self.n_epochs):
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(X)
                loss = criterion(output, y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch [{epoch+1}/{self.n_epochs}], Loss: {loss.item():.4f}")
            
            self.is_trained = True
            logger.info("CNN training completed")
            
        except Exception as e:
            logger.error("Error during CNN training")
            logger.exception(e)
            raise
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict segmentation using CNN.
        
        Args:
            image: Image to segment
            
        Returns:
            numpy.ndarray: Predicted segmentation mask
        """
        try:
            logger.debug("Starting CNN prediction")
            self.validate_image(image)
            
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")
            
            # Extract features and prepare input
            features = self.feature_extractor.extract_features(image)
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(X)
                pred = output.argmax(dim=1).squeeze().cpu().numpy()
            
            logger.debug("Prediction completed")
            return pred
            
        except Exception as e:
            logger.error("Error during CNN prediction")
            logger.exception(e)
            raise
    
    def save_model(self, path: str) -> None:
        """Save CNN model to disk.
        
        Args:
            path: Save path
        """
        try:
            logger.info(f"Saving model to {path}")
            if self.model is None:
                raise ValueError("No model to save")
                
            save_dict = {
                'model_state': self.model.state_dict(),
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'n_epochs': self.n_epochs,
                'is_trained': self.is_trained
            }
            torch.save(save_dict, path)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model to {path}")
            logger.exception(e)
            raise
    
    def load_model(self, path: str) -> None:
        """Load CNN model from disk.
        
        Args:
            path: Load path
        """
        try:
            logger.info(f"Loading model from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model parameters
            self.batch_size = checkpoint['batch_size']
            self.learning_rate = checkpoint['learning_rate']
            self.n_epochs = checkpoint['n_epochs']
            self.is_trained = checkpoint['is_trained']
            
            # Initialize and load model state
            if self.model is not None:
                self.model.load_state_dict(checkpoint['model_state'])
                logger.info("Model loaded successfully")
            else:
                logger.warning("Model architecture not initialized. Call train() first.")
            
        except Exception as e:
            logger.error(f"Error loading model from {path}")
            logger.exception(e)
            raise