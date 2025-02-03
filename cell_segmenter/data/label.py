"""Module for managing segmentation labels and masks."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from PyQt6.QtGui import QColor
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Label:
    """Class to store label information and masks.
    
    Attributes:
        name: Label name
        color: QColor for display
        masks: Dictionary mapping frame numbers to boolean masks
    """
    name: str
    color: QColor
    masks: Dict[int, np.ndarray] = field(default_factory=dict)
    
    def add_mask(self, frame: int, mask: np.ndarray) -> None:
        """Add or update mask for a specific frame.
        
        Args:
            frame: Frame number
            mask: Boolean mask array
        """
        try:
            if not isinstance(mask, np.ndarray):
                raise ValueError("Mask must be a numpy array")
            if mask.dtype != bool:
                mask = mask.astype(bool)
            self.masks[frame] = mask
            logger.debug(f"Added mask for frame {frame} to label '{self.name}'")
        except Exception as e:
            logger.error(f"Error adding mask to label '{self.name}': {str(e)}")
            raise
    
    def get_mask(self, frame: int) -> np.ndarray:
        """Get mask for a specific frame.
        
        Args:
            frame: Frame number
            
        Returns:
            numpy.ndarray: Boolean mask array
        """
        try:
            return self.masks.get(frame, None)
        except Exception as e:
            logger.error(f"Error retrieving mask for frame {frame}: {str(e)}")
            raise
    
    def merge_mask(self, frame: int, new_mask: np.ndarray) -> None:
        """Merge new mask with existing mask using OR operation.
        
        Args:
            frame: Frame number
            new_mask: Boolean mask array to merge
        """
        try:
            if frame in self.masks:
                self.masks[frame] = np.logical_or(self.masks[frame], new_mask)
            else:
                self.add_mask(frame, new_mask)
            logger.debug(f"Merged mask for frame {frame} in label '{self.name}'")
        except Exception as e:
            logger.error(f"Error merging mask for label '{self.name}': {str(e)}")
            raise
    
    def clear_mask(self, frame: int) -> None:
        """Clear mask for a specific frame.
        
        Args:
            frame: Frame number
        """
        try:
            if frame in self.masks:
                del self.masks[frame]
                logger.debug(f"Cleared mask for frame {frame} in label '{self.name}'")
        except Exception as e:
            logger.error(f"Error clearing mask for label '{self.name}': {str(e)}")
            raise