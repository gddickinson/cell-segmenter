"""Feature visualization dialog for examining extracted features."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                           QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
                           QPushButton, QGroupBox, QScrollArea, QWidget)
from PyQt6.QtCore import Qt, pyqtSlot
import numpy as np
import pyqtgraph as pg
from ..models.features import FeatureExtractor
from ..utils.logger import setup_logger
from typing import List, Dict

logger = setup_logger(__name__)

class FeatureVisualizerDialog(QDialog):
    """Dialog for visualizing extracted features with parameter adjustment."""
    
    def __init__(self, image: np.ndarray, parent=None):
        """Initialize the feature visualizer.
        
        Args:
            image: Current frame to extract features from
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Feature Visualization")
        self.resize(1200, 800)
        
        self.image = image
        self.feature_extractor = FeatureExtractor()
        self.image_views: List[pg.ImageView] = []
        self.current_features = None
        
        self.setup_ui()
        self.update_features()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QHBoxLayout(self)
        
        # Left panel - Parameters
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Parameter groups
        left_layout.addWidget(self.create_gaussian_group())
        left_layout.addWidget(self.create_edge_group())
        left_layout.addWidget(self.create_lbp_group())
        
        # Update button
        update_btn = QPushButton("Update Features")
        update_btn.clicked.connect(self.update_features)
        left_layout.addWidget(update_btn)
        
        # Add stretch to keep controls at top
        left_layout.addStretch()
        
        # Scroll area for left panel
        scroll = QScrollArea()
        scroll.setWidget(left_panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        layout.addWidget(scroll)
        
        # Right panel - Feature grid
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        
        # Scroll area for feature grid
        feature_scroll = QScrollArea()
        feature_scroll.setWidget(self.grid_widget)
        feature_scroll.setWidgetResizable(True)
        layout.addWidget(feature_scroll)
        
    def create_gaussian_group(self) -> QGroupBox:
        """Create Gaussian feature parameter group."""
        group = QGroupBox("Gaussian Features")
        layout = QVBoxLayout()
        
        # Sigma controls
        self.sigma_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 10.0)
            spin.setSingleStep(0.1)
            spin.setValue(self.feature_extractor.sigmas[i])
            layout.addWidget(QLabel(f"Sigma {i+1}:"))
            layout.addWidget(spin)
            self.sigma_spins.append(spin)
        
        group.setLayout(layout)
        return group
    
    def create_edge_group(self) -> QGroupBox:
        """Create edge detection parameter group."""
        group = QGroupBox("Edge Detection")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Edge Sigma:"))
        self.edge_sigma = QDoubleSpinBox()
        self.edge_sigma.setRange(0.1, 10.0)
        self.edge_sigma.setSingleStep(0.1)
        self.edge_sigma.setValue(2.0)
        layout.addWidget(self.edge_sigma)
        
        group.setLayout(layout)
        return group
    
    def create_lbp_group(self) -> QGroupBox:
        """Create LBP parameter group."""
        group = QGroupBox("Local Binary Pattern")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Points:"))
        self.lbp_points = QSpinBox()
        self.lbp_points.setRange(4, 16)
        self.lbp_points.setValue(8)
        layout.addWidget(self.lbp_points)
        
        layout.addWidget(QLabel("Radius:"))
        self.lbp_radius = QSpinBox()
        self.lbp_radius.setRange(1, 5)
        self.lbp_radius.setValue(1)
        layout.addWidget(self.lbp_radius)
        
        group.setLayout(layout)
        return group
    
    def update_feature_extractor(self):
        """Update feature extractor with current parameters."""
        self.feature_extractor.sigmas = [spin.value() for spin in self.sigma_spins]
        # Other parameters would be updated here if they were instance variables
        
    def clear_feature_grid(self):
        """Clear all features from the grid."""
        self.image_views.clear()
        
        # Remove all widgets from grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def update_features(self):
        """Extract and display features with current parameters."""
        try:
            logger.debug("Updating feature visualization")
            
            # Update feature extractor parameters
            self.update_feature_extractor()
            
            # Extract features
            features = self.feature_extractor.extract_features(self.image)
            self.current_features = features
            
            # Clear existing views
            self.clear_feature_grid()
            
            # Create feature grid
            feature_names = self.get_feature_names()
            rows = (len(feature_names) + 2) // 3  # 3 columns
            
            for idx, name in enumerate(feature_names):
                row = idx // 3
                col = idx % 3
                
                # Create label
                label = QLabel(name)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.grid_layout.addWidget(label, row*2, col)
                
                # Create image view
                view = pg.ImageView()
                view.ui.roiBtn.hide()
                view.ui.menuBtn.hide()
                view.setImage(features[idx])
                self.grid_layout.addWidget(view, row*2+1, col)
                self.image_views.append(view)
            
            logger.debug(f"Updated {len(feature_names)} features")
            
        except Exception as e:
            logger.error("Error updating features")
            logger.exception(e)
            raise
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for display."""
        names = ["Original Intensity"]
        
        # Gaussian features
        for sigma in self.feature_extractor.sigmas:
            names.extend([
                f"Gaussian (σ={sigma})",
                f"Gradient Magnitude (σ={sigma})",
                f"Laplacian (σ={sigma})"
            ])
        
        # Edge detection features
        names.extend([
            "Sobel Edges",
            "Scharr Edges",
            "Roberts Edges",
            "Prewitt Edges",
            "LBP Texture"
        ])
        
        return names