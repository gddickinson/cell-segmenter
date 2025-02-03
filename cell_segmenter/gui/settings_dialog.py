"""Settings dialog for managing application configuration."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                           QWidget, QLabel, QSpinBox, QDoubleSpinBox,
                           QPushButton, QGroupBox, QFormLayout, QMessageBox,
                           QComboBox, QLineEdit)
from PyQt6.QtCore import Qt
import json
from pathlib import Path
from typing import Dict, Any
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class SettingsDialog(QDialog):
    """Dialog for editing application settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(500, 400)
        
        # Load current settings
        self.current_settings = self.load_settings()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Add tabs for different setting categories
        tabs.addTab(self.create_feature_tab(), "Feature Extraction")
        tabs.addTab(self.create_model_tab(), "Model Parameters")
        tabs.addTab(self.create_gui_tab(), "GUI Settings")
        tabs.addTab(self.create_logging_tab(), "Logging")
        
        layout.addWidget(tabs)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        button_layout.addWidget(apply_button)
        
        layout.addLayout(button_layout)
    
    def create_feature_tab(self) -> QWidget:
        """Create the feature extraction settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Gaussian parameters
        gaussian_group = QGroupBox("Gaussian Features")
        gaussian_layout = QFormLayout()
        
        self.sigma_boxes = []
        current_sigmas = self.current_settings.get('GAUSSIAN_SIGMAS', [1, 2, 4])
        for i, sigma in enumerate(current_sigmas):
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 10.0)
            spin.setSingleStep(0.1)
            spin.setValue(sigma)
            gaussian_layout.addRow(f"Sigma {i+1}:", spin)
            self.sigma_boxes.append(spin)
        
        gaussian_group.setLayout(gaussian_layout)
        layout.addWidget(gaussian_group)
        
        # Edge detection
        edge_group = QGroupBox("Edge Detection")
        edge_layout = QFormLayout()
        
        self.edge_sigma = QDoubleSpinBox()
        self.edge_sigma.setRange(0.1, 10.0)
        self.edge_sigma.setSingleStep(0.1)
        self.edge_sigma.setValue(self.current_settings.get('EDGE_DETECTION_SIGMA', 2))
        edge_layout.addRow("Edge Detection Sigma:", self.edge_sigma)
        
        edge_group.setLayout(edge_layout)
        layout.addWidget(edge_group)
        
        # LBP parameters
        lbp_group = QGroupBox("Local Binary Pattern")
        lbp_layout = QFormLayout()
        
        self.lbp_points = QSpinBox()
        self.lbp_points.setRange(4, 16)
        self.lbp_points.setValue(self.current_settings.get('LBP_POINTS', 8))
        lbp_layout.addRow("Number of Points:", self.lbp_points)
        
        self.lbp_radius = QSpinBox()
        self.lbp_radius.setRange(1, 5)
        self.lbp_radius.setValue(self.current_settings.get('LBP_RADIUS', 1))
        lbp_layout.addRow("Radius:", self.lbp_radius)
        
        lbp_group.setLayout(lbp_layout)
        layout.addWidget(lbp_group)
        
        layout.addStretch()
        return tab
    
    def create_model_tab(self) -> QWidget:
        """Create the model parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Random Forest parameters
        rf_group = QGroupBox("Random Forest")
        rf_layout = QFormLayout()
        
        self.rf_n_estimators = QSpinBox()
        self.rf_n_estimators.setRange(10, 1000)
        self.rf_n_estimators.setSingleStep(10)
        self.rf_n_estimators.setValue(self.current_settings.get('RF_N_ESTIMATORS', 100))
        rf_layout.addRow("Number of Trees:", self.rf_n_estimators)
        
        self.rf_max_depth = QSpinBox()
        self.rf_max_depth.setRange(1, 100)
        self.rf_max_depth.setValue(self.current_settings.get('RF_MAX_DEPTH', 10))
        rf_layout.addRow("Maximum Depth:", self.rf_max_depth)
        
        rf_group.setLayout(rf_layout)
        layout.addWidget(rf_group)
        
        # CNN parameters
        cnn_group = QGroupBox("CNN")
        cnn_layout = QFormLayout()
        
        self.cnn_batch_size = QSpinBox()
        self.cnn_batch_size.setRange(1, 32)
        self.cnn_batch_size.setValue(self.current_settings.get('CNN_BATCH_SIZE', 4))
        cnn_layout.addRow("Batch Size:", self.cnn_batch_size)
        
        self.cnn_learning_rate = QDoubleSpinBox()
        self.cnn_learning_rate.setRange(0.0001, 0.1)
        self.cnn_learning_rate.setDecimals(4)
        self.cnn_learning_rate.setSingleStep(0.0001)
        self.cnn_learning_rate.setValue(self.current_settings.get('CNN_LEARNING_RATE', 0.001))
        cnn_layout.addRow("Learning Rate:", self.cnn_learning_rate)
        
        self.cnn_epochs = QSpinBox()
        self.cnn_epochs.setRange(1, 1000)
        self.cnn_epochs.setValue(self.current_settings.get('CNN_EPOCHS', 50))
        cnn_layout.addRow("Epochs:", self.cnn_epochs)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group)
        
        layout.addStretch()
        return tab
    
    def create_gui_tab(self) -> QWidget:
        """Create the GUI settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Brush settings
        brush_group = QGroupBox("Brush Settings")
        brush_layout = QFormLayout()
        
        self.default_brush_size = QSpinBox()
        self.default_brush_size.setRange(1, 50)
        self.default_brush_size.setValue(self.current_settings.get('DEFAULT_BRUSH_SIZE', 5))
        brush_layout.addRow("Default Brush Size:", self.default_brush_size)
        
        self.overlay_opacity = QDoubleSpinBox()
        self.overlay_opacity.setRange(0.0, 1.0)
        self.overlay_opacity.setSingleStep(0.1)
        self.overlay_opacity.setValue(self.current_settings.get('OVERLAY_OPACITY', 0.5))
        brush_layout.addRow("Overlay Opacity:", self.overlay_opacity)
        
        brush_group.setLayout(brush_layout)
        layout.addWidget(brush_group)
        
        layout.addStretch()
        return tab
    
    def create_logging_tab(self) -> QWidget:
        """Create the logging settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Logging settings
        log_group = QGroupBox("Logging Configuration")
        log_layout = QFormLayout()
        
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level.setCurrentText(self.current_settings.get('LOG_LEVEL', "DEBUG"))
        log_layout.addRow("Log Level:", self.log_level)
        
        self.log_format = QLineEdit()
        self.log_format.setText(self.current_settings.get('LOG_FORMAT', 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        log_layout.addRow("Log Format:", self.log_format)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        return tab
    
    def load_settings(self) -> Dict[str, Any]:
        """Load current settings from config.py."""
        try:
            from .. import config
            return {k: v for k, v in vars(config).items() 
                   if not k.startswith('__')}
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            return {}
    
    def collect_settings(self) -> Dict[str, Any]:
        """Collect current settings from UI components."""
        settings = {
            # Feature extraction
            'GAUSSIAN_SIGMAS': [box.value() for box in self.sigma_boxes],
            'EDGE_DETECTION_SIGMA': self.edge_sigma.value(),
            'LBP_POINTS': self.lbp_points.value(),
            'LBP_RADIUS': self.lbp_radius.value(),
            
            # Random Forest
            'RF_N_ESTIMATORS': self.rf_n_estimators.value(),
            'RF_MAX_DEPTH': self.rf_max_depth.value(),
            
            # CNN
            'CNN_BATCH_SIZE': self.cnn_batch_size.value(),
            'CNN_LEARNING_RATE': self.cnn_learning_rate.value(),
            'CNN_EPOCHS': self.cnn_epochs.value(),
            
            # GUI
            'DEFAULT_BRUSH_SIZE': self.default_brush_size.value(),
            'OVERLAY_OPACITY': self.overlay_opacity.value(),
            
            # Logging
            'LOG_LEVEL': self.log_level.currentText(),
            'LOG_FORMAT': self.log_format.text()
        }
        return settings
    
    def save_settings(self):
        """Save settings to config.py."""
        try:
            settings = self.collect_settings()
            
            # Create backup of current config
            config_path = Path(__file__).parent.parent / 'config.py'
            backup_path = config_path.with_suffix('.py.bak')
            config_path.rename(backup_path)
            
            # Write new config
            with open(config_path, 'w') as f:
                f.write('"""Configuration settings for the cell segmentation application."""\n')
                f.write('import os\n')
                f.write('from pathlib import Path\n\n')
                
                # Project paths (preserve from original)
                f.write('# Project paths\n')
                f.write('PROJECT_ROOT = Path(__file__).parent\n')
                f.write('DATA_DIR = PROJECT_ROOT / "data"\n')
                f.write('MODELS_DIR = PROJECT_ROOT / "models"\n\n')
                f.write('# Create directories if they don\'t exist\n')
                f.write('os.makedirs(DATA_DIR, exist_ok=True)\n')
                f.write('os.makedirs(MODELS_DIR, exist_ok=True)\n\n')
                
                # Write all settings
                for key, value in settings.items():
                    if isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(f'{key} = {value}\n')
            
            logger.info("Settings saved successfully")
            QMessageBox.information(self, "Success", "Settings saved successfully")
            self.accept()
            
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error saving settings: {str(e)}")
    
    def apply_settings(self):
        """Apply settings without closing dialog."""
        try:
            settings = self.collect_settings()
            
            # Update config module
            from .. import config
            for key, value in settings.items():
                setattr(config, key, value)
            
            logger.info("Settings applied")
            QMessageBox.information(self, "Success", "Settings applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error applying settings: {str(e)}")
