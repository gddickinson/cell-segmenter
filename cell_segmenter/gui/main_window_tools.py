"""Tools and dialogs of the main window: settings, about, feature viewer, configuration reload.

Mixin of MainWindow (gui/main_window.py); split out to keep files under 500 lines, behaviour unchanged.
"""
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QRadioButton, QGroupBox, QLabel,
                           QFileDialog, QListWidget, QListWidgetItem,
                           QColorDialog, QInputDialog, QSpinBox, QComboBox,
                           QMessageBox, QDialog, QProgressBar, QCheckBox, QMenu, QMenuBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QAction
import pyqtgraph as pg
import tifffile
from typing import Optional, Dict
from cell_segmenter.utils.logger import setup_logger, log_exception
from cell_segmenter.data.label import Label
from cell_segmenter.utils.image_utils import normalize_stack
from cell_segmenter.models.random_forest import RandomForestModel
from cell_segmenter.models.cnn import CNNModel
from .paint_tool import PaintTool
from .widgets import ProgressDialog
from .training_panel import TrainingFeaturePanel
from .. import config

import importlib
from .settings_dialog import SettingsDialog
from .feature_viewer import FeatureVisualizerDialog

logger = setup_logger(__name__)

class ToolsMixin:
    """Tools and dialogs of the main window (mixed into MainWindow)."""

    def show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Reload configuration if settings were saved
            self.load_configuration()

    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(self,
            "About Cell Segmentation Tool",
            """<h3>Cell Segmentation Tool</h3>
            <p>A PyQt-based application for segmenting and analyzing
            microscopy image sequences.</p>
            <p>Version 1.0</p>"""
        )

    def show_feature_viewer(self):
        """Show the feature visualization dialog."""
        if self.image_data is None:
            QMessageBox.warning(self, "Warning",
                "Please load an image first before viewing features.")
            return

        try:
            # Log initial state
            logger.debug(f"Current feature viewer state: {hasattr(self, 'feature_viewer')}")
            if hasattr(self, 'feature_viewer'):
                logger.debug(f"Existing viewer: {self.feature_viewer}")

            # Create new viewer if none exists or if previous was closed
            if not hasattr(self, 'feature_viewer') or not self.feature_viewer:
                logger.debug("Creating new feature viewer")
                self.feature_viewer = FeatureVisualizerDialog(
                    self.image_data[self.current_frame], self)
                logger.debug(f"New viewer created: {self.feature_viewer}")
                logger.debug(f"Viewer title: {self.feature_viewer.windowTitle()}")
                logger.debug(f"Viewer attributes: {dir(self.feature_viewer)}")

            # Show the viewer
            self.feature_viewer.show()
            self.feature_viewer.raise_()  # Bring to front

            # Log final state
            logger.debug(f"Feature viewer displayed: {self.feature_viewer}")
            logger.debug(f"Viewer parent: {self.feature_viewer.parent()}")
            logger.debug(f"Main window feature_viewer attribute: {getattr(self, 'feature_viewer', None)}")

        except Exception as e:
            logger.error("Error showing feature viewer")
            logger.exception(e)
            QMessageBox.critical(self, "Error",
                f"Error showing feature viewer: {str(e)}")
        except Exception as e:
            logger.error("Error showing feature viewer")
            logger.exception(e)
            QMessageBox.critical(self, "Error",
                f"Error showing feature viewer: {str(e)}")

    def load_configuration(self):
        """Reload configuration after settings change."""
        try:
            from .. import config
            importlib.reload(config)

            # Update UI components with new settings
            self.brush_size.setValue(config.DEFAULT_BRUSH_SIZE)

            # Update model parameters if models exist
            if self.current_model and isinstance(self.current_model, RandomForestModel):
                self.current_model.n_estimators = config.RF_N_ESTIMATORS
                self.current_model.max_depth = config.RF_MAX_DEPTH

            logger.debug("Configuration reloaded")

        except Exception as e:
            logger.error(f"Error reloading configuration: {str(e)}")
            QMessageBox.warning(self, "Warning",
                "Error reloading configuration. Some settings may not be applied.")
