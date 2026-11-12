"""Main window implementation for the cell segmentation application."""
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

from .main_window_layout import LayoutMixin
from .main_window_labels import LabelActionsMixin
from .main_window_model import ModelActionsMixin
from .main_window_tools import ToolsMixin

logger = setup_logger(__name__)


class MainWindow(LayoutMixin, LabelActionsMixin, ModelActionsMixin, ToolsMixin, QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Cell Segmentation Tool")

        # Initialize state
        self.image_data = None
        self.current_frame = 0
        self.paint_tool = None
        self.labels = []
        self.active_label = None
        self.overlay = None
        self.current_model = None
        self.feature_viewer = None  # Add feature viewer reference

        try:
            self.setup_ui()
            logger.info("MainWindow initialized successfully")
        except Exception as e:
            log_exception(logger, e, "Error initializing MainWindow")
            raise

    def load_tiff(self):
        """Load a TIFF stack file."""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Open TIFF Stack",
                "",
                "TIFF files (*.tif *.tiff);;All files (*.*)"
            )

            if file_name:
                logger.info(f"Loading TIFF file: {file_name}")
                raw = tifffile.imread(file_name)

                if raw.ndim in (2, 3):
                    # Normalize each frame to [0, 1] (a single page becomes a one-frame stack)
                    self.image_data = normalize_stack(raw)

                    # Display first frame
                    self.image_view.setImage(self.image_data)
                    self.current_frame = 0
                    logger.info(f"Loaded TIFF stack with shape {self.image_data.shape}")

                    # Update paint tool if active
                    if self.paint_tool:
                        self.paint_tool.image_shape = self.image_data[0].shape
                else:
                    QMessageBox.warning(self, "Unsupported TIFF",
                                        f"Expected a single image or a (frames, rows, cols) stack, "
                                        f"got shape {raw.shape}")

        except Exception as e:
            log_exception(logger, e, "Error loading TIFF file")
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def update_brush_size(self, size: int):
        """Update the brush size.

        Args:
            size: New brush size
        """
        try:
            if self.paint_tool:
                self.paint_tool.brush_size = size
                logger.debug(f"Updated brush size to {size}")
        except Exception as e:
            log_exception(logger, e, "Error updating brush size")

    def set_tool_mode(self, mode: str):
        """Set the current tool mode.

        Args:
            mode: Tool mode ('paint' or 'navigate')
        """
        try:
            # Clear paint tool if exists
            if self.paint_tool is not None:
                self.image_view.removeItem(self.paint_tool)
                self.paint_tool = None

            if mode == 'paint' and self.image_data is not None:
                self.paint_tool = PaintTool(self.image_view)
                self.paint_tool.parent_window = self
                self.paint_tool.image_shape = self.image_data[0].shape
                self.paint_tool.brush_size = self.brush_size.value()
                self.image_view.addItem(self.paint_tool)

            self.image_view.view.setMouseMode(self.image_view.view.PanMode)
            logger.debug(f"Set tool mode to {mode}")

        except Exception as e:
            log_exception(logger, e, "Error setting tool mode")

    def frame_changed(self):
        """Handle frame change in the image stack."""
        try:
            if self.image_data is not None:
                new_frame = self.image_view.currentIndex
                if new_frame != self.current_frame:
                    self.current_frame = new_frame
                    self.update_overlay()
                    logger.debug(f"Changed to frame {new_frame}")

        except Exception as e:
            log_exception(logger, e, "Error handling frame change")

    def update_overlay(self):
        """Update the label overlay display."""
        try:
            if self.image_data is None:
                return

            # Create RGBA overlay
            overlay = np.zeros((*self.image_data[0].shape, 4), dtype=np.float32)

            # Add each label's mask
            for label in self.labels:
                if self.current_frame in label.masks:
                    mask = label.masks[self.current_frame]
                    color = label.color
                    # Set color with defined transparency
                    overlay[mask] = [
                        color.red()/255,
                        color.green()/255,
                        color.blue()/255,
                        config.OVERLAY_OPACITY
                    ]

            # Update display
            if self.overlay is not None:
                self.image_view.removeItem(self.overlay)

            self.overlay = pg.ImageItem(overlay)
            self.overlay.setZValue(10)  # Ensure overlay is on top
            self.image_view.addItem(self.overlay)

            logger.debug("Updated overlay display")

        except Exception as e:
            log_exception(logger, e, "Error updating overlay")

    def closeEvent(self, event):
        """Handle application closure."""
        try:
            # Clean up resources
            if self.paint_tool is not None:
                self.image_view.removeItem(self.paint_tool)
            if self.overlay is not None:
                self.image_view.removeItem(self.overlay)
            if self.feature_viewer is not None:
                self.feature_viewer.close()

            logger.info("Application closing")
            event.accept()

        except Exception as e:
            log_exception(logger, e, "Error during application closure")
            event.accept()  # Still close even if there's an error
