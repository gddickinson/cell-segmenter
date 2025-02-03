"""Main window implementation for the cell segmentation application."""
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QRadioButton, QGroupBox, QLabel,
                           QFileDialog, QListWidget, QListWidgetItem,
                           QColorDialog, QInputDialog, QSpinBox, QComboBox,
                           QMessageBox, QDialog, QProgressBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import pyqtgraph as pg
import tifffile
from typing import Optional, Dict
from cell_segmenter.utils.logger import setup_logger, log_exception
from cell_segmenter.data.label import Label
from cell_segmenter.models.random_forest import RandomForestModel
from cell_segmenter.models.cnn import CNNModel
from .paint_tool import PaintTool
from .widgets import ProgressDialog
from .. import config

logger = setup_logger(__name__)

class MainWindow(QMainWindow):
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

        try:
            self.setup_ui()
            logger.info("MainWindow initialized successfully")
        except Exception as e:
            log_exception(logger, e, "Error initializing MainWindow")
            raise

    def setup_ui(self):
        """Set up the user interface."""
        try:
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QHBoxLayout(central_widget)

            # Left panel with image view
            self.setup_image_panel(layout)

            # Right panel with controls
            self.setup_control_panel(layout)

            logger.debug("UI setup completed")

        except Exception as e:
            log_exception(logger, e, "Error setting up UI")
            raise

    def setup_image_panel(self, layout: QHBoxLayout):
        """Set up the image display panel.

        Args:
            layout: Main layout to add the panel to
        """
        try:
            # Image view
            self.image_view = pg.ImageView()
            self.image_view.ui.roiBtn.hide()
            self.image_view.ui.menuBtn.hide()
            layout.addWidget(self.image_view, stretch=2)

            # Connect frame change signal
            self.image_view.sigTimeChanged.connect(self.frame_changed)

        except Exception as e:
            log_exception(logger, e, "Error setting up image panel")
            raise

    def setup_control_panel(self, layout: QHBoxLayout):
        """Set up the control panel.

        Args:
            layout: Main layout to add the panel to
        """
        try:
            control_panel = QWidget()
            control_layout = QVBoxLayout(control_panel)
            layout.addWidget(control_panel)

            # File controls
            self.setup_file_controls(control_layout)

            # Label management
            self.setup_label_controls(control_layout)

            # Brush control
            self.setup_brush_controls(control_layout)

            # Mode selection
            self.setup_mode_controls(control_layout)

            # Model controls
            self.setup_model_controls(control_layout)

            # Status display
            self.pos_label = QLabel("Mouse Position: ")
            control_layout.addWidget(self.pos_label)

            control_layout.addStretch()

        except Exception as e:
            log_exception(logger, e, "Error setting up control panel")
            raise

    def setup_file_controls(self, layout: QVBoxLayout):
        """Set up file loading controls."""
        try:
            self.load_button = QPushButton("Load TIFF Stack")
            self.load_button.clicked.connect(self.load_tiff)
            layout.addWidget(self.load_button)
        except Exception as e:
            log_exception(logger, e, "Error setting up file controls")
            raise

    def setup_label_controls(self, layout: QVBoxLayout):
        """Set up label management controls."""
        try:
            label_group = QGroupBox("Labels")
            label_layout = QVBoxLayout()

            self.label_list = QListWidget()
            self.label_list.itemClicked.connect(self.select_label)
            label_layout.addWidget(self.label_list)

            add_label_btn = QPushButton("Add Label")
            add_label_btn.clicked.connect(self.add_label)
            label_layout.addWidget(add_label_btn)

            label_group.setLayout(label_layout)
            layout.addWidget(label_group)
        except Exception as e:
            log_exception(logger, e, "Error setting up label controls")
            raise

    def setup_brush_controls(self, layout: QVBoxLayout):
        """Set up brush size controls."""
        try:
            brush_group = QGroupBox("Brush Control")
            brush_layout = QHBoxLayout()

            brush_layout.addWidget(QLabel("Size:"))
            self.brush_size = QSpinBox()
            self.brush_size.setRange(1, 50)
            self.brush_size.setValue(config.DEFAULT_BRUSH_SIZE)
            self.brush_size.valueChanged.connect(self.update_brush_size)
            brush_layout.addWidget(self.brush_size)

            brush_group.setLayout(brush_layout)
            layout.addWidget(brush_group)
        except Exception as e:
            log_exception(logger, e, "Error setting up brush controls")
            raise

    def setup_mode_controls(self, layout: QVBoxLayout):
        """Set up mode selection controls."""
        try:
            mode_group = QGroupBox("Mode")
            mode_layout = QVBoxLayout()

            self.paint_mode = QRadioButton("Paint")
            self.nav_mode = QRadioButton("Navigate")
            self.nav_mode.setChecked(True)

            mode_layout.addWidget(self.paint_mode)
            mode_layout.addWidget(self.nav_mode)

            self.paint_mode.toggled.connect(lambda: self.set_tool_mode('paint'))
            self.nav_mode.toggled.connect(lambda: self.set_tool_mode('navigate'))

            mode_group.setLayout(mode_layout)
            layout.addWidget(mode_group)
        except Exception as e:
            log_exception(logger, e, "Error setting up mode controls")
            raise

    def setup_model_controls(self, layout: QVBoxLayout):
        """Set up model selection and training controls."""
        try:
            model_group = QGroupBox("Segmentation")
            model_layout = QVBoxLayout()

            # Model selection
            self.model_combo = QComboBox()
            self.model_combo.addItems(['Random Forest', 'CNN'])
            self.model_combo.currentTextChanged.connect(self.select_model)
            model_layout.addWidget(self.model_combo)

            # Training button
            self.train_button = QPushButton("Train Model")
            self.train_button.clicked.connect(self.train_model)
            model_layout.addWidget(self.train_button)

            # Segment buttons
            self.segment_button = QPushButton("Segment Current Frame")
            self.segment_button.clicked.connect(self.segment_current_frame)
            model_layout.addWidget(self.segment_button)

            self.segment_all_button = QPushButton("Segment All Frames")
            self.segment_all_button.clicked.connect(self.segment_all_frames)
            model_layout.addWidget(self.segment_all_button)

            model_group.setLayout(model_layout)
            layout.addWidget(model_group)
        except Exception as e:
            log_exception(logger, e, "Error setting up model controls")
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
                self.image_data = tifffile.imread(file_name)

                if len(self.image_data.shape) == 3:
                    # Normalize data
                    self.image_data = self.image_data.astype(float)
                    for i in range(len(self.image_data)):
                        frame = self.image_data[i]
                        frame = (frame - frame.min()) / (frame.max() - frame.min())
                        self.image_data[i] = frame

                    # Display first frame
                    self.image_view.setImage(self.image_data)
                    self.current_frame = 0
                    logger.info(f"Loaded TIFF stack with shape {self.image_data.shape}")

                    # Update paint tool if active
                    if self.paint_tool:
                        self.paint_tool.image_shape = self.image_data[0].shape

        except Exception as e:
            log_exception(logger, e, "Error loading TIFF file")
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def add_label(self):
        """Add a new label."""
        try:
            color_dialog = QColorDialog(self)
            if color_dialog.exec():
                color = color_dialog.currentColor()

                name, ok = QInputDialog.getText(
                    self, "Label Name", "Enter name for the label:")

                if ok and name:
                    # Create new label
                    label = Label(name, color)
                    self.labels.append(label)

                    # Add to list widget
                    item = QListWidgetItem(name)
                    item.setBackground(color)
                    self.label_list.addItem(item)

                    # Set as active if first label
                    if len(self.labels) == 1:
                        self.label_list.setCurrentItem(item)
                        self.select_label(item)

                    logger.debug(f"Added new label: {name}")

        except Exception as e:
            log_exception(logger, e, "Error adding label")

    def select_label(self, item: QListWidgetItem):
        """Select a label for painting.

        Args:
            item: Selected list item
        """
        try:
            idx = self.label_list.row(item)
            self.active_label = self.labels[idx]
            self.update_overlay()
            logger.debug(f"Selected label: {self.active_label.name}")

        except Exception as e:
            log_exception(logger, e, "Error selecting label")

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

    def select_model(self, model_name: str):
        """Select the segmentation model.

        Args:
            model_name: Name of the model to use
        """
        try:
            if model_name == 'Random Forest':
                self.current_model = RandomForestModel()
            else:
                self.current_model = CNNModel()
            logger.debug(f"Selected model: {model_name}")

        except Exception as e:
            log_exception(logger, e, "Error selecting model")

    def train_model(self):
        """Train the selected model."""
        try:
            if self.image_data is None or not self.labels:
                QMessageBox.warning(self, "Warning",
                                  "Please load an image and create labels first")
                return

            if self.current_model is None:
                self.select_model(self.model_combo.currentText())

            # Prepare labels dictionary
            labels_dict = {
                label.name: label.masks.get(self.current_frame,
                    np.zeros_like(self.image_data[0], dtype=bool))
                for label in self.labels
            }

            # Train model
            self.current_model.train(self.image_data[self.current_frame],
                                   labels_dict)
            logger.info("Model training completed")

            QMessageBox.information(self, "Success", "Model training completed")

        except Exception as e:
            log_exception(logger, e, "Error training model")
            QMessageBox.critical(self, "Error", f"Training error: {str(e)}")

    def segment_current_frame(self):
        """Segment the current frame using the trained model."""
        try:
            if self.current_model is None or not self.current_model.is_trained:
                QMessageBox.warning(self, "Warning",
                                  "Please train the model first")
                return

            # Run segmentation
            segmentation = self.current_model.predict(
                self.image_data[self.current_frame])

            # Create new label for result
            result_label = Label(
                f"Segmentation_{len(self.labels)}",
                QColor(255, 165, 0)  # Orange
            )
            result_label.masks[self.current_frame] = segmentation > 0
            self.labels.append(result_label)

            # Add to list widget
            item = QListWidgetItem(result_label.name)
            item.setBackground(result_label.color)
            self.label_list.addItem(item)

            self.update_overlay()
            logger.info("Segmentation completed")

        except Exception as e:
            log_exception(logger, e, "Error during segmentation")
            QMessageBox.critical(self, "Error", f"Segmentation error: {str(e)}")

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

    def segment_all_frames(self):
        """Segment all frames in the stack."""
        try:
            if self.current_model is None or not self.current_model.is_trained:
                QMessageBox.warning(self, "Warning",
                                  "Please train the model first")
                return

            if self.image_data is None:
                return

            # Create progress dialog
            progress = ProgressDialog(self,
                                    title="Segmenting Frames",
                                    label_text="Processing frames...")
            progress.set_range(0, len(self.image_data))
            progress.show()

            # Create new label for results
            result_label = Label(
                f"Segmentation_{len(self.labels)}",
                QColor(255, 165, 0)  # Orange
            )

            # Process each frame
            for frame_idx in range(len(self.image_data)):
                if progress.was_cancelled():
                    break

                # Update progress
                progress.set_value(frame_idx)
                progress.label.setText(f"Processing frame {frame_idx + 1} of {len(self.image_data)}")

                # Segment frame
                segmentation = self.current_model.predict(self.image_data[frame_idx])
                result_label.masks[frame_idx] = segmentation > 0

            # Only add the label if we weren't cancelled
            if not progress.was_cancelled():
                self.labels.append(result_label)

                # Add to list widget
                item = QListWidgetItem(result_label.name)
                item.setBackground(result_label.color)
                self.label_list.addItem(item)

                self.update_overlay()
                logger.info("All frames segmented")

            progress.close()

        except Exception as e:
            logger.error("Error segmenting all frames")
            logger.exception(e)
            QMessageBox.critical(self, "Error", f"Error segmenting frames: {str(e)}")

    def closeEvent(self, event):
        """Handle application closure.

        Args:
            event: Close event
        """
        try:
            # Clean up resources
            if self.paint_tool is not None:
                self.image_view.removeItem(self.paint_tool)
            if self.overlay is not None:
                self.image_view.removeItem(self.overlay)

            logger.info("Application closing")
            event.accept()

        except Exception as e:
            log_exception(logger, e, "Error during application closure")
            event.accept()  # Still close even if there's an error
