"""Model actions of the main window: choose the model, train it, segment the current frame or all frames.

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
from ..utils.training_data import gather_training_data, n_classes_painted

import importlib
from .settings_dialog import SettingsDialog
from .feature_viewer import FeatureVisualizerDialog

logger = setup_logger(__name__)

class ModelActionsMixin:
    """Model actions of the main window (mixed into MainWindow)."""

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

            # Get selected training features. The panel lists features only after the Feature Viewer has been
            # opened; until then every feature is used (the models always extract the full feature set).
            selected_features = self.training_panel.get_selected_features()
            if self.training_panel.feature_items and not selected_features:
                QMessageBox.warning(self, "Warning",
                    "Please select features for training")
                return

            logger.info(f"Features used for training: {selected_features}")


            # Strokes of the current frame, or of every frame when "Use training data from all frames" is
            # ticked (Random Forest; the CNN trains on one frame). Segmentation results are never training data.
            all_frames = self.use_all_frames.isChecked()
            if all_frames and not isinstance(self.current_model, RandomForestModel):
                QMessageBox.information(self, "Info",
                    "Training on all frames is available for the Random Forest; the CNN uses the current frame.")
                all_frames = False
            data = gather_training_data(self.labels, self.current_frame, all_frames, self.image_data[0].shape)
            if n_classes_painted(data) < 2:
                where = "any frame" if all_frames else f"frame {self.current_frame}"
                QMessageBox.warning(self, "Warning",
                    f"Paint training strokes for at least two labels on {where} first")
                return

            if all_frames:
                self.current_model.train_frames(
                    [self.image_data[f] for f, _ in data], [d for _, d in data],
                    selected_features=selected_features)
                logger.info(f"Trained on frames {[f for f, _ in data]}")
            else:
                # Train model with selected features
                self.current_model.train(
                    self.image_data[self.current_frame],
                    data[0][1],
                    selected_features=selected_features
                )

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
                QColor(255, 165, 0),  # Orange
                is_result=True
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
                QColor(255, 165, 0),  # Orange
                is_result=True
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
