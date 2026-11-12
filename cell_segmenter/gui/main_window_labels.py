"""Label management of the main window: add, select, rename, recolour, remove labels; undo and clear training strokes.

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

class LabelActionsMixin:
    """Label management of the main window (mixed into MainWindow)."""

    def add_label(self):
        """Add a new label."""
        try:
            color_dialog = QColorDialog(self)
            if color_dialog.exec():
                color = color_dialog.currentColor()

                name, ok = QInputDialog.getText(
                    self, "Label Name", "Enter name for the label:")

                if ok and name:
                    # Check for duplicate label name
                    existing_names = [l.name for l in self.labels]
                    if name in existing_names:
                        QMessageBox.warning(
                            self, "Duplicate Name",
                            f"A label named '{name}' already exists. Please choose a different name."
                        )
                        return

                    # Check for color collision (same RGB within tolerance)
                    for existing_label in self.labels:
                        ec = existing_label.color
                        if (abs(ec.red() - color.red()) < 10 and
                            abs(ec.green() - color.green()) < 10 and
                            abs(ec.blue() - color.blue()) < 10):
                            QMessageBox.warning(
                                self, "Similar Color",
                                f"The selected color is very similar to label '{existing_label.name}'. "
                                "This may cause confusion in the overlay. Consider choosing a different color."
                            )
                            break

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

    def undo_last_training(self):
        """Remove the last training data added to the current frame."""
        try:
            if not self.labels or self.active_label is None:
                return

            # Remove mask for current frame if it exists
            if self.current_frame in self.active_label.masks:
                del self.active_label.masks[self.current_frame]
                self.update_overlay()
                logger.debug(f"Removed training data for frame {self.current_frame}")

                # Show confirmation
                QMessageBox.information(self, "Success",
                    f"Removed training data for frame {self.current_frame}")
            else:
                QMessageBox.information(self, "Info",
                    "No training data to undo for this frame")

        except Exception as e:
            logger.error("Error undoing last training")
            logger.exception(e)
            QMessageBox.critical(self, "Error", f"Error undoing training: {str(e)}")

    def show_label_context_menu(self, position):
        """Show context menu for label list items.

        Args:
            position: Position where menu should appear
        """
        try:
            item = self.label_list.itemAt(position)
            if item is None:
                return

            menu = QMenu(self)
            rename_action = menu.addAction("Rename")
            change_color_action = menu.addAction("Change Color")
            remove_action = menu.addAction("Remove")

            # Show menu and get selected action
            action = menu.exec(self.label_list.mapToGlobal(position))

            if action == rename_action:
                self.rename_label(item)
            elif action == change_color_action:
                self.change_label_color(item)
            elif action == remove_action:
                self.remove_label(item)

        except Exception as e:
            logger.error("Error showing context menu")
            logger.exception(e)

    def rename_label(self, item):
        """Rename a label.

        Args:
            item: Label list item to rename
        """
        try:
            idx = self.label_list.row(item)
            label = self.labels[idx]

            name, ok = QInputDialog.getText(
                self, "Rename Label",
                "Enter new name:",
                text=label.name
            )

            if ok and name:
                label.name = name
                item.setText(name)
                logger.debug(f"Renamed label to: {name}")

        except Exception as e:
            logger.error("Error renaming label")
            logger.exception(e)

    def change_label_color(self, item):
        """Change a label's color.

        Args:
            item: Label list item to change color for
        """
        try:
            idx = self.label_list.row(item)
            label = self.labels[idx]

            color = QColorDialog.getColor(label.color, self)
            if color.isValid():
                label.color = color
                item.setBackground(color)
                self.update_overlay()
                logger.debug(f"Changed label color for: {label.name}")

        except Exception as e:
            logger.error("Error changing label color")
            logger.exception(e)

    def remove_label(self, item):
        """Remove a label.

        Args:
            item: Label list item to remove
        """
        try:
            idx = self.label_list.row(item)
            label = self.labels[idx]

            reply = QMessageBox.question(
                self,
                "Confirm Removal",
                f"Are you sure you want to remove the label '{label.name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Remove from list and data
                self.label_list.takeItem(idx)
                self.labels.pop(idx)

                # Reset active label if it was removed
                if self.active_label == label:
                    self.active_label = None if not self.labels else self.labels[0]

                # Update display
                self.update_overlay()

                # Reset model training state since labels changed
                if self.current_model is not None:
                    self.current_model.is_trained = False

                logger.info(f"Removed label: {label.name}")

        except Exception as e:
            logger.error("Error removing label")
            logger.exception(e)

    def remove_selected_label(self):
        """Remove currently selected label."""
        try:
            item = self.label_list.currentItem()
            if item is not None:
                self.remove_label(item)
            else:
                QMessageBox.information(self, "Info", "Please select a label to remove")

        except Exception as e:
            logger.error("Error removing selected label")
            logger.exception(e)

    def clear_all_training(self):
        """Clear all training data from all frames."""
        try:
            if not self.labels:
                return

            # Ask for confirmation
            reply = QMessageBox.question(self, "Confirm Clear",
                "Are you sure you want to clear all training data?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                # Clear all masks from all labels
                for label in self.labels:
                    label.masks.clear()

                self.update_overlay()
                logger.info("Cleared all training data")

                # Reset model training state
                if self.current_model is not None:
                    self.current_model.is_trained = False

                QMessageBox.information(self, "Success",
                    "All training data has been cleared")

        except Exception as e:
            logger.error("Error clearing all training data")
            logger.exception(e)
            QMessageBox.critical(self, "Error",
                f"Error clearing training data: {str(e)}")
