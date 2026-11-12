"""Feature-parameter tab of the feature viewer: parameter widgets, save/load/preset, applying them to the extractor.

Mixin of FeatureVisualizerDialog (gui/feature_viewer.py); split out to keep files under 500 lines, behaviour unchanged.
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                           QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
                           QPushButton, QGroupBox, QScrollArea, QWidget,
                           QCheckBox, QTableWidget, QTableWidgetItem, QFileDialog,
                           QTabWidget, QMessageBox, QToolButton, QSizePolicy, QInputDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon
import numpy as np
import pyqtgraph as pg
import json
from pathlib import Path
from ..models.features import FeatureExtractor
from ..utils.logger import setup_logger
from typing import List, Dict, Optional, Tuple

logger = setup_logger(__name__)

from .feature_viewer_widgets import CollapsibleGroupBox, FeatureParameters  # noqa: F401


class FeatureParamsMixin:
    """Parameters tab and parameter persistence (mixed into FeatureVisualizerDialog)."""

    def create_parameters_tab(self) -> QWidget:
        """Create the parameters management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Parameter presets
        preset_group = QGroupBox("Parameter Presets")
        preset_layout = QHBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Default", "Fine Detail", "Large Scale", "Custom"])
        preset_layout.addWidget(self.preset_combo)

        save_preset = QPushButton("Save")
        save_preset.clicked.connect(self.save_preset)
        preset_layout.addWidget(save_preset)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Parameter groups
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)

        # Gaussian parameters
        gaussian_group = CollapsibleGroupBox("Gaussian Parameters")
        gaussian_layout = QVBoxLayout()
        for name, control in self.parameters['gaussian'].controls.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(name.replace('_', ' ').title()))
            row.addWidget(control)
            gaussian_layout.addLayout(row)
        gaussian_group.content_layout.addLayout(gaussian_layout)
        params_layout.addWidget(gaussian_group)

        # LBP parameters
        lbp_group = CollapsibleGroupBox("LBP Parameters")
        lbp_layout = QVBoxLayout()
        for name, control in self.parameters['lbp'].controls.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(name.replace('_', ' ').title()))
            row.addWidget(control)
            lbp_layout.addLayout(row)
        lbp_group.content_layout.addLayout(lbp_layout)
        params_layout.addWidget(lbp_group)

        # Gabor parameters
        gabor_group = CollapsibleGroupBox("Gabor Parameters")
        gabor_layout = QVBoxLayout()
        for name, control in self.parameters['gabor'].controls.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(name.replace('_', ' ').title()))
            row.addWidget(control)
            gabor_layout.addLayout(row)
        gabor_group.content_layout.addLayout(gabor_layout)
        params_layout.addWidget(gabor_group)

        # Additional parameters
        extra_group = CollapsibleGroupBox("Additional Parameters")
        extra_layout = QVBoxLayout()
        for name, control in self.parameters['extra'].controls.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(name.replace('_', ' ').title()))
            row.addWidget(control)
            extra_layout.addLayout(row)
        extra_group.content_layout.addLayout(extra_layout)
        params_layout.addWidget(extra_group)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidget(params_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Control buttons
        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save Parameters")
        save_btn.clicked.connect(self.save_parameters)
        button_layout.addWidget(save_btn)

        load_btn = QPushButton("Load Parameters")
        load_btn.clicked.connect(self.load_parameters)
        button_layout.addWidget(load_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.update_features)
        button_layout.addWidget(apply_btn)

        layout.addLayout(button_layout)

        return tab

    def save_parameters(self):
        """Save current parameters to file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Parameters", "", "JSON files (*.json)")

            if filename:
                params = {}
                for category, param_group in self.parameters.items():
                    params[category] = param_group.get_values()

                # Add feature selection
                params['features'] = {
                    name: cb.isChecked()
                    for name, cb in self.feature_checkboxes.items()
                }

                with open(filename, 'w') as f:
                    json.dump(params, f, indent=4)

                QMessageBox.information(self, "Success",
                    "Parameters saved successfully!")

        except Exception as e:
            logger.error(f"Error saving parameters: {str(e)}")
            QMessageBox.critical(self, "Error",
                f"Error saving parameters: {str(e)}")

    def load_parameters(self):
        """Load parameters from file."""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Parameters", "", "JSON files (*.json)")

            if filename:
                with open(filename, 'r') as f:
                    params = json.load(f)

                # Update parameters
                for category, values in params.items():
                    if category == 'features':
                        # Update feature selection
                        for name, checked in values.items():
                            if name in self.feature_checkboxes:
                                self.feature_checkboxes[name].setChecked(checked)
                    elif category in self.parameters:
                        # Update parameter values
                        self.parameters[category].set_values(values)

                self.update_features()
                QMessageBox.information(self, "Success",
                    "Parameters loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}")
            QMessageBox.critical(self, "Error",
                f"Error loading parameters: {str(e)}")

    def save_preset(self):
        """Save current parameters as a preset."""
        name, ok = QInputDialog.getText(self,
            "Save Preset", "Enter preset name:")

        if ok and name:
            try:
                # Get current parameters
                preset = {}
                for category, param_group in self.parameters.items():
                    preset[category] = param_group.get_values()

                # Add to presets combo
                self.preset_combo.addItem(name)
                self.preset_combo.setCurrentText(name)

                QMessageBox.information(self, "Success",
                    f"Preset '{name}' saved successfully!")

            except Exception as e:
                logger.error(f"Error saving preset: {str(e)}")
                QMessageBox.critical(self, "Error",
                    f"Error saving preset: {str(e)}")

    def update_feature_extractor(self):
        """Update feature extractor with current parameters."""
        try:
            # Update Gaussian parameters
            if self.parameters['gaussian'].enabled:
                self.feature_extractor.sigmas = [
                    self.parameters['gaussian'].controls[f'sigma_{i+1}'].value()
                    for i in range(3)
                ]

            # Update LBP parameters
            if self.parameters['lbp'].enabled:
                self.feature_extractor.lbp_points = (
                    self.parameters['lbp'].controls['points'].value())
                self.feature_extractor.lbp_radius = (
                    self.parameters['lbp'].controls['radius'].value())

            # Update Gabor parameters
            if self.parameters['gabor'].enabled:
                self.feature_extractor.gabor_frequencies = [
                    self.parameters['gabor'].controls[f'frequency_{i+1}'].value()
                    for i in range(3)
                ]

            # Update additional parameters
            if self.parameters['extra'].enabled:
                self.feature_extractor.entropy_radius = (
                    self.parameters['extra'].controls['entropy_radius'].value())

        except Exception as e:
            logger.error(f"Error updating feature extractor: {str(e)}")
            logger.exception(e)
