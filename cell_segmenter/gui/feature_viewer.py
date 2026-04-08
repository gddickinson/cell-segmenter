"""Advanced feature visualization with organized parameter management."""
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

class CollapsibleGroupBox(QGroupBox):
    """A collapsible group box for parameter organization."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setTitle(title)
        self.setCheckable(True)
        self.setChecked(False)

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Widget to hold content
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.main_layout.addWidget(self.content)

        # Connect toggle
        self.toggled.connect(self.on_toggled)
        self.on_toggled(False)

    def on_toggled(self, checked: bool):
        """Handle group box toggle."""
        self.content.setVisible(checked)

class FeatureParameters:
    """Container for feature parameter controls."""

    def __init__(self):
        self.controls = {}
        self.enabled = True

    def add_control(self, name: str, control: QWidget):
        """Add a parameter control."""
        self.controls[name] = control

    def get_values(self) -> dict:
        """Get current parameter values."""
        values = {}
        for name, control in self.controls.items():
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                values[name] = control.value()
            elif isinstance(control, QComboBox):
                values[name] = control.currentText()
        return values

    def set_values(self, values: dict):
        """Set parameter values."""
        for name, value in values.items():
            if name in self.controls:
                control = self.controls[name]
                if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                    control.setValue(value)
                elif isinstance(control, QComboBox):
                    index = control.findText(str(value))
                    if index >= 0:
                        control.setCurrentIndex(index)

class FeatureVisualizerDialog(QDialog):
    """Advanced dialog for feature visualization and parameter management."""

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Visualization")
        self.resize(1200, 800)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)

        # Initialize attributes
        self.image = image
        self.feature_extractor = FeatureExtractor()
        self.image_views = []
        self.current_features = None
        self.feature_stats = {}

        # Feature parameters
        self.parameters = {}
        self.setup_parameters()

        # Create main layout
        self.setup_ui()

        # Initial update
        self.update_features()

    def setup_parameters(self):
        """Initialize feature parameters."""
        # Gaussian parameters
        gaussian_params = FeatureParameters()
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 10.0)
            spin.setSingleStep(0.1)
            spin.setValue(1.0 * (i + 1))
            gaussian_params.add_control(f"sigma_{i+1}", spin)
        self.parameters['gaussian'] = gaussian_params

        # LBP parameters
        lbp_params = FeatureParameters()
        points = QSpinBox()
        points.setRange(4, 16)
        points.setValue(8)
        lbp_params.add_control("points", points)

        radius = QSpinBox()
        radius.setRange(1, 5)
        radius.setValue(1)
        lbp_params.add_control("radius", radius)
        self.parameters['lbp'] = lbp_params

        # Gabor parameters
        gabor_params = FeatureParameters()
        for i in range(3):
            freq = QDoubleSpinBox()
            freq.setRange(0.01, 1.0)
            freq.setDecimals(3)
            freq.setSingleStep(0.05)
            freq.setValue(0.1 * (i + 1))
            gabor_params.add_control(f"frequency_{i+1}", freq)
        self.parameters['gabor'] = gabor_params

        # Additional parameters
        extra_params = FeatureParameters()
        entropy = QSpinBox()
        entropy.setRange(3, 15)
        entropy.setValue(5)
        extra_params.add_control("entropy_radius", entropy)
        self.parameters['extra'] = extra_params

    def setup_ui(self):
        """Set up the user interface."""
        layout = QHBoxLayout(self)

        # Left side - Parameter management
        left_panel = QTabWidget()
        left_panel.setFixedWidth(350)

        # Features tab
        features_tab = self.create_features_tab()
        left_panel.addTab(features_tab, "Features")

        # Parameters tab
        params_tab = self.create_parameters_tab()
        left_panel.addTab(params_tab, "Parameters")

        # Statistics tab
        stats_tab = self.create_statistics_tab()
        left_panel.addTab(stats_tab, "Statistics")

        layout.addWidget(left_panel)

        # Right side - Feature visualization
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)

        # Feature display grid
        display_scroll = QScrollArea()
        display_widget = QWidget()
        self.display_layout = QGridLayout(display_widget)
        display_scroll.setWidget(display_widget)
        display_scroll.setWidgetResizable(True)
        viz_layout.addWidget(display_scroll)

        layout.addWidget(viz_container)

    def create_features_tab(self) -> QWidget:
        """Create the features selection tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Feature categories
        categories = {
            'Basic Features': [
                'Original Intensity',
                'Gaussian Features',
                'Edge Features'
            ],
            'Texture Features': [
                'LBP Features',
                'Gabor Features',
                'Entropy Features'
            ],
            'Advanced Features': [
                'Hessian Features',
                'Structure Tensor'
            ]
        }

        # Create collapsible groups for each category
        self.feature_checkboxes = {}
        for category, features in categories.items():
            group = CollapsibleGroupBox(category)
            group_layout = QVBoxLayout()

            for feature in features:
                cb = QCheckBox(feature)
                cb.setChecked(True)
                cb.stateChanged.connect(self.on_feature_toggled)
                self.feature_checkboxes[feature] = cb
                group_layout.addWidget(cb)

            group.content_layout.addLayout(group_layout)
            layout.addWidget(group)

        # Selection buttons
        button_layout = QHBoxLayout()
        select_all = QPushButton("Select All")
        select_all.clicked.connect(self.select_all_features)
        button_layout.addWidget(select_all)

        select_none = QPushButton("Select None")
        select_none.clicked.connect(self.select_no_features)
        button_layout.addWidget(select_none)

        layout.addLayout(button_layout)

        layout.addStretch()
        return tab

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

    def create_statistics_tab(self) -> QWidget:
        """Create the statistics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels(
            ["Feature", "Min", "Max", "Mean", "Std"])
        layout.addWidget(self.stats_table)

        return tab

    def select_all_features(self):
        """Select all features."""
        for cb in self.feature_checkboxes.values():
            cb.setChecked(True)

    def select_no_features(self):
        """Deselect all features."""
        for cb in self.feature_checkboxes.values():
            cb.setChecked(False)

    def on_feature_toggled(self):
        """Handle feature selection changes."""
        self.update_features()

        # Update parameter visibility
        for feature, cb in self.feature_checkboxes.items():
            if feature == 'Gaussian Features':
                self.parameters['gaussian'].enabled = cb.isChecked()
            elif feature == 'LBP Features':
                self.parameters['lbp'].enabled = cb.isChecked()
            elif feature == 'Gabor Features':
                self.parameters['gabor'].enabled = cb.isChecked()

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

    def update_features(self):
        """Update feature visualization."""
        try:
            logger.debug("Updating features")

            # Clear current display
            while self.display_layout.count():
                item = self.display_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.image_views.clear()

            # Update feature extractor parameters
            self.update_feature_extractor()

            # Extract features
            features = self.feature_extractor.extract_features(self.image)
            self.current_features = features

            # Get selected features
            selected_features = [name for name, cb in self.feature_checkboxes.items()
                               if cb.isChecked()]

            if not selected_features:
                msg = QLabel("No features selected")
                msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.display_layout.addWidget(msg, 0, 0)
                return

            # Create feature grid
            row = 0
            col = 0
            max_cols = 2

            for feature_idx, feature in enumerate(features):
                title = self.get_feature_title(feature_idx)

                # Create feature display
                container = QWidget()
                container_layout = QVBoxLayout(container)

                # Add title and stats
                stats_text = f"Mean: {feature.mean():.3f}\nStd: {feature.std():.3f}"
                label = QLabel(f"{title}\n{stats_text}")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setWordWrap(True)
                container_layout.addWidget(label)

                # Add image view
                view = pg.ImageView()
                view.ui.roiBtn.hide()
                view.ui.menuBtn.hide()
                view.setImage(feature)
                view.setMinimumHeight(200)
                container_layout.addWidget(view)

                self.display_layout.addWidget(container, row, col)
                self.image_views.append(view)

                # Update grid position
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

            # Update statistics
            self.update_statistics()

        except Exception as e:
            logger.error(f"Error updating features: {str(e)}")
            logger.exception(e)

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

    def update_statistics(self):
        """Update statistics table."""
        if self.current_features is None:
            return

        try:
            self.stats_table.setRowCount(len(self.current_features))

            for idx, feature in enumerate(self.current_features):
                stats = {
                    'min': np.min(feature),
                    'max': np.max(feature),
                    'mean': np.mean(feature),
                    'std': np.std(feature)
                }

                title = self.get_feature_title(idx)
                self.stats_table.setItem(idx, 0, QTableWidgetItem(title))
                self.stats_table.setItem(idx, 1, QTableWidgetItem(f"{stats['min']:.3f}"))
                self.stats_table.setItem(idx, 2, QTableWidgetItem(f"{stats['max']:.3f}"))
                self.stats_table.setItem(idx, 3, QTableWidgetItem(f"{stats['mean']:.3f}"))
                self.stats_table.setItem(idx, 4, QTableWidgetItem(f"{stats['std']:.3f}"))

        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
            logger.exception(e)

    def get_feature_title(self, idx: int) -> str:
        """Get the title for a feature."""
        try:
            titles = []

            # Original intensity
            if self.feature_checkboxes['Original Intensity'].isChecked():
                titles.append("Original Intensity")

            # Gaussian features
            if self.feature_checkboxes['Gaussian Features'].isChecked():
                for sigma in [self.parameters['gaussian'].controls[f'sigma_{i+1}'].value()
                            for i in range(3)]:
                    titles.extend([
                        f"Gaussian (σ={sigma:.1f})",
                        f"Gradient Magnitude (σ={sigma:.1f})",
                        f"Laplacian (σ={sigma:.1f})"
                    ])

            # Edge features
            if self.feature_checkboxes['Edge Features'].isChecked():
                titles.extend([
                    "Sobel Edges",
                    "Scharr Edges",
                    "Roberts Edges",
                    "Prewitt Edges"
                ])

            # LBP features
            if self.feature_checkboxes['LBP Features'].isChecked():
                points = self.parameters['lbp'].controls['points'].value()
                radius = self.parameters['lbp'].controls['radius'].value()
                titles.append(f"LBP (P={points}, R={radius})")

            # Gabor features
            if self.feature_checkboxes['Gabor Features'].isChecked():
                for freq in [self.parameters['gabor'].controls[f'frequency_{i+1}'].value()
                           for i in range(3)]:
                    for angle in [0, 45, 90, 135]:
                        titles.append(f"Gabor (f={freq:.3f}, θ={angle}°)")

            # Entropy features
            if self.feature_checkboxes['Entropy Features'].isChecked():
                titles.append(f"Local Entropy (r={self.parameters['extra'].controls['entropy_radius'].value()})")

            # Hessian features (these were missing proper titles)
            if self.feature_checkboxes['Hessian Features'].isChecked():
                for sigma in [self.parameters['gaussian'].controls[f'sigma_{i+1}'].value()
                            for i in range(3)]:
                    titles.extend([
                        f"Hessian Det (σ={sigma:.1f})",
                        f"Hessian Trace (σ={sigma:.1f})",
                        f"Hessian Eigenvalue 1 (σ={sigma:.1f})",
                        f"Hessian Eigenvalue 2 (σ={sigma:.1f})"
                    ])

            # Structure tensor features
            if self.feature_checkboxes['Structure Tensor'].isChecked():
                titles.extend([
                    "Structure Tensor (Orientation)",
                    "Structure Tensor (Coherence)"
                ])

            # Add debug logging
            logger.debug(f"Generated {len(titles)} titles for {self.current_features.shape[0] if self.current_features is not None else 0} features")
            if idx >= len(titles):
                logger.warning(f"Missing title for feature {idx} (total titles: {len(titles)})")

            return titles[idx] if idx < len(titles) else f"Feature {idx+1} (type unknown)"

        except Exception as e:
            logger.error(f"Error generating feature title: {str(e)}")
            return f"Feature {idx+1}"
