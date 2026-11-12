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

from .feature_viewer_widgets import CollapsibleGroupBox, FeatureParameters  # noqa: F401 (re-exported)
from .feature_viewer_params import FeatureParamsMixin

logger = setup_logger(__name__)


class FeatureVisualizerDialog(FeatureParamsMixin, QDialog):
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
