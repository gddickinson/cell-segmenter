"""Widget layout of the main window: image panel, control panel groups and the menu bar.

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

class LayoutMixin:
    """Widget layout of the main window (mixed into MainWindow)."""

    def setup_ui(self):
        """Set up the user interface."""
        try:
            # Set up menu bar
            self.setup_menu_bar()

            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QHBoxLayout(central_widget)

            # Left panel with controls
            control_panel = QWidget()
            control_layout = QVBoxLayout(control_panel)

            # Add standard controls
            self.setup_file_controls(control_layout)
            self.setup_label_controls(control_layout)
            self.setup_brush_controls(control_layout)
            self.setup_mode_controls(control_layout)
            self.setup_model_controls(control_layout)

            # Add position label
            self.pos_label = QLabel("Mouse Position: ")
            control_layout.addWidget(self.pos_label)

            control_layout.addStretch()
            layout.addWidget(control_panel)

            # Middle panel with image view
            self.setup_image_panel(layout)

            # Right panel for training features
            self.training_panel = TrainingFeaturePanel(self)
            layout.addWidget(self.training_panel)

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

            # Label list with context menu
            self.label_list = QListWidget()
            self.label_list.itemClicked.connect(self.select_label)
            self.label_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.label_list.customContextMenuRequested.connect(self.show_label_context_menu)
            label_layout.addWidget(self.label_list)

            # Buttons for label management
            button_layout = QHBoxLayout()

            add_label_btn = QPushButton("Add Label")
            add_label_btn.clicked.connect(self.add_label)
            button_layout.addWidget(add_label_btn)

            remove_label_btn = QPushButton("Remove Label")
            remove_label_btn.clicked.connect(self.remove_selected_label)
            button_layout.addWidget(remove_label_btn)

            undo_label_btn = QPushButton("Undo Last")
            undo_label_btn.clicked.connect(self.undo_last_training)
            button_layout.addWidget(undo_label_btn)

            clear_labels_btn = QPushButton("Clear All")
            clear_labels_btn.clicked.connect(self.clear_all_training)
            button_layout.addWidget(clear_labels_btn)

            label_layout.addLayout(button_layout)

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

            # Training data selector
            self.use_all_frames = QCheckBox("Use training data from all frames")
            model_layout.addWidget(self.use_all_frames)

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

    def setup_menu_bar(self):
        """Set up the main window menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        load_action = QAction("&Load TIFF Stack", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_tiff)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")

        undo_action = QAction("&Undo Last", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_last_training)
        edit_menu.addAction(undo_action)

        clear_action = QAction("&Clear All Training", self)
        clear_action.triggered.connect(self.clear_all_training)
        edit_menu.addAction(clear_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")

        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)

        features_action = QAction("&Feature Viewer...", self)
        features_action.triggered.connect(self.show_feature_viewer)
        tools_menu.addAction(features_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
