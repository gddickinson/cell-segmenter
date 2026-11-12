"""Small widgets used by the feature viewer: CollapsibleGroupBox and FeatureParameters."""
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
