"""Custom widgets for the cell segmentation application."""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                           QLabel, QSpinBox, QComboBox, QGroupBox, QDialog, QProgressBar, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class FrameNavigator(QGroupBox):
    """Widget for navigating through image frames."""

    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("Frame Navigation", parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        layout = QHBoxLayout()

        # Previous frame button
        self.prev_button = QPushButton("←")
        self.prev_button.clicked.connect(self.previous_frame)
        layout.addWidget(self.prev_button)

        # Frame counter
        self.frame_counter = QSpinBox()
        self.frame_counter.setMinimum(0)
        self.frame_counter.valueChanged.connect(self.frame_changed.emit)
        layout.addWidget(self.frame_counter)

        # Next frame button
        self.next_button = QPushButton("→")
        self.next_button.clicked.connect(self.next_frame)
        layout.addWidget(self.next_button)

        self.setLayout(layout)

    def set_max_frame(self, max_frame: int):
        """Set the maximum frame number."""
        self.frame_counter.setMaximum(max_frame)

    def previous_frame(self):
        """Go to previous frame."""
        current = self.frame_counter.value()
        if current > 0:
            self.frame_counter.setValue(current - 1)

    def next_frame(self):
        """Go to next frame."""
        current = self.frame_counter.value()
        if current < self.frame_counter.maximum():
            self.frame_counter.setValue(current + 1)

class IntensityAdjuster(QGroupBox):
    """Widget for adjusting image intensity display."""

    intensity_changed = pyqtSignal(float, float)  # min, max

    def __init__(self, parent=None):
        super().__init__("Intensity Adjustment", parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()

        # Minimum intensity
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min:"))
        self.min_spin = QSpinBox()
        self.min_spin.setRange(0, 65535)
        self.min_spin.valueChanged.connect(self.update_intensity)
        min_layout.addWidget(self.min_spin)
        layout.addLayout(min_layout)

        # Maximum intensity
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max:"))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 65535)
        self.max_spin.setValue(65535)
        self.max_spin.valueChanged.connect(self.update_intensity)
        max_layout.addWidget(self.max_spin)
        layout.addLayout(max_layout)

        # Auto button
        self.auto_button = QPushButton("Auto")
        self.auto_button.clicked.connect(self.auto_adjust)
        layout.addWidget(self.auto_button)

        self.setLayout(layout)

    def update_intensity(self):
        """Emit signal when intensity values change."""
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()
        if min_val < max_val:
            self.intensity_changed.emit(float(min_val), float(max_val))

    def auto_adjust(self):
        """Auto-adjust intensity based on image data."""
        # To be connected to image statistics
        pass

class ModelParametersWidget(QGroupBox):
    """Widget for adjusting model parameters."""

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__("Model Parameters", parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()

        # Random Forest parameters
        rf_group = QGroupBox("Random Forest")
        rf_layout = QVBoxLayout()

        # Number of estimators
        est_layout = QHBoxLayout()
        est_layout.addWidget(QLabel("Trees:"))
        self.n_estimators = QSpinBox()
        self.n_estimators.setRange(1, 1000)
        self.n_estimators.setValue(100)
        est_layout.addWidget(self.n_estimators)
        rf_layout.addLayout(est_layout)

        # Max depth
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Max Depth:"))
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 100)
        self.max_depth.setValue(10)
        depth_layout.addWidget(self.max_depth)
        rf_layout.addLayout(depth_layout)

        rf_group.setLayout(rf_layout)
        layout.addWidget(rf_group)

        # CNN parameters
        cnn_group = QGroupBox("CNN")
        cnn_layout = QVBoxLayout()

        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.learning_rate = QComboBox()
        self.learning_rate.addItems(['0.1', '0.01', '0.001', '0.0001'])
        self.learning_rate.setCurrentText('0.001')
        lr_layout.addWidget(self.learning_rate)
        cnn_layout.addLayout(lr_layout)

        # Epochs
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("Epochs:"))
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(50)
        epoch_layout.addWidget(self.epochs)
        cnn_layout.addLayout(epoch_layout)

        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group)

        # Apply button
        self.apply_button = QPushButton("Apply Parameters")
        self.apply_button.clicked.connect(self.update_parameters)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def update_parameters(self):
        """Emit signal with current parameter values."""
        parameters = {
            'random_forest': {
                'n_estimators': self.n_estimators.value(),
                'max_depth': self.max_depth.value()
            },
            'cnn': {
                'learning_rate': float(self.learning_rate.currentText()),
                'epochs': self.epochs.value()
            }
        }
        self.parameters_changed.emit(parameters)

class ProgressDialog(QDialog):
    """Dialog showing progress of long operations."""

    def __init__(self, parent=None, title="Processing", label_text="Processing frames..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        # Create layout
        layout = QVBoxLayout(self)

        # Add label
        self.label = QLabel(label_text)
        layout.addWidget(self.label)

        # Add progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Add cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_operation)
        layout.addWidget(self.cancel_button)

        self.cancelled = False

    def cancel_operation(self):
        """Handle cancel button click."""
        self.cancelled = True

    def set_range(self, minimum: int, maximum: int):
        """Set the progress bar range."""
        self.progress_bar.setRange(minimum, maximum)

    def set_value(self, value: int):
        """Update progress bar value."""
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # Ensure UI updates

    def was_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self.cancelled
