"""Training feature selection panel implementation."""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                           QLabel, QScrollArea, QCheckBox, QGroupBox)
from PyQt6.QtCore import Qt

from cell_segmenter.utils.logger import setup_logger, log_exception

logger = setup_logger(__name__)


def find_feature_viewer_by_title(main_window) -> bool:
    """Find feature viewer window by its title."""
    for child in main_window.children():
        if hasattr(child, 'windowTitle') and child.windowTitle() == "Feature Visualization":
            return child
    return None

class TrainingFeaturePanel(QWidget):
    """Panel for selecting features to use in training and segmentation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.feature_items = {}  # Store feature checkboxes/info
        self.setup_ui()

    def setup_ui(self):
        """Set up the panel UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Training Features")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Update button
        update_btn = QPushButton("Get Available Features")
        update_btn.clicked.connect(self.update_feature_list)
        layout.addWidget(update_btn)

        # Feature selection controls
        selection_layout = QHBoxLayout()

        select_all = QPushButton("Select All")
        select_all.clicked.connect(self.select_all_features)
        selection_layout.addWidget(select_all)

        select_none = QPushButton("Select None")
        select_none.clicked.connect(self.select_no_features)
        selection_layout.addWidget(select_none)

        layout.addLayout(selection_layout)

        # Scrollable feature list
        scroll = QScrollArea()
        self.feature_list = QWidget()
        self.feature_layout = QVBoxLayout(self.feature_list)
        scroll.setWidget(self.feature_list)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(scroll)

        # Status
        self.status_label = QLabel("No features configured")
        layout.addWidget(self.status_label)

    def get_main_window(self):
        """Find the MainWindow instance by traversing up the widget hierarchy."""
        parent = self.parent()
        while parent is not None:
            # Check if this is our MainWindow class
            if parent.__class__.__name__ == 'MainWindow':
                return parent
            parent = parent.parent()
        return None

    def update_feature_list(self):
        """Update the list of available features."""
        try:
            # Clear current list
            while self.feature_layout.count():
                item = self.feature_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.feature_items.clear()

            # Get main window instance
            main_window = self.get_main_window()
            logger.debug(f"Found main window: {main_window}")
            logger.debug(f"Main window type: {type(main_window)}")

            if not main_window:
                self.status_label.setText("Cannot access main window")
                logger.warning("Could not find MainWindow instance")
                return

            if not hasattr(main_window, 'feature_viewer'):
                self.status_label.setText("Please open Feature Viewer first")
                logger.warning("Feature viewer attribute not found in main window")
                return

            viewer = main_window.feature_viewer
            if not viewer:
                self.status_label.setText("Please open Feature Viewer first")
                logger.warning("Feature viewer is None")
                return

            logger.debug(f"Found feature viewer: {viewer}")
            logger.debug(f"Feature viewer type: {type(viewer)}")
            logger.debug(f"Feature viewer title: {viewer.windowTitle()}")

            if not hasattr(viewer, 'feature_checkboxes'):
                self.status_label.setText("Feature viewer not properly initialized")
                logger.warning("Feature viewer has no feature_checkboxes")
                return

            viewer = main_window.feature_viewer
            if not viewer:
                self.status_label.setText("Please open Feature Viewer first")
                logger.warning("Feature viewer is None")
                return

            logger.debug(f"Found feature viewer: {viewer}")
            logger.debug(f"Feature viewer type: {type(viewer)}")
            logger.debug(f"Feature viewer title: {viewer.windowTitle()}")
            logger.debug(f"Feature viewer attributes: {dir(viewer)}")

            if not hasattr(viewer, 'feature_checkboxes'):
                self.status_label.setText("Feature viewer not properly initialized")
                logger.warning("Feature viewer has no feature_checkboxes")
                return

            # Rest of the method remains the same...


            logger.debug(f"Accessing feature viewer: {viewer}")

            if not hasattr(viewer, 'feature_checkboxes'):
                self.status_label.setText("Feature viewer not properly initialized")
                logger.warning("Feature viewer has no feature_checkboxes")
                return

            # Create group boxes for feature categories
            basic_group = QGroupBox("Basic Features")
            texture_group = QGroupBox("Texture Features")
            advanced_group = QGroupBox("Advanced Features")

            basic_layout = QVBoxLayout()
            texture_layout = QVBoxLayout()
            advanced_layout = QVBoxLayout()

            feature_count = 0

            # Check each potential feature
            feature_groups = {
                'Basic': ['Original Intensity', 'Gaussian Features', 'Edge Features'],
                'Texture': ['LBP Features', 'Gabor Features', 'Entropy Features'],
                'Advanced': ['Hessian Features', 'Structure Tensor']
            }

            logger.debug(f"Found {len(viewer.feature_checkboxes)} features in viewer")

            for group_name, features in feature_groups.items():
                target_layout = {
                    'Basic': basic_layout,
                    'Texture': texture_layout,
                    'Advanced': advanced_layout
                }[group_name]

                for feature_name in features:
                    if (feature_name in viewer.feature_checkboxes and
                        viewer.feature_checkboxes[feature_name].isChecked()):
                        # Create feature item
                        item = QWidget()
                        item_layout = QVBoxLayout()

                        # Checkbox with feature name
                        checkbox = QCheckBox(feature_name)
                        checkbox.setChecked(True)  # Default to selected
                        item_layout.addWidget(checkbox)

                        # Add parameter info if available
                        params = self.get_feature_params(feature_name, viewer)
                        if params:
                            param_label = QLabel(params)
                            param_label.setStyleSheet("color: gray; margin-left: 20px;")
                            param_label.setWordWrap(True)
                            item_layout.addWidget(param_label)

                        item.setLayout(item_layout)
                        target_layout.addWidget(item)

                        # Store checkbox
                        self.feature_items[feature_name] = checkbox
                        feature_count += 1

                        logger.debug(f"Added feature: {feature_name}")

            # Add groups if they have content
            if basic_layout.count():
                basic_group.setLayout(basic_layout)
                self.feature_layout.addWidget(basic_group)

            if texture_layout.count():
                texture_group.setLayout(texture_layout)
                self.feature_layout.addWidget(texture_group)

            if advanced_layout.count():
                advanced_group.setLayout(advanced_layout)
                self.feature_layout.addWidget(advanced_group)

            self.status_label.setText(f"{feature_count} features available")
            logger.debug(f"Total features added: {feature_count}")

        except Exception as e:
            logger.error(f"Error updating feature list: {str(e)}")
            logger.exception(e)
            self.status_label.setText(f"Error: {str(e)}")

    def get_feature_params(self, feature_name: str, viewer) -> str:
        """Get parameter string for a feature."""
        try:
            if feature_name == 'Gaussian Features':
                sigmas = [spin.value() for spin in viewer.parameters['gaussian'].controls.values()]
                return f"Sigmas: {', '.join(f'{s:.1f}' for s in sigmas)}"

            elif feature_name == 'LBP Features':
                points = viewer.parameters['lbp'].controls['points'].value()
                radius = viewer.parameters['lbp'].controls['radius'].value()
                return f"Points: {points}, Radius: {radius}"

            elif feature_name == 'Gabor Features':
                freqs = [spin.value() for spin in viewer.parameters['gabor'].controls.values()]
                return f"Frequencies: {', '.join(f'{f:.3f}' for f in freqs)}"

            elif feature_name == 'Entropy Features':
                radius = viewer.parameters['extra'].controls['entropy_radius'].value()
                return f"Radius: {radius}"

        except Exception:
            return ""

        return ""

    def select_all_features(self):
        """Select all features."""
        for checkbox in self.feature_items.values():
            checkbox.setChecked(True)

    def select_no_features(self):
        """Deselect all features."""
        for checkbox in self.feature_items.values():
            checkbox.setChecked(False)

    def get_selected_features(self) -> dict:
        """Get dictionary of selected features and their parameters."""
        selected = {}
        for name, checkbox in self.feature_items.items():
            if checkbox.isChecked():
                selected[name] = True
        return selected
