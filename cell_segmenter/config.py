"""Configuration settings for the cell segmentation application."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Logging configuration
LOG_LEVEL = "DEBUG"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "cell_segmentation.log"

# Feature extraction parameters
GAUSSIAN_SIGMAS = [1, 2, 4]
EDGE_DETECTION_SIGMA = 2
LBP_POINTS = 8
LBP_RADIUS = 1

# Model parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# CNN parameters
CNN_BATCH_SIZE = 4
CNN_LEARNING_RATE = 0.001
CNN_EPOCHS = 50

# GUI parameters
DEFAULT_BRUSH_SIZE = 5
OVERLAY_OPACITY = 0.5