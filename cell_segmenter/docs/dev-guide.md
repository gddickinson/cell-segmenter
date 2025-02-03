# Developer Guide

## Project Structure

```
cell_segmenter/
├── __init__.py
├── config.py                 # Configuration parameters
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Logging configuration
│   └── image_utils.py       # Image processing utilities
├── gui/
│   ├── __init__.py
│   ├── main_window.py       # Main application window
│   ├── paint_tool.py        # Custom painting widget
│   └── widgets.py           # Custom GUI components
├── models/
│   ├── __init__.py
│   ├── base.py             # Abstract model interface
│   ├── random_forest.py    # Random Forest implementation
│   ├── cnn.py              # CNN implementation
│   └── features.py         # Feature extraction
└── data/
    ├── __init__.py
    └── label.py            # Label management
```

## Development Setup

1. **Create Development Environment**
   ```bash
   conda create -n cell-seg python=3.8
   conda activate cell-seg
   pip install -e .
   pip install -r requirements-dev.txt
   ```

2. **Install Development Tools**
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

## Coding Standards

1. **Code Formatting**
   - Use Black for Python code formatting
   ```bash
   black cell_segmenter/
   ```

2. **Type Hints**
   ```python
   from typing import Dict, List, Optional, Tuple
   
   def process_image(
       image: np.ndarray,
       sigma: float = 1.0
   ) -> Tuple[np.ndarray, Dict[str, float]]:
       # Function implementation
   ```

3. **Documentation**
   - Google style docstrings
   ```python
   def function_name(param1: type, param2: type) -> return_type:
       """Short description.
       
       Longer description if needed.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ExceptionType: Description of when this exception occurs
       """
   ```

## Adding New Features

### 1. New Feature Extractors

```python
class NewFeatureExtractor:
    def __init__(self, params):
        self.params = params
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        # Implementation
        return features
```

### 2. New Model Implementation

1. **Create new model class**
   ```python
   from .base import SegmentationModel
   
   class NewModel(SegmentationModel):
       def __init__(self):
           super().__init__()
           # Model-specific initialization
   
       def train(self, image, labels_dict):
           # Training implementation
   
       def predict(self, image):
           # Prediction implementation
   ```

2. **Register in UI**
   ```python
   # In main_window.py
   self.model_combo.addItems(['Random Forest', 'CNN', 'New Model'])
   ```

### 3. New GUI Components

1. **Create widget class**
   ```python
   from PyQt6.QtWidgets import QWidget
   
   class CustomWidget(QWidget):
       def __init__(self, parent=None):
           super().__init__(parent)
           self.setup_ui()
   
       def setup_ui(self):
           # UI setup code
   ```

2. **Add to main window**
   ```python
   # In main_window.py
   self.custom_widget = CustomWidget()
   layout.addWidget(self.custom_widget)
   ```

## Testing

### 1. Unit Tests

```python
# test_features.py
import pytest
import numpy as np
from cell_segmenter.models.features import FeatureExtractor

def test_feature_extraction():
    extractor = FeatureExtractor()
    image = np.random.rand(100, 100)
    features = extractor.extract_features(image)
    assert features.shape[0] == expected_number_of_features
```

### 2. Integration Tests

```python
# test_segmentation.py
def test_full_pipeline():
    # Setup
    model = RandomForestModel()
    image = load_test_image()
    labels = create_test_labels()
    
    # Train
    model.train(image, labels)
    
    # Predict
    result = model.predict(image)
    
    # Verify
    assert result.shape == image.shape
    assert np.all(np.unique(result) >= 0)
```

## Error Handling

1. **Use Custom Exceptions**
   ```python
   class SegmentationError(Exception):
       """Base exception for segmentation errors."""
       pass
   
   class ModelNotTrainedError(SegmentationError):
       """Raised when attempting prediction with untrained model."""
       pass
   ```

2. **Logging**
   ```python
   from cell_segmenter.utils.logger import setup_logger
   
   logger = setup_logger(__name__)
   
   try:
       # Operation that might fail
       logger.debug("Attempting operation")
   except Exception as e:
       logger.error(f"Operation failed: {str(e)}")
       logger.exception(e)
   ```

## Performance Optimization

1. **Profile Code**
   ```python
   import cProfile
   import pstats
   
   def profile_function():
       profiler = cProfile.Profile()
       profiler.enable()
       # Code to profile
       profiler.disable()
       stats = pstats.Stats(profiler).sort_stats('cumtime')
       stats.print_stats()
   ```

2. **Memory Management**
   ```python
   @contextmanager
   def memory_tracking():
       import psutil
       process = psutil.Process()
       mem_before = process.memory_info().rss
       yield
       mem_after = process.memory_info().rss
       print(f"Memory change: {(mem_after - mem_before) / 1024 / 1024:.2f} MB")
   ```

## Release Process

1. **Version Bumping**
   ```python
   # In __init__.py
   __version__ = '0.1.0'  # Follow semantic versioning
   ```

2. **Release Checklist**
   - Update version number
   - Run full test suite
   - Update documentation
   - Create release notes
   - Tag release in git
   - Build and upload to PyPI

## Contributing Guidelines

1. **Pull Request Process**
   - Fork the repository
   - Create feature branch
   - Make changes
   - Run tests and linting
   - Submit PR with description

2. **Code Review Checklist**
   - Tests included
   - Documentation updated
   - Type hints present
   - Error handling appropriate
   - Performance considered