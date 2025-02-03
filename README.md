# Cell Segmentation Tool

A PyQt-based application for segmenting and analyzing microscopy image sequences, specifically designed for DIC (Differential Interference Contrast) images from TIRF microscopy. The tool provides an interactive interface for training machine learning models to segment cellular structures.

## Features

- Load and view multi-frame TIFF stacks
- Interactive painting tools for creating training data
- Multiple segmentation approaches:
  - Random Forest classifier
  - Convolutional Neural Network (CNN)
- Real-time visualization of segmentation results
- Batch processing capabilities
- Training data management:
  - Multiple label classes
  - Undo/redo functionality
  - Label class editing and removal
  - Option to use training data from single or multiple frames

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:
- PyQt6
- pyqtgraph
- numpy
- scikit-image
- scipy
- torch
- tifffile

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cell-segmentation-tool.git
cd cell-segmentation-tool
```

2. Install in development mode:
```bash
pip install -e .
```

## Usage

### Starting the Application

```bash
python main.py
```

### Basic Workflow

1. **Load Data**
   - Click "Load TIFF Stack" to open your image sequence
   - The application accepts multi-frame TIFF files

2. **Create Labels**
   - Click "Add Label" to create a new label class
   - Select a color for the label
   - Create multiple labels for different cellular structures

3. **Add Training Data**
   - Select a label from the list
   - Switch to "Paint" mode
   - Paint over regions in the image to mark training data
   - Use mouse wheel or brush size control to adjust brush size
   - Navigate through frames using the frame slider

4. **Train Model**
   - Choose between Random Forest and CNN
   - Select whether to use training data from current frame or all frames
   - Click "Train Model" to train the classifier

5. **Segment Images**
   - Use "Segment Current Frame" to segment the current frame
   - Use "Segment All Frames" to process the entire stack
   - Results appear as a new label in the label list

### Advanced Features

- **Training Data Management**
  - "Undo Last" removes the last added training data
  - "Clear All" removes all training data
  - Right-click labels to rename, change color, or remove them

- **Feature Extraction**
  The tool extracts multiple features for segmentation:
  - Basic intensity
  - Multi-scale Gaussian derivatives
  - Edge detection features (Sobel, Scharr, Roberts, Prewitt)
  - Texture features using Local Binary Patterns

## Technical Details

### Image Processing

The tool uses several image processing techniques:
- Multi-scale feature extraction
- Edge enhancement
- Texture analysis
- Intensity normalization

### Machine Learning

Two classification approaches are available:

1. **Random Forest**
   - Fast training and prediction
   - Good for initial segmentation attempts
   - Works well with limited training data

2. **CNN**
   - More powerful for complex patterns
   - Requires more training data
   - Better at capturing spatial relationships

### File Format Support

- Supports TIFF stacks (txy format)
- Handles both 8-bit and 16-bit images
- Automatic normalization of input data

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```
[Citation information to be added]
```

## Troubleshooting

### Common Issues

1. **Image Loading**
   - Ensure TIFF files are in the correct format (txy)
   - Check image bit depth is supported (8 or 16-bit)

2. **Training**
   - If segmentation quality is poor, try adding more training data
   - Ensure training data includes examples of all relevant structures
   - Try using multi-frame training data for better results

3. **Memory Issues**
   - For large image stacks, consider processing fewer frames at a time
   - Close and reopen the application to free memory

### Getting Help

- Open an issue on GitHub for bug reports or feature requests
- Check existing issues for solutions to common problems

## Acknowledgments

This tool was developed for cellular imaging analysis, with inspiration from tools like ilastik and other bioimage analysis software.