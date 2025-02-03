# Technical Documentation

## Feature Extraction Pipeline

### Image Pre-processing
1. **Normalization**
   - Min-max scaling to [0,1] range
   - Handles both 8-bit and 16-bit input
   - Robust to outliers using percentile-based normalization

### Feature Categories

1. **Intensity Features**
   ```python
   # Basic intensity
   normalized_image = (image - image.min()) / (image.max() - image.min())
   ```

2. **Multi-scale Gaussian Features**
   - Applied at scales σ = [1, 2, 4]
   ```python
   # For each scale
   gaussian_filtered = filters.gaussian(image, sigma=sigma)
   gradient_x = filters.sobel_h(gaussian_filtered)
   gradient_y = filters.sobel_v(gaussian_filtered)
   gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
   laplacian = filters.laplace(gaussian_filtered)
   ```

3. **Edge Detection Features**
   ```python
   edges_sobel = filters.sobel(image)
   edges_scharr = filters.scharr(image)
   edges_roberts = filters.roberts(image)
   edges_prewitt = filters.prewitt(image)
   ```

4. **Texture Features**
   ```python
   # Local Binary Pattern
   lbp = feature.local_binary_pattern(
       image, P=8, R=1, method='uniform'
   )
   ```

### Feature Selection
- Random Forest importance-based feature selection
- Features ranked by their contribution to classification

## Machine Learning Models

### Random Forest Classifier

1. **Architecture**
   - Number of trees: 100 (configurable)
   - Maximum depth: 10 (configurable)
   - Feature importance tracking

2. **Training Process**
   ```python
   # Training data preparation
   X = []  # Features
   y = []  # Labels
   for label_idx, mask in labels_dict.items():
       if np.any(mask):
           f = features[:, mask].T
           X.append(f)
           y.extend([label_idx] * f.shape[0])
   ```

3. **Performance Optimization**
   - Parallel processing using all available cores
   - Efficient sparse matrix operations
   - Memory-efficient feature extraction

### Convolutional Neural Network

1. **Architecture**
   ```python
   class CNNArchitecture(nn.Module):
       def __init__(self, n_input_channels, n_classes):
           super().__init__()
           self.encoder = nn.Sequential(
               # Initial convolution block
               nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
               nn.BatchNorm2d(32),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(2),
               
               # Second block
               nn.Conv2d(32, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(2),
               
               # Third block
               nn.Conv2d(64, 128, kernel_size=3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(inplace=True)
           )
           
           self.decoder = nn.Sequential(
               # Upsampling
               nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True),
               
               nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
               nn.BatchNorm2d(32),
               nn.ReLU(inplace=True),
               
               nn.Conv2d(32, n_classes, kernel_size=1)
           )
   ```

2. **Training Configuration**
   - Learning rate: 0.001
   - Batch size: 4
   - Epochs: 50
   - Optimizer: Adam
   - Loss: CrossEntropyLoss

## Performance Considerations

### Memory Management
1. **Large Stack Handling**
   - Frame-by-frame processing
   - Memory-mapped TIFF reading
   - Efficient feature caching

2. **Training Data Management**
   ```python
   # Efficient mask storage
   class Label:
       def __init__(self, name, color):
           self.masks = {}  # frame -> mask mapping
   ```

### Processing Speed
1. **Optimization Techniques**
   - NumPy vectorized operations
   - Parallel processing where applicable
   - GPU acceleration for CNN
   - Efficient data structures

2. **Batch Processing**
   ```python
   def apply_batch_operations(stack, operations):
       processed = np.zeros_like(stack)
       for t in range(stack.shape[0]):
           frame = stack[t].copy()
           for op in operations:
               frame = op(frame)
           processed[t] = frame
       return processed
   ```

## UI Architecture

### Main Components
1. **Image Display**
   - PyQtGraph ImageView widget
   - Custom overlay system
   - Real-time update handling

2. **Paint Tool**
   ```python
   class PaintTool(pg.GraphicsObject):
       def paint_at_pos(self, x, y):
           # Create circular brush
           y_idx, x_idx = np.ogrid[-self.brush_size:self.brush_size+1,
                                -self.brush_size:self.brush_size+1]
           dist = np.sqrt(x_idx*x_idx + y_idx*y_idx)
           brush = dist <= self.brush_size/2
   ```

### Event Handling
1. **Mouse Events**
   - Custom coordinate transformation
   - Efficient event propagation
   - Debounced updates

2. **Frame Navigation**
   - Efficient frame switching
   - Memory-efficient display updates

## Testing Framework

### Unit Tests
1. **Feature Extraction Tests**
   ```python
   def test_feature_extraction():
       # Test with synthetic data
       image = np.random.rand(100, 100)
       extractor = FeatureExtractor()
       features = extractor.extract_features(image)
       assert features.shape[0] == expected_feature_count
   ```

2. **Model Tests**
   ```python
   def test_random_forest_training():
       model = RandomForestModel()
       X = np.random.rand(1000, 10)
       y = np.random.randint(0, 2, 1000)
       model.train_on_collected_data(X, y)
       assert model.is_trained
   ```

### Integration Tests
1. **UI Tests**
   - Event handling verification
   - Paint tool accuracy
   - Label management

2. **End-to-End Tests**
   - Full segmentation pipeline
   - Model training and prediction
   - File I/O operations