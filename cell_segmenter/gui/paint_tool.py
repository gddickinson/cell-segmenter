"""Paint tool for labeling regions in images."""
import numpy as np
from PyQt6.QtCore import Qt, QRectF
import pyqtgraph as pg
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class PaintTool(pg.GraphicsObject):
    """Custom paint tool for image labeling."""
    
    def __init__(self, image_view: pg.ImageView):
        """Initialize paint tool.
        
        Args:
            image_view: PyQtGraph ImageView widget
        """
        super().__init__()
        self.brush_size = 5
        self.image_shape = None
        self.image_view = image_view
        self.parent_window = None  # Set by MainWindow
        logger.debug("Initialized PaintTool")
    
    def paint(self, p, *args):
        """Paint method required by PyQtGraph."""
        pass
    
    def boundingRect(self) -> QRectF:
        """Define bounding rectangle for the paint tool.
        
        Returns:
            QRectF: Bounding rectangle
        """
        if self.image_shape is None:
            return QRectF(0, 0, 0, 0)
        return QRectF(0, 0, self.image_shape[1], self.image_shape[0])
    
    def mousePressEvent(self, ev):
        """Handle mouse press events.
        
        Args:
            ev: Mouse event
        """
        if ev.button() == Qt.MouseButton.LeftButton:
            try:
                self.handle_mouse_event(ev)
                ev.accept()
            except Exception as e:
                logger.error("Error in mouse press event")
                logger.exception(e)
    
    def mouseMoveEvent(self, ev):
        """Handle mouse move events.
        
        Args:
            ev: Mouse event
        """
        if ev.buttons() & Qt.MouseButton.LeftButton:
            try:
                self.handle_mouse_event(ev)
                ev.accept()
            except Exception as e:
                logger.error("Error in mouse move event")
                logger.exception(e)
    
    def handle_mouse_event(self, ev):
        """Process mouse events for painting.
        
        Args:
            ev: Mouse event
        """
        try:
            if not self.parent_window or not self.parent_window.active_label:
                logger.debug("No active label for painting")
                return
            
            # Get image coordinates
            image_item = self.image_view.getImageItem()
            scene_pos = ev.scenePos()
            item_pos = image_item.mapFromScene(scene_pos)
            
            # Transform coordinates
            x = int(item_pos.x())
            y = int(item_pos.y())
            transformed_x = y
            transformed_y = x
            
            self.paint_at_pos(transformed_x, transformed_y)
            
        except Exception as e:
            logger.error("Error handling mouse event")
            logger.exception(e)
    
    def paint_at_pos(self, x: int, y: int):
        """Paint at the specified position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        try:
            if self.image_shape is None:
                logger.warning("No image shape set for painting")
                return
            
            # Get current label's mask
            current_frame = self.parent_window.current_frame
            if current_frame not in self.parent_window.active_label.masks:
                self.parent_window.active_label.masks[current_frame] = np.zeros(
                    self.image_shape, dtype=bool)
            
            mask = self.parent_window.active_label.masks[current_frame]
            
            # Create circular brush
            y_idx, x_idx = np.ogrid[-self.brush_size:self.brush_size+1,
                                  -self.brush_size:self.brush_size+1]
            dist = np.sqrt(x_idx*x_idx + y_idx*y_idx)
            brush = dist <= self.brush_size/2
            
            # Calculate brush bounds
            y_start = max(0, y - self.brush_size)
            y_end = min(self.image_shape[0], y + self.brush_size + 1)
            x_start = max(0, x - self.brush_size)
            x_end = min(self.image_shape[1], x + self.brush_size + 1)
            
            # Calculate brush array bounds
            brush_y_start = max(0, -(y - self.brush_size))
            brush_y_end = brush.shape[0] - max(0, y_end - self.image_shape[0])
            brush_x_start = max(0, -(x - self.brush_size))
            brush_x_end = brush.shape[1] - max(0, x_end - self.image_shape[1])
            
            # Apply brush
            mask[y_start:y_end, x_start:x_end] |= \
                brush[brush_y_start:brush_y_end,
                      brush_x_start:brush_x_end]
            
            # Update display
            self.parent_window.update_overlay()
            
        except Exception as e:
            logger.error("Error painting at position")
            logger.exception(e)
    
    def set_image_shape(self, shape: tuple):
        """Set the shape of the image being painted.
        
        Args:
            shape: Image shape tuple
        """
        try:
            self.image_shape = shape
            self.prepareGeometryChange()
            logger.debug(f"Set image shape to {shape}")
        except Exception as e:
            logger.error("Error setting image shape")
            logger.exception(e)