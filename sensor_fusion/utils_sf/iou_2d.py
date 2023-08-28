from shapely.geometry import Polygon
import numpy as np
from shapely import intersection

def rect_to_polygon(x, y, w, h, heading):
    ### x, y, w, h, heading: float
    dx = w / 2.0
    dy = h / 2.0
    
    dxc = dx * np.cos(heading)
    dyc = dx * np.sin(heading)
    
    dxs = dy * np.sin(heading)
    dys = dy * np.cos(heading)
    
    x1, y1 = x + dxc + dxs, y + dyc - dys
    x2, y2 = x + dxc - dxs, y + dyc + dys
    x3, y3 = x - dxc - dxs, y - dyc + dys
    x4, y4 = x - dxc + dxs, y - dyc - dys
    
    return Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

def compute_2d_iou(rect1, rect2):
    # rect: [x, y, w, h, heading]
    x1, y1, w1, h1, heading1 = rect1.astype(np.float64)
    x2, y2, w2, h2, heading2 = rect2.astype(np.float64)
    
    poly1 = rect_to_polygon(x1, y1, w1, h1, heading1)
    poly2 = rect_to_polygon(x2, y2, w2, h2, heading2)
    intersection = poly1.intersection(poly2, grid_size=0.01).area
    union = poly1.area + poly2.area - intersection
    iou = intersection / union if union > 0 else 0
    
    return iou