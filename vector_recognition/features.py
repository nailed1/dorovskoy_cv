import numpy as np
from skimage.measure import label


def count_holes(region):
    """Count the number of holes in a region."""
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1


def extractor(region):
    """Extract features from a region for classification.
    
    Returns a normalized feature vector including:
    - Area ratio (fill factor)
    - Normalized centroid coordinates
    - Number of holes (topological feature)
    - Line structure indicators (vertical/horizontal strokes)
    - Shape descriptors (eccentricity, aspect ratio, solidity)
    - Perimeter-based compactness
    """
    image = region.image
    h, w = image.shape
    area = region.area
    size = h * w
    
    # Normalized centroid
    cy, cx = region.centroid_local
    cy_norm = cy / h
    cx_norm = cx / w
    
    # Area fill ratio
    fill_ratio = area / size
    
    # Number of holes (topological feature)
    holes = count_holes(region)
    
    # Line structure: count rows/cols that are fully filled
    # Vertical lines: columns filled from top to bottom
    vertical_lines = (np.sum(image, axis=0) == h).sum()
    # Horizontal lines: rows filled from left to right
    horizontal_lines = (np.sum(image, axis=1) == w).sum()
    
    # Shape descriptors
    eccentricity = region.eccentricity
    aspect_ratio = h / (w + 1e-6)  # Avoid division by zero
    
    # Compactness: how close to a circle (4*pi*area/perimeter^2)
    perimeter = region.perimeter
    compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
    
    # Solidity: area / convex_area (how convex the shape is)
    solidity = area / (region.convex_area + 1e-6)
    
    # Extent: area / bounding_box_area
    extent = area / (region.bbox_area + 1e-6)
    
    return np.array([
        fill_ratio,
        cy_norm,
        cx_norm,
        holes,
        vertical_lines,
        horizontal_lines,
        eccentricity,
        aspect_ratio,
        compactness,
        solidity,
        extent
    ])