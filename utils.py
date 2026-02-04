"""Utility functions for image conversions."""
import cv2
import numpy as np
from PIL import Image


def pil_to_cv(image):
    """Convert PIL image to OpenCV format."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv_to_pil(image):
    """Convert OpenCV image to PIL format."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if len(image.shape) == 3 else Image.fromarray(image)
