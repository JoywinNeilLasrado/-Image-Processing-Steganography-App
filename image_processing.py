"""Image processing functions for various effects and filters."""
import cv2
import numpy as np


def convert_to_grayscale(img):
    """Convert image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_canny(img, threshold1, threshold2):
    """Apply Canny edge detection."""
    return cv2.Canny(img, threshold1, threshold2)


def apply_gaussian_blur(img, kernel_size):
    """Apply Gaussian blur filter."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def apply_thresholding(img, threshold_value):
    """Apply thresholding to image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh


def detect_contours(img):
    """Detect and draw contours on image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


def apply_sharpening(img):
    """Apply sharpening filter to image."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def pencil_sketch(img):
    """Create pencil sketch effect."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    return cv2.divide(gray, 255 - blur, scale=256)


def apply_sepia(img):
    """Apply sepia tone effect."""
    kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)


def rotate_image(img, angle):
    """Rotate image by specified angle."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def cartoonize_image(img):
    """Apply cartoon effect to image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)


def glitch_effect(img):
    """Apply glitch effect to image."""
    rows, cols, _ = img.shape
    shift = cols // 10
    glitch = img.copy()
    for i in range(3):
        channel = glitch[:, :, i]
        offset = np.random.randint(-shift, shift)
        channel[:, np.maximum(offset, 0):] = channel[:, :cols - np.maximum(offset, 0)]
    return glitch


def oil_paint_effect(img):
    """Apply oil painting effect to image."""
    return cv2.stylization(img, sigma_s=60, sigma_r=0.6)


def pixelate(img, pixel_size):
    """Apply pixelation effect to image."""
    h, w = img.shape[:2]
    temp = cv2.resize(img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
