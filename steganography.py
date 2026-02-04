"""Steganography functions for encoding and decoding secret messages in images."""
from PIL import Image
from io import BytesIO
from stegano import lsb


def encode_message(image, secret_message):
    """Encode a secret message into an image using LSB steganography.
    
    Args:
        image: PIL Image object
        secret_message: String message to hide
        
    Returns:
        BytesIO buffer containing the encoded image
    """
    encoded_image = lsb.hide(image, secret_message)
    buffer = BytesIO()
    encoded_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def decode_message(encoded_image):
    """Decode a secret message from an encoded image.
    
    Args:
        encoded_image: PIL Image object containing hidden message
        
    Returns:
        Decoded message string
    """
    return lsb.reveal(encoded_image)
