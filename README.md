# Image Processing and Steganography Tool

This is an interactive Streamlit application that allows users to perform various image processing tasks, encode/decode secret messages into images (steganography), and detect faces & eyes in images.
## Live Application
     https://image-processing-steganography.streamlit.app/
     
## Features

### 1. **Image Processing**
   - **Operations Supported**:
     - Grayscale Conversion
     - Edge Detection (Canny)
     - Gaussian Blur
     - Thresholding
     - Contour Detection
     - Sharpening
     - Pencil Sketch Effect
     - Sepia Effect
     - Image Rotation
     - Cartoon Effect
     - Glitch Effect
     - Oil Painting Effect
     - Pixelation Effect
   - **Input Options**:
     - Upload an image or capture one using your webcam.
   - **Customizable Parameters**:
     - Adjust thresholds, kernel sizes, angles, pixelation levels, etc., via sliders.

### 2. **Image Steganography**
   - **Encode a Secret Message**:
     - Embed text into an image without visible changes.
     - Download the encoded image after embedding.
   - **Decode a Secret Message**:
     - Extract hidden messages from encoded images.

### 3. **Face & Eye Detection**
   - Detect faces and eyes in uploaded or captured images using OpenCV's Haar Cascade classifiers.
   - Visualize detected faces and eyes with bounding boxes.

## How to Run the Application

1. **Prerequisites**:
   - Python 3.x installed.
   - Install required libraries:
     ```bash
     pip install streamlit opencv-python-headless pillow stegano numpy
     ```

2. **Run the App**:
   - Save the code to a file, e.g., `app.py`.
   - Start the Streamlit server:
     ```bash
     streamlit run app.py
     ```
   - Access the app via the provided local URL in your browser.

## Usage Notes
- Images can be uploaded in formats like JPG, JPEG, or PNG.
- For steganography, only PNG format is supported for encoding due to lossless compression.
- Face and eye detection works best with clear frontal face images.

## Technologies Used
- **Streamlit**: For building the interactive web app.
- **OpenCV**: For image processing and computer vision tasks.
- **Stegano**: For steganography (hiding and revealing messages in images).
- **Pillow**: For handling image formats and conversions.

## Future Enhancements
- Support for additional image processing techniques.
- Improved UI/UX with better layout and design.
- Enhanced steganography methods for larger data embedding.

## License
This project is open-source. Feel free to use, modify, and distribute it as per your needs.

---
