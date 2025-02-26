import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
from stegano import lsb  # For image steganography

# Convert PIL image to OpenCV format
def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Convert OpenCV image to PIL format
def cv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if len(image.shape) == 3 else Image.fromarray(image)

# Image processing functions
def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_canny(img, threshold1, threshold2):
    return cv2.Canny(img, threshold1, threshold2)

def apply_gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_thresholding(img, threshold_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def detect_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output

def apply_sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    return cv2.divide(gray, 255 - blur, scale=256)

def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def cartoonize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def glitch_effect(img):
    rows, cols, _ = img.shape
    shift = cols // 10
    glitch = img.copy()
    for i in range(3):
        channel = glitch[:, :, i]
        offset = np.random.randint(-shift, shift)
        channel[:, np.maximum(offset, 0):] = channel[:, :cols - np.maximum(offset, 0)]
    return glitch

def oil_paint_effect(img):
    return cv2.stylization(img, sigma_s=60, sigma_r=0.6)

def pixelate(img, pixel_size):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

# Encode Message into Image (Steganography)
def encode_message(image, secret_message):
    encoded_image = lsb.hide(image, secret_message)
    buffer = BytesIO()
    encoded_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# Decode Message from Image (Steganography)
def decode_message(encoded_image):
    return lsb.reveal(encoded_image)

# Face and Eye Detection
def detect_faces_and_eyes(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return img


# Streamlit App
st.title("📷 Advanced Image Processing, Steganography Tool with Face & Eye Detection ")
tab1, tab2, tab3 = st.tabs(["🎨 Image Processing", "🔐 Image Steganography","👁️ Face & Eye Detection"])

with tab1:
    st.subheader("Upload an Image")
    option = st.selectbox("Choose Image Source", ["Upload Image", "Use Webcam"], key="tab1_selectbox")

    if option == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="tab1_file_uploader")
        if uploaded_image:
            image = Image.open(uploaded_image)
    elif option == "Use Webcam":
        webcam_image = st.camera_input("Take a picture", key="tab1_camera_input")
        if webcam_image:
            image = Image.open(webcam_image)

    if 'image' in locals():
        st.image(image, caption="Selected Image", use_container_width=True)
        
        operation = st.selectbox("Select an Operation", [
            "Grayscale", "Edge Detection", "Gaussian Blur", "Thresholding",
            "Contour Detection", "Sharpening", "Pencil Sketch", "Sepia Effect",
            "Rotate Image", "Cartoon Effect", "Glitch Effect", "Oil Painting Effect", "Pixelation Effect"
        ], key="tab1_operation_selectbox")
        
        # Operation-specific inputs
        threshold1 = st.slider("Threshold 1 (Edge Detection)", 0, 255, 100, key="tab1_threshold1_slider")
        threshold2 = st.slider("Threshold 2 (Edge Detection)", 0, 255, 200, key="tab1_threshold2_slider")
        kernel_size = st.slider("Kernel Size (Gaussian Blur)", 1, 15, 5, step=2, key="tab1_kernel_size_slider")
        threshold_value = st.slider("Threshold Value", 0, 255, 127, key="tab1_threshold_value_slider")
        angle = st.slider("Rotation Angle", -180, 180, 0, step=5, key="tab1_angle_slider")
        pixel_size = st.slider("Pixelation Size", 1, 50, 10, key="tab1_pixel_size_slider")

        if st.button("Process Image", key="tab1_process_button"):
            img_cv = pil_to_cv(image)

            if operation == "Grayscale":
                processed_img = convert_to_grayscale(img_cv)
            elif operation == "Edge Detection":
                processed_img = apply_canny(img_cv, threshold1, threshold2)
            elif operation == "Gaussian Blur":
                processed_img = apply_gaussian_blur(img_cv, kernel_size)
            elif operation == "Thresholding":
                processed_img = apply_thresholding(img_cv, threshold_value)
            elif operation == "Contour Detection":
                processed_img = detect_contours(img_cv)
            elif operation == "Sharpening":
                processed_img = apply_sharpening(img_cv)
            elif operation == "Pencil Sketch":
                processed_img = pencil_sketch(img_cv)
            elif operation == "Sepia Effect":
                processed_img = apply_sepia(img_cv)
            elif operation == "Rotate Image":
                processed_img = rotate_image(img_cv, angle)
            elif operation == "Cartoon Effect":
                processed_img = cartoonize_image(img_cv)
            elif operation == "Glitch Effect":
                processed_img = glitch_effect(img_cv)
            elif operation == "Oil Painting Effect":
                processed_img = oil_paint_effect(img_cv)
            elif operation == "Pixelation Effect":
                processed_img = pixelate(img_cv, pixel_size)

            st.image(cv_to_pil(processed_img), caption="Processed Image", use_container_width=True)

with tab2:
    st.subheader("Steganography")
    option = st.selectbox("Choose Image Source", ["Upload Image", "Use Webcam"], key="tab2_selectbox")

    if option == "Upload Image":
        steg_image = st.file_uploader("Upload an Image for Steganography", type=["png"], key="tab2_file_uploader")
        if steg_image:
            image = Image.open(steg_image)
    elif option == "Use Webcam":
        webcam_image = st.camera_input("Take a picture", key="tab2_camera_input")
        if webcam_image:
            image = Image.open(webcam_image)

    if 'image' in locals():
        st.image(image, caption="Selected Image", use_container_width=True)
        secret_message = st.text_input("Enter a Secret Message", key="tab2_secret_message_input")

        if st.button("Encode Message", key="tab2_encode_button"):
            if image and secret_message:
                encoded_buffer = encode_message(image, secret_message)
                st.download_button("Download Encoded Image", encoded_buffer, "encoded.png", key="tab2_download_button")

    encoded_image = st.file_uploader("Upload Encoded Image to Decode", type=["png"], key="tab2_encoded_image_uploader")
    if st.button("Decode Message", key="tab2_decode_button"):
        if encoded_image:
            message = decode_message(Image.open(encoded_image))
            st.write(f"Decoded Message: {message}")  
with tab3:
    st.subheader("Face & Eye Detection")
    option = st.selectbox("Choose Image Source", ["Upload Image", "Use Webcam"], key="tab3_selectbox")

    if option == "Upload Image":
        face_image = st.file_uploader("Upload an Image for Face & Eye Detection", type=["jpg", "jpeg", "png"], key="tab3_file_uploader")
        if face_image:
            image = Image.open(face_image)
    elif option == "Use Webcam":
        webcam_image = st.camera_input("Take a picture", key="tab3_camera_input")
        if webcam_image:
            image = Image.open(webcam_image)

    if 'image' in locals():
        st.image(image, caption="Selected Image", use_container_width=True)

        if st.button("Detect Faces & Eyes", key="tab3_detect_button"):
            img_cv = pil_to_cv(image)
            detected_img = detect_faces_and_eyes(img_cv)
            st.image(cv_to_pil(detected_img), caption="Detected Faces & Eyes", use_container_width=True)