"""Main Streamlit application for image processing and steganography."""
import streamlit as st
from PIL import Image

# Import modules
from utils import pil_to_cv, cv_to_pil
from image_processing import (
    convert_to_grayscale, apply_canny, apply_gaussian_blur, 
    apply_thresholding, detect_contours, apply_sharpening, 
    pencil_sketch, apply_sepia, rotate_image, cartoonize_image, 
    glitch_effect, oil_paint_effect, pixelate
)
from steganography import encode_message, decode_message
from face_detection import detect_faces_and_eyes


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