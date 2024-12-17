import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.preprocess import remove_background, detect_keypoints

st.title("Virtual Try-On Preprocessing")

# Upload image file
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])

if uploaded_file:
    # Convert file to NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Background removal
    removed_bg = remove_background(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    # Keypoint detection
    processed_image = detect_keypoints(image)

    # Display results
    st.image([removed_bg, cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)], 
             caption=["Background Removed", "Keypoints Detected"], use_column_width=True)
