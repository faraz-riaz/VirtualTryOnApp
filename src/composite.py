# src/composite.py
import cv2

def overlay_images(background, clothing, mask):
    # Resize clothing to fit the detected body mask
    resized_clothing = cv2.resize(clothing, (mask.shape[1], mask.shape[0]))
    
    # Composite clothing onto body
    result = cv2.addWeighted(background, 0.5, resized_clothing, 0.5, 0)
    return result
