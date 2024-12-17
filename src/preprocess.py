# src/preprocess.py
import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
import requests

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_image(image_path):
    print(image_path)
    # Load and convert the image
    image = cv2.imread(image_path)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
    
    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)



# Load U2Net Model
def load_u2net_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    return model

# Remove Background
def remove_background(image):
    model = load_u2net_model()

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out']
        mask = output.squeeze().argmax(0).numpy()
    
    # Apply mask
    background_removed = Image.fromarray((mask * 255).astype('uint8'))
    return background_removed

def detect_keypoints(image):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if results.pose_landmarks:
        # Draw pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
    
    return image
