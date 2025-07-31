# Importing necessary libraries to process the image for color segmentation
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageDraw
from moviepy.editor import VideoFileClip

# Load the image
image_path = 'paint.jpeg'
image = cv2.imread(image_path)

# Convert the image to RGB (OpenCV loads images in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image_rgb.reshape(-1, 3)

# Using KMeans clustering to segment colors (with 100 clusters)
kmeans = KMeans(n_clusters=50)
kmeans.fit(pixels)

# Get the cluster centers (dominant colors)
dominant_colors = kmeans.cluster_centers_.astype(int)

# Convert dominant colors to HSV
dominant_colors_hsv = cv2.cvtColor(np.uint8([dominant_colors]), cv2.COLOR_RGB2HSV)[0]

def categorize_color(hsv):
    hue, saturation, value = hsv
    is_cool = bool(60 <= hue <= 240)  # Convert to Python boolean
    is_dark = bool(value < 128)  # Convert to Python boolean
    # Use ~ for boolean negation, and negate hue as an integer
    return (~is_cool, ~is_dark, -int(hue), value)

# Sort colors according to the reversed desired flow
sorted_colors = sorted(enumerate(dominant_colors_hsv), key=lambda x: categorize_color(x[1]))
sorted_indices = [i for i, _ in sorted_colors]

# Create a blank canvas
canvas = np.ones_like(image_rgb) * 255

# Function to get connected components
def get_connected_components(mask):
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    return labels, num_labels

# Set desired video duration (in seconds) and frame rate
painting_duration = 30  # Change this to your desired duration for the painting process
freeze_duration = 3  # Duration to freeze the last frame
fps = 30

# Set up video writer for the painting process
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
painting_video = cv2.VideoWriter('final_painting_process.mp4', fourcc, fps, (image_rgb.shape[1], image_rgb.shape[0]))

# Calculate total frames to write
total_frames = int(painting_duration * fps)
freeze_frames = int(freeze_duration * fps)

# Fill the canvas with sorted colors and connected components
total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
pixels_painted = 0
frames_written = 0

for i, color_index in enumerate(sorted_indices):
    mask = (kmeans.labels_ == color_index).reshape(image_rgb.shape[:2])
    labels, num_labels = get_connected_components(mask)
    
    # Get all connected components for this color
    components = [np.argwhere(labels == label) for label in range(1, num_labels)]
    
    # Shuffle the order of components
    np.random.shuffle(components)
    
    for component_pixels in components:
        np.random.shuffle(component_pixels)  # Shuffle the pixels within the component
        
        for pixel in component_pixels:
            canvas[pixel[0], pixel[1]] = dominant_colors[color_index]
            pixels_painted += 1
            
            # Calculate how many frames should have been written by now
            frames_to_write = int((pixels_painted / total_pixels) * total_frames) - frames_written
            
            # Write frames if needed
            for _ in range(frames_to_write):
                frame = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
                painting_video.write(frame)
                frames_written += 1
                
                # Check if we've reached the total number of frames
                if frames_written >= total_frames:
                    break
            
            if frames_written >= total_frames:
                break
        
        if frames_written >= total_frames:
            break
    
    if frames_written >= total_frames:
        break

# Ensure we've written exactly the right number of frames
while frames_written < total_frames:
    frame = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
    painting_video.write(frame)
    frames_written += 1

# Add freeze frames
last_frame = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
for _ in range(freeze_frames):
    painting_video.write(last_frame)
    frames_written += 1

# Release the painting process video writer
painting_video.release()

print(f"Final video saved as 'final_painting_process.mp4' (Duration: {painting_duration + freeze_duration} seconds, FPS: {fps})")
