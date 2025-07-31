# Import all libray we need :

import whisper
import cv2
import os
import numpy as np
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from sklearn.cluster import KMeans
from moviepy.editor import VideoFileClip
from PIL import Image
import time

# 1. Speach to text 
def speech_to_text(audio_path):
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Transcribe audio to text
    result = model.transcribe(audio_path)
    lyrics = result['text']
    
    # Create 'lyrics' folder if it doesn't exist
    if not os.path.exists('lyrics'):
        os.makedirs('lyrics')
    
    # Extract song name from the audio file path and create filename
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    lyrics_file = f'lyrics/{song_name}_lyrics.txt'
    
    # Save lyrics to file
    with open(lyrics_file, 'w', encoding='utf-8') as file:
        file.write(lyrics)
    
    return lyrics

# 2.   analyze sentiment of text
def analyze_sentiment(lyrics):
    # Load the sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    # Analyze sentiment of the lyrics
    analysis = sentiment_analyzer(lyrics)
    emotion = analysis[0]['label']
    
    # Detect topic based on keywords in the lyrics
    if "love" in lyrics.lower():
        topic = "Love"
    elif "breakup" in lyrics.lower() or "goodbye" in lyrics.lower():
        topic = "Breakup"
    else:
        topic = "Other"
    
    return emotion, topic

# 3. create paint
def generate_painting(emotion, topic):
    # Generate a descriptive prompt for the painting
    prompt = (
        f"A breathtaking and hyper-realistic artwork that captures the emotional essence of {emotion}. "
        f"The scene should reflect the theme of {topic}, depicted with intricate details and a visually stunning composition. "
        f"The artwork should feel like a magical blend of reality and imagination, with soft, harmonious colors. "
        f"If a person is included, they should be strikingly beautiful, with captivating features, a serene expression, and an attractive aura. "
        f"The overall image should be a masterpiece, evoking deep emotion and visual appeal."
    )

    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
    
    # Generate image based on the prompt
    image = pipe(prompt).images[0]
    
    # Create 'paintings' folder if it doesn't exist
    if not os.path.exists('paintings'):
        os.makedirs('paintings')
    
    # Save the generated painting
    image_file = f'paintings/{emotion}_{topic}_painting.jpeg'
    image.save(image_file)
    
    return image_file

# 4. craete paiting video
def create_painting_video(image_path):
    # Load the image and prepare it for processing
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    
    # Apply KMeans to find the dominant colors in the image
    kmeans = KMeans(n_clusters=50)
    kmeans.fit(pixels)
    
    # Get the dominant colors from the KMeans result
    dominant_colors = kmeans.cluster_centers_.astype(int)
    canvas = np.ones_like(image_rgb) * 255  # Create a blank white canvas
    
    # Define video parameters (duration, FPS, etc.)
    painting_duration = 30  # 30 seconds
    freeze_duration = 3  # 3 seconds freeze frame at the end
    fps = 30
    total_frames = int(painting_duration * fps)
    freeze_frames = int(freeze_duration * fps)
    
    # Create the 'videos' folder if it doesn't exist
    if not os.path.exists('videos'):
        os.makedirs('videos')
    
    # Prepare the video writer
    video_file = f'videos/{audio_path}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    painting_video = cv2.VideoWriter(video_file, fourcc, fps, (image_rgb.shape[1], image_rgb.shape[0]))
    
    # Shuffle the color indices to simulate a random painting process
    sorted_indices = np.arange(len(dominant_colors))
    np.random.shuffle(sorted_indices)
    
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]  # Total number of pixels
    pixels_painted = 0
    frames_written = 0
    
    # Start painting simulation
    for color_index in sorted_indices:
        mask = (kmeans.labels_ == color_index).reshape(image_rgb.shape[:2])
        
        for (x, y) in np.argwhere(mask):
            canvas[x, y] = dominant_colors[color_index]
            pixels_painted += 1
            
            # Calculate how many frames to write at this point
            frames_to_write = int((pixels_painted / total_pixels) * total_frames) - frames_written
            
            # Write the frames to simulate the painting process
            for _ in range(frames_to_write):
                frame = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
                painting_video.write(frame)
                frames_written += 1
                if frames_written >= total_frames:
                    break
            if frames_written >= total_frames:
                break
        if frames_written >= total_frames:
            break
    
    # Write the final freeze frame
    last_frame = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
    for _ in range(freeze_frames):
        painting_video.write(last_frame)
    
    painting_video.release()  # Finish the video
    print(f"Painting process video saved as '{video_file}'.")

# 5. Run
def process_audio_file(audio_file):
    # Step 1: Convert audio to text
    lyrics = speech_to_text(audio_file)
    
    # Step 2: Analyze the sentiment and topic of the lyrics
    emotion, topic = analyze_sentiment(lyrics)
    
    # Step 3: Generate a painting based on the emotion and topic
    painting_path = generate_painting(emotion, topic)
    
    # Step 4: Create a painting process video
    create_painting_video(painting_path)

# Example usage
audio_path = 'music/Softcore By TheNeighbourhood.mp3'  # Path to the audio file
process_audio_file(audio_path)
