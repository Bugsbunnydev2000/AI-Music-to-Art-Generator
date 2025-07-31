# AI-Music-to-Art-Generator
This project transforms music or lyrics into visual art using AI. It transcribes audio, analyzes sentiment, generates paintings with Stable Diffusion, and creates videos simulating the painting process.

-----------

**Features!**

Transcribe audio to lyrics with Whisper.

Analyze lyrics for emotion and topic with Transformers.

Generate paintings with Stable Diffusion.

Create videos showing the painting process.

Web interface via Gradio for lyrics-based painting.

**Prerequisites**

Python: 3.8+

FFmpeg: Install via sudo apt-get install ffmpeg (Ubuntu), brew install ffmpeg (macOS), or download for Windows.

Hugging Face Account: Set token for Stable Diffusion:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

Hardware: 8GB+ RAM, GPU recommended.

---------

**Usage Examples :**

1. Process an Audio File (main.py)

Generate lyrics, painting, and video from an audio file.

```bash
# Place song.mp3 in music/
# Edit main.py to set audio_path = 'music/song.mp3'
python main.py
```

Output: Lyrics in lyrics/, painting in paintings/, video in videos/.

2. Generate Painting from Lyrics (paint.py)

Launch Gradio interface to input lyrics and get a painting.

```bash
python paint.py
```

3. Create Painting Video (repaint.py)

Generate a video for an existing image.

```bash
# Place image.jpeg in project root
# Edit repaint.py to set image_path = 'image.jpeg'
python repaint.py
```

Output: Video saved as final_painting_process.mp4.

-------------

**Notes**


Ensure input files exist (.mp3 for audio, .jpeg/.png for images).

Stable Diffusion is resource-intensive; GPU speeds up processing.

Customize video duration or clusters in main.py/repaint.py.
