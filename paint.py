import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline

def analyze_sentiment(lyrics):
    sentiment_analyzer = pipeline("sentiment-analysis")
    analysis = sentiment_analyzer(lyrics)
    
    emotion = analysis[0]['label']
    
    if "love" in lyrics.lower():
        topic = "Love"
    elif "breakup" in lyrics.lower() or "goodbye" in lyrics.lower():
        topic = "Breakup"
    else:
        topic = "Other"
    
    return emotion, topic

def generate_painting(emotion, topic):

    prompt = (
    f"A creatively expressive artwork reflecting the deep emotions of {emotion}, "
    f"depicting the struggles and joys of {topic} through vivid imagery and powerful symbolism. "
    f"The painting should convey a strong connection to the lyrics, allowing viewers to feel the essence of the music, "
    f"with characters and scenes that encapsulate the emotions and narrative of the song."
)

    
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to("cpu")  # Change to "cuda" if you have a GPU available
    
    image = pipe(prompt).images[0]
    
    return image

def process_lyrics(lyrics):
    emotion, topic = analyze_sentiment(lyrics)
    
    painting = generate_painting(emotion, topic)
    
    return painting

interface = gr.Interface(
    fn=process_lyrics,
    inputs="text",
    outputs="image",
    title="AI Music Emotion to Painting",
    description="Enter the lyrics of a song, and the AI will analyze the emotions of its lyrics and generate a classical-style painting based on the emotions."
)

interface.launch()
