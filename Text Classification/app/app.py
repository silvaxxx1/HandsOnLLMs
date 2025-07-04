import gradio as gr
from transformers import pipeline

# Load your fine-tuned model from Hugging Face Hub
model_id = "transformersbook/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)

def classify_emotion(text):
    results = classifier(text)
    label = results[0]['label']
    score = results[0]['score']
    # Color map for emotion pills
    color_map = {
        "joy": "green",
        "sadness": "blue",
        "love": "pink",
        "anger": "red",
        "fear": "purple",
        "surprise": "orange"
    }
    color = color_map.get(label.lower(), "gray")

    # Return styled HTML pill with label and confidence
    return f"<div style='display:inline-block; padding:8px 15px; border-radius:25px; background-color:{color}; color:white; font-weight:bold; font-size:16px;'>{label} ({score:.2f})</div>"

iface = gr.Interface(
    fn=classify_emotion,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here..."),
    outputs=gr.HTML(),
    title="🧠 Emotion Classifier — Powered by AI",
    description=(
        "Welcome! This app lets you detect emotions in any text you enter — "
        "from joy to sadness, anger, love, fear, and surprise.\n\n"
        "Just type a sentence or phrase expressing how you feel or what’s on your mind, "
        "and the AI model will instantly tell you the most likely emotion behind your words, "
        "along with a confidence score.\n\n"
        "Each emotion is shown as a colorful badge to make it easy to understand at a glance.\n\n"
        "*Built with ❤️ by Silva using state-of-the-art AI and Hugging Face Transformers.*"
    )
)

if __name__ == "__main__":
    iface.launch(share=True)  # Public link enabled
