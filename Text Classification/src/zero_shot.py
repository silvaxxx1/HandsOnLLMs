import os
import openai
from config import CONFIG
from .eval import evaluate  # optional, not needed here if no eval

def build_prompt(text, labels):
    label_list = ", ".join(labels)
    few_shot_examples = [
        ("I'm feeling great today!", "joy"),
        ("Why is everything so hard today?", "sadness"),
        ("I just got promoted!", "joy"),
        ("You're the worst person ever!", "anger"),
        ("I miss you so much.", "love"),
        ("I can't believe this is happening!", "surprise")
    ]

    prompt = f"Classify the following sentence into one of: [{label_list}].\n\n"
    for example, label in few_shot_examples:
        prompt += f"Example: {example}\nLabel: {label}\n\n"

    prompt += f"Example: {text}\nLabel:"
    return prompt


def run_zero_shot(config, input_text: str):
    api_key_env = config["zero_shot"]["api_key_env"]
    model_name = config["zero_shot"]["model"]

    openai.api_key = os.getenv(api_key_env)
    if not openai.api_key:
        raise ValueError(f"❌ OpenAI API key not found. Set the environment variable: {api_key_env}")

    label_names = config["labels"]
    prompt = build_prompt(input_text, label_names)

    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )

        pred_label = response.choices[0].message.content.strip().lower()
        print(f"\n[🔍 Prediction]: {pred_label}")

        if pred_label not in label_names:
            print("⚠️ Warning: Model returned an unknown label.")

    except Exception as e:
        print(f"[ERROR] OpenAI API error: {e}")
