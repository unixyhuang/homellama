import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load summarization model and tokenizer
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)


def summarize_dialog_history(dialog_history):
    summary_prompt = "Summarize the following conversation for learning user preference in a smart home:\n"
    for message in dialog_history:
        summary_prompt += f"{message['role']}: {message['content']}\n"

    inputs = summarizer_tokenizer.encode(summary_prompt, return_tensors="pt")
    summary_ids = summarizer_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def append_user_profile(summary, user_profile_path):
    os.makedirs(os.path.dirname(user_profile_path), exist_ok=True)
    with open(user_profile_path, "a", encoding="utf-8") as f:
        f.write(summary + "\n\n")

def load_user_profile(user_profile_path):
    if os.path.exists(user_profile_path):
        with open(user_profile_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""