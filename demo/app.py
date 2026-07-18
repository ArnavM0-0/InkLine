"""Live demo for Hugging Face Spaces.

Setup on huggingface.co:
  1. Create a new Space -> SDK: Gradio -> free CPU hardware.
  2. Upload: app.py, requirements.txt, encoder-model.h5, inf-model.h5,
     word_dict.json  (all in the Space's root).
  3. requirements.txt should contain:
        tensorflow==2.15.0
        gradio
        numpy
     (Pin the same TF major version you trained with, or the .h5 files
      may fail to load. If you retrained on modern TF, pin that instead.)
  4. The Space builds automatically and gives you a public URL.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import time

import gradio as gr
import numpy as np
from tensorflow import keras

from beam_search import beam_search_decode  # keep beam_search.py beside app.py

# ----- Load artifacts once at startup -----
enc_model = keras.models.load_model("encoder-model.h5", compile=False)
inf_model = keras.models.load_model("inf-model.h5", compile=False)

with open("word_dict.json") as f:
    word_dict = json.load(f)
tokenizer = keras.preprocessing.text.Tokenizer(filters="", num_words=10000)
tokenizer.word_index = word_dict


def suggest(prefix):
    prefix = (prefix or "").strip()
    if len(prefix.split()) < 2:
        return "Type at least two words to get suggestions..."

    start = time.perf_counter()
    suggestions = beam_search_decode(
        prefix, enc_model, inf_model, tokenizer, beam_width=3, top_n=3
    )
    latency_ms = (time.perf_counter() - start) * 1000

    lines = [
        f"{i + 1}. {prefix} **{completion}**"
        for i, (completion, _) in enumerate(suggestions)
        if completion.strip()
    ]
    lines.append(f"\n*Generated in {latency_ms:.0f} ms (beam width 3)*")
    return "\n\n".join(lines)


demo = gr.Interface(
    fn=suggest,
    inputs=gr.Textbox(
        label="Start typing an email...",
        placeholder="e.g. let me know",
        lines=2,
    ),
    outputs=gr.Markdown(label="Suggestions"),
    live=True,  # updates as the user types, Smart Compose style
    title="InkLine — Email Autocomplete",  # <- your project name here
    description=(
        "A seq2seq model trained on the Enron email corpus suggests how "
        "your sentence continues. Built with Keras; decoding via beam "
        "search. Try: 'have a', 'thanks for', 'please call me'."
    ),
    examples=[["let me know"], ["have a"], ["thanks for"], ["please review"]],
)

if __name__ == "__main__":
    demo.launch()
