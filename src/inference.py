"""Model loading + greedy decoding, refactored out of the notebook.

Usage:
    from src.inference import load_artifacts, greedy_decode

    enc_model, inf_model, tokenizer = load_artifacts("models/")
    print(greedy_decode("let me know", enc_model, inf_model, tokenizer))

For top-k suggestions, prefer `src.beam_search.beam_search_decode`.
"""

import json
from pathlib import Path

import numpy as np
from tensorflow import keras

VOCAB_MAX_SIZE = 10000
MAX_LENGTH_IN = 21
MAX_LENGTH_OUT = 20


def load_artifacts(model_dir="models"):
    """Load the encoder, inference decoder, and tokenizer from `model_dir`.

    Expects: encoder-model.h5, inf-model.h5, word_dict.json
    """
    model_dir = Path(model_dir)
    enc_model = keras.models.load_model(model_dir / "encoder-model.h5",
                                        compile=False)
    inf_model = keras.models.load_model(model_dir / "inf-model.h5",
                                        compile=False)

    with open(model_dir / "word_dict.json") as f:
        word_dict = json.load(f)
    tokenizer = keras.preprocessing.text.Tokenizer(
        filters="", num_words=VOCAB_MAX_SIZE
    )
    tokenizer.word_index = word_dict

    return enc_model, inf_model, tokenizer


def tokenize_text(text, tokenizer):
    """Wrap in <start>/<end>, convert to ids, pad to MAX_LENGTH_IN."""
    text = "<start> " + text.lower() + " <end>"
    tensor = tokenizer.texts_to_sequences([text])
    return keras.preprocessing.sequence.pad_sequences(
        tensor, maxlen=MAX_LENGTH_IN, padding="post"
    )


def greedy_decode(input_sentence, enc_model, inf_model, tokenizer):
    """Generate a completion by taking the argmax token at each step."""
    index_to_word = dict(map(reversed, tokenizer.word_index.items()))

    state = enc_model.predict(tokenize_text(input_sentence, tokenizer),
                              verbose=0)

    target_seq = np.array([[tokenizer.word_index["<start>"]]])
    decoded_words = []

    for _ in range(MAX_LENGTH_OUT - 1):
        output_tokens, state = inf_model.predict([target_seq, state],
                                                 verbose=0)
        token_id = int(np.argmax(output_tokens[0, 0]))

        if token_id == 0:  # padding — nothing more to predict
            break
        word = index_to_word[token_id]
        if word == "<end>":
            break

        decoded_words.append(word)
        target_seq = np.array([[token_id]])

    return " ".join(decoded_words)
