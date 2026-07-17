"""Basic sanity tests for text preprocessing.

Run with:  python -m pytest tests/
These don't need the trained models — they test the tokenization contract.
"""

from tensorflow import keras

from src.inference import tokenize_text, MAX_LENGTH_IN


def make_tokenizer():
    tok = keras.preprocessing.text.Tokenizer(filters="", num_words=10000)
    tok.word_index = {"<start>": 1, "<end>": 2, "have": 21, "a": 13,
                      "good": 5, "weekend": 40}
    return tok


def test_output_is_padded_to_fixed_length():
    tok = make_tokenizer()
    tensor = tokenize_text("have a", tok)
    assert tensor.shape == (1, MAX_LENGTH_IN)


def test_start_and_end_tokens_are_added():
    tok = make_tokenizer()
    tensor = tokenize_text("have a", tok)
    ids = [t for t in tensor[0] if t != 0]
    assert ids[0] == tok.word_index["<start>"]
    assert ids[-1] == tok.word_index["<end>"]


def test_input_is_lowercased():
    tok = make_tokenizer()
    assert (tokenize_text("HAVE A", tok) == tokenize_text("have a", tok)).all()
