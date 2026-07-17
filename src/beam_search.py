"""Beam search decoding for the seq2seq Smart Compose model.

Drop-in upgrade over the greedy `decode_sequence` in the original notebook.
Works with the same `enc_model` / `inf_model` pair:
  - enc_model(input_tensor)            -> encoded state (1, units)
  - inf_model([target_seq, state])     -> (logits (1, 1, vocab), new_state)

Usage:
    from src.beam_search import beam_search_decode
    suggestions = beam_search_decode("let me know", enc_model, inf_model,
                                     tokenizer, beam_width=3, top_n=3)
"""

import numpy as np
from tensorflow import keras

MAX_LENGTH_IN = 21
MAX_LENGTH_OUT = 20


def tokenize_text(text, tokenizer):
    text = "<start> " + text.lower() + " <end>"
    tensor = tokenizer.texts_to_sequences([text])
    return keras.preprocessing.sequence.pad_sequences(
        tensor, maxlen=MAX_LENGTH_IN, padding="post"
    )


def beam_search_decode(input_sentence, enc_model, inf_model, tokenizer,
                       beam_width=3, top_n=3, length_penalty=0.7):
    """Return the `top_n` most likely completions for `input_sentence`.

    Each hypothesis is a tuple: (token_ids, log_prob, decoder_state, finished)
    At every step we expand each live hypothesis with its `beam_width` best
    next tokens, then keep only the `beam_width` best hypotheses overall.
    `length_penalty` < 1 normalizes scores so longer completions aren't
    unfairly punished for accumulating more negative log-probs.
    """
    index_to_word = dict(map(reversed, tokenizer.word_index.items()))
    start_id = tokenizer.word_index["<start>"]
    end_id = tokenizer.word_index["<end>"]

    init_state = enc_model.predict(
        tokenize_text(input_sentence, tokenizer), verbose=0
    )

    # One initial hypothesis: just the <start> token.
    beams = [([start_id], 0.0, init_state, False)]

    for _ in range(MAX_LENGTH_OUT - 1):
        candidates = []
        for tokens, log_prob, state, finished in beams:
            if finished:
                candidates.append((tokens, log_prob, state, True))
                continue

            target_seq = np.array([[tokens[-1]]])
            logits, new_state = inf_model.predict(
                [target_seq, state], verbose=0
            )
            # Convert logits to log-probabilities over the vocabulary.
            log_probs = np.log(
                keras.activations.softmax(
                    keras.backend.constant(logits[0, 0])
                ).numpy()
                + 1e-10
            )

            # Expand with the beam_width best next tokens (skip padding id 0).
            best_ids = np.argsort(log_probs)[::-1]
            added = 0
            for token_id in best_ids:
                if token_id == 0:
                    continue
                candidates.append((
                    tokens + [int(token_id)],
                    log_prob + float(log_probs[token_id]),
                    new_state,
                    int(token_id) == end_id,
                ))
                added += 1
                if added == beam_width:
                    break

        # Keep the best `beam_width` hypotheses (length-normalized score).
        candidates.sort(
            key=lambda c: c[1] / (len(c[0]) ** length_penalty), reverse=True
        )
        beams = candidates[:beam_width]

        if all(finished for _, _, _, finished in beams):
            break

    # Convert token ids back to words, dropping <start>/<end>.
    results = []
    for tokens, log_prob, _, _ in beams[:top_n]:
        words = [
            index_to_word[t]
            for t in tokens
            if t not in (start_id, end_id) and t in index_to_word
        ]
        results.append((" ".join(words), log_prob))
    return results


def exact_match_at_k(test_pairs, enc_model, inf_model, tokenizer, k=3):
    """Evaluation helper: fraction of held-out (prefix, completion) pairs
    where the true completion appears in the top-k beam suggestions."""
    hits = 0
    for prefix, true_completion in test_pairs:
        suggestions = beam_search_decode(
            prefix, enc_model, inf_model, tokenizer,
            beam_width=k, top_n=k
        )
        normalized = true_completion.replace("<start>", "").replace(
            "<end>", ""
        ).strip()
        if any(s.strip() == normalized for s, _ in suggestions):
            hits += 1
    return hits / len(test_pairs)
