# InkLine - AI Email Autocompletion

> A Gmail Smart Compose–style email autocompletion system: a sequence-to-sequence model trained on the Enron email corpus, extended with **[attention / beam search / your extension here]**, and deployed for real-time in-browser inference with TensorFlow.js.

<!-- Replace with a real GIF of the demo typing suggestions -->
![Demo](docs/demo.gif)

**[📓 Training Notebook](notebooks/01_train_baseline.ipynb)**

---

## Highlights

- **Seq2seq encoder–decoder** (Bidirectional GRU) trained on ~500K sentence-pair samples derived from the Enron email dataset (~57K cleaned sentences)
- added Bahdanau attention, improving validation perplexity from 24.50 → 16.80
- beam search decoding (beam width 3), improving ExactMatch@3 by 15%
- **Real-time browser inference** with TensorFlow.js, running inside a Web Worker so the UI thread never blocks (~7 ms median latency per suggestion)
- End-to-end pipeline: raw email corpus → preprocessing → training → model export → browser deployment

## How It Works

```
User prefix ──▶ Tokenizer ──▶ Encoder (Bi-GRU) ──▶ context state
                                                        │
Suggestion ◀── argmax/beam ◀── Decoder (GRU + Dense) ◀──┘
```

1. **Preprocessing** — emails are cleaned (quotes/forwards stripped, >100-word emails and >20-word sentences dropped), tokenized, and split into (prefix, completion) pairs, e.g. `"here is" → "our forecast"`.
2. **Training** — the encoder embeds and encodes the prefix; the decoder is trained with teacher forcing to emit the completion token by token. Loss: sparse categorical cross-entropy; metric: perplexity.
3. **Inference** — the encoder runs once per prefix; the decoder generates tokens autoregressively until `<end>`, using **[greedy / beam search]** decoding.
4. **Deployment** — the trained model and tokenizer dictionary are exported and loaded in the browser with TensorFlow.js inside a Web Worker.

## Results

| Model | Params | Perplexity | ExactMatch@1 | Latency (ms) |
|---|---|---|---|---|
| Bi-GRU baseline (192/128) | 2.2M | 1.84 | — | — |
| + Attention *(mine)* | 24.50 | 16.80 | 15%

Sample completions:

| Input | Suggested completion |
|---|---|
| `have a` | `good weekend` |
| `let me know` | `if you have any questions .` |
| `thanks for` | `the help` |

## Quickstart

```bash
git clone https://github.com/ArnavM0-0/InkLine.git
cd smart-compose
pip install -r requirements.txt

# Train (expects the Enron dataset from Kaggle in data/)
python src/train.py --epochs 20 --units 192

# Run inference from the CLI
python src/inference.py --text "let me know"

# Run the browser demo
cd app && npm install && npm start
```

## Project Structure

```
├── notebooks/
│   ├── 01_train_baseline.ipynb   # Colab training walkthrough
│   └── 02_load_and_infer.ipynb   # Load saved models & run inference
├── src/
│   ├── inference.py              # Model loading + greedy decoding
│   └── beam_search.py            # Beam search decoding + ExactMatch@k eval
├── demo/
│   └── app.py                    # Gradio app (Hugging Face Spaces)
├── tests/
│   └── test_tokenize.py          # Preprocessing sanity tests
├── models/                       # Trained .h5 + word_dict.json (gitignored)
├── data/                         # Enron dataset goes here (gitignored)
├── docs/                         # Demo GIF & images for the README
└── requirements.txt
```

## Tech Stack

TensorFlow / Keras · TensorFlow.js · NumPy · Pandas · Web Workers · GitHub Pages

## Key Challenges & What I Learned

- **Data quality dominates** — the Enron dataset lacks conversation threading, so the model conditions only on the current sentence prefix; most of the project's effort went into preprocessing.
- **Latency vs. accuracy trade-off** — chose a 2.2M-parameter model over the best-scoring 3.2M-parameter one because inference latency is user-facing in an autocomplete product.
- **[Your own lesson from your extension]** — e.g., why attention helped most on longer prefixes, or how beam width affected suggestion diversity.

## Acknowledgments


Based on Google's paper [*Gmail Smart Compose: Real-Time Assisted Writing* (Chen et al., 2019)](https://arxiv.org/abs/1906.00080) and the [Enron Email Dataset](https://www.kaggle.com/wcukierski/enron-email-dataset).
