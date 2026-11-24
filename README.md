# GenEdit — AI Image Editor

Edit, organise and search images with plain-language instructions. GenEdit uses Gemini to edit and caption images, and OpenAI embeddings to make your library searchable by description.

## Features

- **Natural-language editing** — "remove the background", "add a tree on the right", "make it more vibrant", or one-click presets
- **Automatic captions** — every upload and edit is captioned by Gemini
- **Semantic search** — find images by meaning ("beach photos", "person with dog") via cosine similarity over caption embeddings
- **Version history** — every edit is a new version that stores its prompt; restore any earlier version in one click
- **Search evaluation** — `eval_search.py` measures Recall@k and MRR on your own labelled queries
- **Offline tests** — model calls are mocked, so the test suite runs without API keys

## How it works

```
upload ──► normalise to PNG ──► Gemini caption ──► OpenAI embedding ──► metadata.json
edit   ──► Gemini 2.5 Flash Image ──► new PNG version ──► re-caption + re-embed
search ──► embed query ──► cosine similarity vs. caption embeddings ──► ranked results
```

| Component | Choice |
|---|---|
| UI | Streamlit |
| Image editing | `gemini-2.5-flash-image` (Google GenAI SDK) |
| Captioning | `gemini-2.5-flash` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Storage | PNG files per version + JSON metadata (atomic writes) |

Model names can be overridden with `GENEDIT_CAPTION_MODEL`, `GENEDIT_EDIT_MODEL` and `GENEDIT_EMBED_MODEL` in `.env`.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/poornachandra-sarasAI/GenEdit-AI-Image-Editor.git
cd GenEdit-AI-Image-Editor
pip install -r requirements.txt
cp .env.example .env        # then add your keys
streamlit run streamlit_app.py
```

Keys: a Gemini key from [Google AI Studio](https://aistudio.google.com/) (`GEMINI_API_KEY`) and an [OpenAI](https://platform.openai.com/) key (`OPENAI_API_KEY`).

## Usage

1. **Add Image** — upload a PNG/JPG with an optional title. It is captioned and indexed automatically.
2. **Library** — browse everything, or search in natural language. Click **Open** on any image.
3. **Image Detail** — apply a preset or write your own instruction, then **Apply edit**. Scroll down for the version history and **Restore**.

If the model refuses or returns no image, the app shows the model's reply and does not save a broken version.

## Evaluating search

Label a few queries with the image titles (or IDs) that should come back:

```json
[
  {"query": "dog playing on a beach", "relevant": ["beach-dog"]},
  {"query": "a red sports car", "relevant": ["red-car"]}
]
```

```bash
python eval_search.py eval/queries.json --k 1 3 5
```

It prints Recall@1/3/5 and MRR, and lists the queries where the top result was wrong. See `eval/queries.example.json` for the format.

## Tests

```bash
pip install pytest
python -m pytest -q
```

Covers storage and versioning, search ranking, PNG normalisation, the no-image error path, the evaluation metrics, and app navigation (Streamlit `AppTest`).

## Project structure

```
streamlit_app.py   # Streamlit UI and page routing
ai_utils.py        # Gemini (edit, caption) and OpenAI (embeddings) calls
storage.py         # Versioned file storage, metadata, vector search
eval_search.py     # Recall@k / MRR evaluation for search
eval/              # Example labelled queries
tests/             # Offline test suite
data/              # Your local image library (git-ignored)
```

## Limitations and next steps

- Single-user, local storage; search is a linear scan, fine for hundreds of images but not millions
- Search runs on captions, so it only finds what the caption mentions. Next: embed images directly (e.g. CLIP) and compare with this evaluation
- Gemini returns opaque images, so "remove background" replaces it with white rather than transparency
- Planned: batch edits, masked/regional edits, cloud storage
