# Kidney Project

A local Flask prototype that classifies kidney CT images with a custom PyTorch Swin model and answers kidney education questions using Google Gemini and a Word-document knowledge base. The classifier returns four labels: `normal`, `stone`, `cyst`, and `tumor`.

This is a research and education prototype. Image labels are unconfirmed predictions, softmax scores are uncalibrated, and the knowledge base is an educational draft awaiting clinical review. It is not a medical diagnostic device.

## Run locally

Use Python 3.12 and run these commands from the project folder in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If you already have a configured `.env`, keep it. Edit the local `.env` to set `GOOGLE_API_KEY` to your own Gemini key. `GEMINI_API_KEY` is also accepted; `GOOGLE_API_KEY` takes priority. `GEMINI_MODEL` and `FLASK_SECRET_KEY` are optional. The application uses `gemini-flash-latest` by default and generates a Flask session secret when none is configured.

The requirements preserve the existing environment's pinned versions. The additional package index supplies CPU PyTorch wheels, as documented in the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/). A fresh installation has not been verified; wheel availability depends on your Python version, operating system, and the pinned releases.

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>. Upload a PNG or JPG CT image for classification, or ask a general kidney question. Gemini answers require network access and a working API key; image classification runs locally on CPU.

## Required assets

- `models/weights/finetuned_v2_best.pth` is the trained checkpoint used by `model_predictor.py` (about 26.4 MiB). It must contain the model state, class names, and configuration matching `custom_swin.py`.
- `data/raw/kidney_knowledge.docx` is the structured educational knowledge base used by `knowledge_base.py`. Retrieval builds a TF-IDF index in memory; no separate index-generation step is required.

These runtime assets are included. The training notebook referenced by `model_predictor.py` and the original training image dataset are not included.

## Checks and terminal tools

```powershell
# Backend checks use injected services and make no Gemini calls.
.\.venv\Scripts\python.exe check_flask_backend.py

# Load the trained checkpoint, optionally supplying a local CT image.
.\.venv\Scripts\python.exe model_predictor.py
.\.venv\Scripts\python.exe model_predictor.py path\to\test_ct.png

# Inspect knowledge-base retrieval without calling Gemini.
.\.venv\Scripts\python.exe knowledge_base.py --condition stone --question "What foods should I eat?"

# Interactive Gemini chat with a manually selected test context.
.\.venv\Scripts\python.exe chatbot.py --condition stone
```

`scan_chat.py` also provides a terminal scan-explanation flow and reusable conversation class. A terminal `--condition` argument supplies a test label; it does not classify an image.

## Project layout and privacy

`app.py` serves the API and browser interface in `templates/` and `static/`. `model_predictor.py` handles checkpoint loading and image preprocessing; `custom_swin.py` defines the classifier. `chatbot.py` combines retrieved passages with recent conversation context for Gemini.

The app accepts images up to 10 MiB, processes them in memory, and does not save uploads. Recent conversation history stays in the local process for 30 idle minutes and is lost on restart. Chat requests send the question, recent dialogue, retrieved passages, and any predicted label to Gemini. This server is configured for local, single-process use on `127.0.0.1`.

The `.gitignore` excludes `.env`, virtual environments, caches, editor settings, local upload/patient-data directories, and extra training checkpoints. Keep API keys and personal medical files out of the repository.
