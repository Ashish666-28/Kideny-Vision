# Kidney Project

A Flask app that classifies kidney CT images with a custom PyTorch Swin model and answers kidney education questions using Google Gemini and a Word-document knowledge base. It can run locally or in a Cloudflare Container behind a Worker. The classifier returns four labels: `normal`, `stone`, `cyst`, and `tumor`.

This is a research and education prototype. Image labels are unconfirmed predictions, softmax scores are uncalibrated, and the knowledge base is an educational draft awaiting clinical review. It is not a medical diagnostic device.

## Run locally

Use Python 3.12 and run these commands from the project folder in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If you already have a configured `.env`, keep it. Edit the local `.env` to set `GOOGLE_API_KEY` to your own Gemini key. `GEMINI_API_KEY` is also accepted; `GOOGLE_API_KEY` takes priority. `GEMINI_MODEL` and `FLASK_SECRET_KEY` are optional. The application uses `gemini-flash-latest` by default and generates a Flask session secret when none is configured.

The requirements list the packages imported by the application and terminal tools; pip installs their transitive dependencies automatically. Version ranges allow pip to select compatible releases within the specified major versions. Install CPU PyTorch first using its dedicated index, as documented in the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

If an activated environment still reports missing modules, use the explicit `.venv` Python commands above so installation and execution use the same interpreter. If that interpreter reports `No Python at ...`, install Python 3.12 and recreate the virtual environment before installing dependencies. Import names differ from package names: `dotenv` comes from `python-dotenv`, `PIL` from `Pillow`, `docx` from `python-docx`, and `sklearn` from `scikit-learn`.

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>. Upload a PNG or JPG CT image for classification, or ask a general kidney question. Gemini answers require network access and a working API key; image classification runs locally on CPU.

## Deploy to Cloudflare Workers and Containers

This project requires a Cloudflare Container because it runs Flask and native PyTorch. The Worker in `src/index.ts` forwards requests to the Flask container. Flask serves the frontend HTML, CSS, and JavaScript from `templates/` and `static/` and runs the bundled `models/weights/model.pth` checkpoint. The Gemini key is read from a Worker secret and passed only to the container, never to the browser or repository.

1. Install Node.js dependencies: `npm install`.
2. Authenticate: `npx wrangler login`.
3. Add secrets interactively. Do not put their values in `wrangler.jsonc`, GitHub Actions, or Git. Set a stable Flask session secret for browser sessions. `GOOGLE_API_KEY` is needed for chat; image classification and the frontend still work without it:

   ```powershell
   npx wrangler secret put GOOGLE_API_KEY
   npx wrangler secret put FLASK_SECRET_KEY
   ```

4. Ensure Docker Desktop is running, then deploy:

   ```powershell
   npm run deploy
   ```

The first deploy builds the Docker image and may take several minutes. The Worker name defaults to `kidney-vision`; change `name` in `wrangler.jsonc` if it is already used in your Cloudflare account. Open the URL printed by Wrangler and check `/health`, `/static/css/style.css`, and `/static/js/script.js` before trying an image upload. After adding or changing `GOOGLE_API_KEY`, redeploy to restart the container with the new environment.

### Deploy on pushes to GitHub

The repository includes a manually triggered `.github/workflows/deploy.yml`. Add `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` as GitHub Actions repository secrets if you choose to deploy on Cloudflare. The token needs permission to deploy Workers and Containers in the target account. Run the first deploy above to create the Worker and set its `GOOGLE_API_KEY` and `FLASK_SECRET_KEY` secrets. Then run the workflow from GitHub Actions when needed.

If you instead connect this repo through Cloudflare Workers Builds, use `npm ci` as the build command and `npm run deploy` as the production deploy command. Cloudflare preview uploads do not update the container image.

## Deploy to Vercel

The repository includes `Dockerfile.vercel`, which Vercel builds as a container-backed Function. This keeps the trained PyTorch model, Flask backend, knowledge base, HTML, CSS, and JavaScript together. Vercel's large Functions support is required for the PyTorch image; new Vercel projects created after June 30, 2026 are enrolled automatically. Older projects may need `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in project settings. Fluid compute must be enabled.

1. Import this GitHub repository into a new Vercel project, keeping the repository root as the project root. Vercel detects `Dockerfile.vercel`; do not select a separate framework build command.
2. In Project Settings > Environment Variables, add `FLASK_SECRET_KEY` as a long random secret for Production and Preview. The container requires this stable key to sign browser sessions across instances. Add `GOOGLE_API_KEY` to enable Gemini chat. `GEMINI_MODEL` is optional and defaults to `gemini-flash-latest`. Do not commit these values. If using a custom domain, set `TRUSTED_HOSTS` to its hostname (or a comma-separated list); Vercel's `*.vercel.app` URLs work by default.
3. Deploy. Open the resulting URL and check `/health`, `/static/css/style.css`, and `/static/js/script.js`. Then upload a sample CT image and ask a kidney question. Classification uses the bundled `models/weights/model.pth`; chat uses the server-side Gemini key. Connected GitHub projects redeploy on pushes to the production branch.

Vercel limits Function request bodies to 4.5 MB, so the browser resizes images above 4 MB before upload. The page still accepts PNG/JPG files up to 10 MB. Browser conversation state is signed by the server and kept in the tab's session storage, allowing the next request to reach a different Function instance. The state expires after 30 minutes. The app remains a research and education prototype, not a diagnostic service.

## Required assets

- `models/weights/model.pth` is the trained checkpoint used by `model_predictor.py` (about 26.4 MiB). It must contain the model state, class names, and configuration matching `custom_swin.py`.
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

The app accepts images up to 10 MiB, processes them in memory, and does not save uploads. On Cloudflare and locally, recent conversation history stays in the process for 30 idle minutes and is lost on restart. On Vercel, conversation state is signed and held in browser session storage so it survives instance changes. Chat requests send the question, recent dialogue, retrieved passages, and any predicted label to Gemini.

The `.gitignore` excludes `.env`, virtual environments, caches, editor settings, local upload/patient-data directories, and extra training checkpoints. Keep API keys and personal medical files out of the repository.
