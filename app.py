"""Local Flask API for the trained KidneySwin model and Gemini RAG.

Copy beside model_predictor.py, custom_swin.py, chatbot.py and knowledge_base.py.
Keep your existing templates and static files.

GET  /api/chat/session -> csrf_token, scan_id, condition, prediction, messages
POST /api/predict      multipart field image -> label, probabilities, scan_id
POST /api/chat/start   JSON {scan_id} -> initial explanation of stored prediction
POST /api/chat/message JSON {message, scan_id} -> general or follow-up answer
POST /api/chat/reset   JSON {} -> clears scan/history and returns new scan_id

Every POST requires X-CSRF-Token from the session endpoint. After predicting,
display the result first, then call chat/start. Only the server's model sets
the condition. General kidney questions work before scanning. scan_id rejects
questions for an old result. The application does not save uploaded images.

Recent dialogue remains in this process for 30 idle minutes; restart loses it.
Browser tabs share a session. This is a local, single-process prototype.
"""

import math
import os
import secrets
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from io import BytesIO
from pathlib import Path
from threading import RLock
from time import monotonic

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, session
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from chatbot import CONDITIONS, ChatbotError, KidneyChatbot, initial_explanation_question


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


class ModelUnavailable(Exception):
    """A model error safe to show in the local interface."""


def checked_prediction(raw):
    """Reject malformed output rather than presenting a guessed result."""
    if not isinstance(raw, Mapping):
        raise ModelUnavailable("The model returned an invalid result.")
    label, values = raw.get("label"), raw.get("probabilities")
    if not isinstance(label, str) or label.strip().lower() not in CONDITIONS:
        raise ModelUnavailable("The model returned an unsupported class label.")
    label = label.strip().lower()
    if not isinstance(values, Mapping) or set(values) != set(CONDITIONS):
        raise ModelUnavailable("The model must return scores for all four classes.")
    scores = {}
    for name in CONDITIONS:
        value = values[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ModelUnavailable("The model returned an invalid class score.")
        scores[name] = float(value)
    if abs(sum(scores.values()) - 1) > 0.0001:
        raise ModelUnavailable("The model's class scores do not sum to one.")
    if scores[label] < max(scores.values()) - 0.000001:
        raise ModelUnavailable("The model's label disagrees with its class scores.")
    return {
        "label": label, "confidence": scores[label], "probabilities": scores,
        "score_type": "uncalibrated_softmax",
    }


def create_app(test_config=None, chatbot_factory=None, classifier=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32),
        SESSION_COOKIE_NAME="kidney_chat_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        MAX_CONTENT_LENGTH=12 * 1024 * 1024,
        MAX_FORM_PARTS=10,
        MAX_FORM_MEMORY_SIZE=512 * 1024,
        MAX_IMAGE_BYTES=10 * 1024 * 1024,
        MAX_IMAGE_PIXELS=16_000_000,
        CHAT_IDLE_SECONDS=30 * 60,
        CHAT_MAX_SESSIONS=100,
        TRUSTED_HOSTS=["127.0.0.1", "localhost", "container"],
    )
    if test_config:
        app.config.update(test_config)

    records, records_lock, bot_lock = {}, RLock(), RLock()
    shared_bot = None
    factory = chatbot_factory or KidneyChatbot

    def get_bot():
        nonlocal shared_bot
        with bot_lock:
            if shared_bot is None:
                shared_bot = factory()
        return shared_bot

    def classify(stream):
        try:
            if classifier is not None:
                result = classifier(stream)
            else:
                from model_predictor import predict_image
                result = predict_image(stream)
            return checked_prediction(result)
        except ModelUnavailable:
            raise
        except Exception as error:
            app.logger.error("Image model failed (%s)", type(error).__name__)
            raise ModelUnavailable(
                "Could not run your trained model. Run model_predictor.py with a CT image "
                "and check custom_swin.py and models/weights/model.pth."
            ) from None

    def new_context():
        return {
            "scan_id": secrets.token_urlsafe(24), "condition": None,
            "prediction": None, "history": [], "initial_answer": None,
        }

    def prune_idle():
        now = monotonic()
        expired = [
            key for key, state in records.items()
            if not state["busy"]
            and now - state["last_seen"] > app.config["CHAT_IDLE_SECONDS"]
        ]
        for key in expired:
            del records[key]

    def bootstrap_session():
        with records_lock:
            prune_idle()
            browser_id = session.get("browser_id")
            if browser_id not in records:
                if len(records) >= app.config["CHAT_MAX_SESSIONS"]:
                    abort(503, "Chat capacity reached. Try again later.")
                browser_id = secrets.token_urlsafe(32)
                session["browser_id"] = browser_id
                session["csrf_token"] = secrets.token_urlsafe(32)
                records[browser_id] = {
                    **new_context(), "busy": False, "last_seen": monotonic()
                }
            state = records[browser_id]
            state["last_seen"] = monotonic()
            return {
                "csrf_token": session["csrf_token"], "scan_id": state["scan_id"],
                "condition": state["condition"], "prediction": state["prediction"],
                "messages": [
                    {"role": message["role"], "content": message["content"]}
                    for message in state["history"] if not message.get("internal")
                ],
                "busy": state["busy"],
                "context_source": "classifier" if state["prediction"] else "general",
            }

    def require_csrf():
        supplied = request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf_token", "")
        if not supplied.isascii() or not expected or not secrets.compare_digest(supplied, expected):
            abort(403, "Session token missing or expired. Refresh the page.")

    def json_body():
        require_csrf()
        if not request.is_json:
            abort(415, "Send JSON with Content-Type: application/json.")
        if request.content_length and request.content_length > 32 * 1024:
            abort(413, "The chat request is too large.")
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, "Send a valid JSON object.")
        return data

    @contextmanager
    def use_conversation():
        with records_lock:
            prune_idle()
            state = records.get(session.get("browser_id"))
            if state is None:
                abort(409, "Your chat expired. Refresh the page and start again.")
            if state["busy"]:
                abort(409, "Wait for the current request to finish.")
            state["busy"] = True
            state["last_seen"] = monotonic()
        try:
            yield state
        finally:
            with records_lock:
                state["busy"] = False
                state["last_seen"] = monotonic()

    def api_errors(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            try:
                return view(*args, **kwargs)
            except HTTPException:
                raise
            except (ChatbotError, ModelUnavailable) as error:
                return jsonify(ok=False, error=str(error)), 503
            except (OSError, ValueError):
                return jsonify(
                    ok=False,
                    error="Chat setup failed. Run knowledge_base.py and check your local .env configuration.",
                ), 503
            except Exception as error:
                # Do not log keys, images, provider traces or patient dialogue.
                app.logger.error("API request failed (%s)", type(error).__name__)
                return jsonify(ok=False, error="Request failed. Please try again."), 500
        return wrapped

    def require_current_scan(data, state):
        scan_id = data.get("scan_id")
        if not isinstance(scan_id, str) or not scan_id or len(scan_id) > 64:
            abort(400, "A valid scan_id is required.")
        if scan_id != state["scan_id"]:
            abort(409, "This question belongs to an older scan. Use the current result.")

    def remember(state, question, answer, automatic=False):
        with records_lock:
            state["history"] = (state["history"] + [
                {"role": "user", "content": question, "internal": automatic},
                {"role": "assistant", "content": answer},
            ])[-8:]

    def reply(state, answer):
        return jsonify(
            ok=True, answer=answer, scan_id=state["scan_id"],
            condition=state["condition"],
            context_source="classifier" if state["prediction"] else "general",
        )

    def uploaded_image():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            abort(400, "Choose a CT image first.")
        if Path(uploaded.filename).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            abort(400, "Choose a PNG or JPG image.")
        raw = uploaded.stream.read(app.config["MAX_IMAGE_BYTES"] + 1)
        if not raw:
            abort(400, "The image file is empty.")
        if len(raw) > app.config["MAX_IMAGE_BYTES"]:
            abort(413, "Choose an image smaller than 10 MB.")
        stream = BytesIO(raw)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(stream) as image:
                    if image.format not in {"PNG", "JPEG"}:
                        abort(400, "The file must contain a PNG or JPG image.")
                    if image.width * image.height > app.config["MAX_IMAGE_PIXELS"]:
                        abort(400, "The image dimensions are too large.")
                    image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            abort(400, "The file could not be read as a PNG or JPG image.")
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            abort(400, "The image dimensions are too large.")
        stream.seek(0)
        return stream

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/health")
    def health():
        """Readiness endpoint used by the Cloudflare Container router."""
        return jsonify(ok=True)

    @app.get("/api/chat/session")
    def chat_session():
        return jsonify(ok=True, **bootstrap_session())

    @app.post("/api/predict")
    @api_errors
    def predict():
        require_csrf()
        if request.mimetype != "multipart/form-data":
            abort(415, "Upload the image using multipart/form-data.")
        image = uploaded_image()
        with use_conversation() as state:
            # Invalidate old context before new inference, including on failure.
            with records_lock:
                state.update(new_context())
            prediction = classify(image)
            with records_lock:
                state.update(condition=prediction["label"], prediction=prediction)
            # Gemini runs separately, after this result is visible.
            return jsonify(
                ok=True, **prediction, scan_id=state["scan_id"],
                condition=state["condition"], context_source="classifier",
            )

    @app.post("/api/chat/start")
    @api_errors
    def start_chat():
        data = json_body()
        with use_conversation() as state:
            require_current_scan(data, state)
            if state["prediction"] is None:
                abort(409, "Analyze an image for a result explanation, or ask a general question.")
            if state["initial_answer"] is not None:
                return reply(state, state["initial_answer"])
            if state["history"]:
                abort(409, "The conversation has already started. Continue asking questions.")
            bot = get_bot()
            condition = state["condition"]
            overview = next(
                (topic for topic in bot.kb.topics if topic["id"] == f"{condition.upper()}_001"),
                None,
            )
            if overview is None:
                raise ValueError("The knowledge base is missing this result overview.")
            question = initial_explanation_question(overview["title"], condition)
            answer = bot.answer(question, condition=condition, history=[])["answer"]
            remember(state, question, answer, automatic=True)
            with records_lock:
                state["initial_answer"] = answer
            return reply(state, answer)

    @app.post("/api/chat/message")
    @api_errors
    def send_message():
        data = json_body()
        message = data.get("message")
        if not isinstance(message, str) or not message.strip():
            abort(400, "Type a question first.")
        message = message.strip()
        if len(message) > 3000:
            abort(400, "Please keep your question under 3,000 characters.")
        with use_conversation() as state:
            require_current_scan(data, state)
            answer = get_bot().answer(
                message, condition=state["condition"],
                history=[dict(item) for item in state["history"]],
            )["answer"]
            remember(state, message, answer)
            return reply(state, answer)

    @app.post("/api/chat/reset")
    @api_errors
    def reset_chat():
        json_body()
        with use_conversation() as state:
            with records_lock:
                state.update(new_context())
            return jsonify(ok=True, scan_id=state["scan_id"], condition=None)

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        if request.path.startswith("/api/"):
            response = error.get_response()
            response.data = app.json.dumps({"ok": False, "error": error.description})
            response.content_type = "application/json"
            return response
        return error

    @app.after_request
    def response_headers(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
