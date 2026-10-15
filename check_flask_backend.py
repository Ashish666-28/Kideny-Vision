"""Check Flask routing and conversation isolation using injected test services.

Copy beside app.py and run with your project's .venv Python. This does not use
Gemini, load trained weights, or connect to your running localhost server.
It verifies application behavior; it does not measure model or medical accuracy.
"""

import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from app import create_app
from chatbot import ChatbotError


class TestBot:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.on_answer = None
        self.kb = SimpleNamespace(topics=[
            {"id": name.upper() + "_001", "title": name + " explanation"}
            for name in ("normal", "stone", "cyst", "tumor")
        ])

    def answer(self, question, condition=None, history=None):
        self.calls.append((question, condition, [dict(item) for item in history or []]))
        if self.on_answer:
            self.on_answer()
        if self.fail:
            raise ChatbotError("Temporary test provider failure.")
        return {"answer": "Test reply " + str(len(self.calls))}


class BackendChecks(unittest.TestCase):
    def setUp(self):
        self.bot = TestBot()
        self.label = "stone"
        self.model_fail = False
        self.bad_result = False
        self.classifier_calls = 0

        def classifier(stream):
            self.classifier_calls += 1
            with Image.open(stream) as image:
                image.load()
            if self.model_fail:
                raise RuntimeError("private-model-detail")
            scores = {name: 0.01 for name in ("normal", "stone", "cyst", "tumor")}
            scores[self.label] = 0.97
            return {"label": "invalid" if self.bad_result else self.label,
                    "probabilities": scores}

        self.app = create_app(
            {"TESTING": True, "SECRET_KEY": "isolated-unit-test-key"},
            chatbot_factory=lambda: self.bot, classifier=classifier,
        )
        # Expected failure checks should not print misleading server error logs.
        self.app.logger.disabled = True
        self.client = self.app.test_client()
        self.context = self.client.get("/api/chat/session").get_json()
        self.headers = {"X-CSRF-Token": self.context["csrf_token"]}

    def post(self, route, data=None, client=None, context=None):
        client = client or self.client
        context = context or self.context
        return client.post(route, json=data or {},
                           headers={"X-CSRF-Token": context["csrf_token"]})

    def upload(self, content=None, filename="test.png"):
        if content is None:
            image = BytesIO()
            Image.new("L", (32, 24), color=128).save(image, format="PNG")
            content = image.getvalue()
        return self.client.post(
            "/api/predict", data={"image": (BytesIO(content), filename)},
            headers=self.headers,
        )

    def test_general_question_before_scan(self):
        response = self.post("/api/chat/message", {
            "message": "Which foods support kidney health?",
            "scan_id": self.context["scan_id"],
        })
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.bot.calls[-1][1])
        self.assertEqual(response.get_json()["context_source"], "general")

    def test_stateless_session_survives_another_instance(self):
        config = {
            "TESTING": True, "SECRET_KEY": "shared-deployment-secret",
            "STATELESS_SESSIONS": True,
        }
        first = create_app(config, chatbot_factory=lambda: self.bot,
                           classifier=lambda stream: {
                               "label": "stone",
                               "probabilities": {"normal": 0.01, "stone": 0.97,
                                                 "cyst": 0.01, "tumor": 0.01},
                           })
        second = create_app(config, chatbot_factory=lambda: self.bot)
        first_client, second_client = first.test_client(), second.test_client()
        initial = first_client.get("/api/chat/session").get_json()
        cookie = first_client.get_cookie("kidney_chat_session")
        second_client.set_cookie("kidney_chat_session", cookie.value)
        headers = {
            "X-CSRF-Token": initial["csrf_token"],
            "X-Conversation-State": initial["state_token"],
        }
        image = BytesIO()
        Image.new("L", (32, 24), color=128).save(image, format="PNG")
        prediction = first_client.post(
            "/api/predict", data={"image": (BytesIO(image.getvalue()), "test.png")},
            headers=headers,
        ).get_json()
        self.assertEqual(prediction["label"], "stone")
        headers["X-Conversation-State"] = prediction["state_token"]
        reply = second_client.post(
            "/api/chat/message",
            json={"message": "What does this mean?", "scan_id": prediction["scan_id"]},
            headers=headers,
        )
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(self.bot.calls[-1][1], "stone")
        headers["X-Conversation-State"] = reply.get_json()["state_token"]
        restored = first_client.get("/api/chat/session", headers=headers).get_json()
        self.assertEqual(restored["prediction"]["label"], "stone")
        self.assertEqual(len(restored["messages"]), 2)

    def test_prediction_does_not_call_gemini(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["label"], "stone")
        self.assertEqual(result["score_type"], "uncalibrated_softmax")
        self.assertNotEqual(result["scan_id"], self.context["scan_id"])
        self.assertEqual(self.bot.calls, [])

    def test_condition_is_server_prediction(self):
        result = self.upload().get_json()
        response = self.post("/api/chat/start", {
            "scan_id": result["scan_id"], "condition": "tumor",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.bot.calls[-1][1], "stone")
        self.assertEqual(response.get_json()["condition"], "stone")

    def test_manual_condition_cannot_start_a_scan(self):
        response = self.post("/api/chat/start", {
            "scan_id": self.context["scan_id"], "condition": "stone",
        })
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.bot.calls, [])

    def test_follow_up_receives_previous_question(self):
        scan_id = self.upload().get_json()["scan_id"]
        self.post("/api/chat/message", {"message": "What food should I eat?", "scan_id": scan_id})
        response = self.post("/api/chat/message", {"message": "Can you list it?", "scan_id": scan_id})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.bot.calls[-1][1], "stone")
        self.assertEqual(self.bot.calls[-1][2][0]["content"], "What food should I eat?")

    def test_initial_explanation_is_cached(self):
        scan_id = self.upload().get_json()["scan_id"]
        first = self.post("/api/chat/start", {"scan_id": scan_id})
        second = self.post("/api/chat/start", {"scan_id": scan_id})
        self.assertEqual(first.get_json(), second.get_json())
        self.assertEqual(len(self.bot.calls), 1)
        messages = self.client.get("/api/chat/session").get_json()["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")

    def test_new_scan_clears_history_and_rejects_old_id(self):
        old_id = self.upload().get_json()["scan_id"]
        self.post("/api/chat/message", {"message": "What food should I eat?", "scan_id": old_id})
        self.label = "cyst"
        new_id = self.upload().get_json()["scan_id"]
        stale = self.post("/api/chat/message", {"message": "Can you list it?", "scan_id": old_id})
        self.assertEqual(stale.status_code, 409)
        response = self.post("/api/chat/message", {"message": "What does this mean?", "scan_id": new_id})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.bot.calls[-1][1], "cyst")
        self.assertEqual(self.bot.calls[-1][2], [])

    def test_browsers_have_separate_history(self):
        scan_id = self.upload().get_json()["scan_id"]
        self.post("/api/chat/message", {"message": "My kidney question", "scan_id": scan_id})
        other = self.app.test_client()
        context = other.get("/api/chat/session").get_json()
        self.assertIsNone(context["condition"])
        self.assertEqual(context["messages"], [])
        rejected = self.post("/api/chat/message", {"message": "A question", "scan_id": scan_id},
                             client=other, context=context)
        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(len(self.bot.calls), 1)

    def test_chat_failure_keeps_classification(self):
        scan_id = self.upload().get_json()["scan_id"]
        self.bot.fail = True
        failed = self.post("/api/chat/start", {"scan_id": scan_id})
        self.assertEqual(failed.status_code, 503)
        context = self.client.get("/api/chat/session").get_json()
        self.assertEqual(context["prediction"]["label"], "stone")
        self.assertEqual(context["scan_id"], scan_id)
        self.bot.fail = False
        response = self.post("/api/chat/message", {"message": "What next?", "scan_id": scan_id})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.bot.calls[-1][1], "stone")

    def test_model_failure_invalidates_old_scan(self):
        old_id = self.upload().get_json()["scan_id"]
        self.model_fail = True
        failed = self.upload()
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("private-model-detail", failed.get_data(as_text=True))
        context = self.client.get("/api/chat/session").get_json()
        self.assertIsNone(context["prediction"])
        self.assertIsNone(context["condition"])
        self.assertNotEqual(context["scan_id"], old_id)

    def test_malformed_image_does_not_run_model(self):
        self.assertEqual(self.upload(b"not an image", "test.jpg").status_code, 400)
        self.assertEqual(self.classifier_calls, 0)

    def test_malformed_model_output_is_rejected(self):
        self.bad_result = True
        response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(self.client.get("/api/chat/session").get_json()["prediction"])

    def test_missing_csrf_is_rejected(self):
        response = self.client.post("/api/chat/message", json={
            "message": "Hello", "scan_id": self.context["scan_id"],
        })
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.bot.calls, [])

    def test_file_size_limit(self):
        self.app.config["MAX_IMAGE_BYTES"] = 1
        response = self.upload()
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.classifier_calls, 0)

    def test_invalid_csrf_text_is_rejected(self):
        response = self.client.post("/api/chat/message", json={
            "message": "Hello", "scan_id": self.context["scan_id"],
        }, headers={"X-CSRF-Token": "\u00e9"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.bot.calls, [])

    def test_reset_removes_result_and_history(self):
        old_id = self.upload().get_json()["scan_id"]
        self.post("/api/chat/message", {"message": "A question", "scan_id": old_id})
        response = self.post("/api/chat/reset")
        self.assertEqual(response.status_code, 200)
        context = self.client.get("/api/chat/session").get_json()
        self.assertEqual(context["messages"], [])
        self.assertIsNone(context["prediction"])
        self.assertNotEqual(context["scan_id"], old_id)

    def test_concurrent_request_does_not_mix_histories(self):
        nested = []
        def while_answering():
            nested.append(self.post("/api/chat/reset").status_code)
        self.bot.on_answer = while_answering
        response = self.post("/api/chat/message", {
            "message": "A question", "scan_id": self.context["scan_id"],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(nested, [409])

    def test_idle_session_expires(self):
        self.app.config["CHAT_IDLE_SECONDS"] = 1
        with patch("app.monotonic", return_value=10**12):
            response = self.post("/api/chat/message", {
                "message": "A question", "scan_id": self.context["scan_id"],
            })
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.bot.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
