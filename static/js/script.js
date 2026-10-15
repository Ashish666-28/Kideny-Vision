/* Connect the existing KidneyVision controls to app.py. No API key belongs here. */
(() => {
    "use strict";

    async function init() {
        const byId = (id) => document.getElementById(id);
        const ui = {
            file: byId("image-input"), choose: byId("choose-image"), uploadBox: byId("upload-box"),
            analyze: byId("analyze-image"), preview: byId("image-preview"),
            placeholder: byId("upload-placeholder"), changeHint: byId("upload-change"), filename: byId("file-name"),
            uploadError: byId("upload-error"), result: byId("result-status"),
            scanPanel: document.querySelector(".scan-panel"),
            results: document.querySelector(".results"), resultBody: byId("result-body"),
            resultClass: byId("result-class"), resultScore: byId("result-score"),
            steps: [...document.querySelectorAll(".scan-steps li")],
            messages: byId("chat-messages"), input: byId("chat-input"),
            send: byId("send-message"), status: document.querySelector(".chat-status"),
        };
        if (Object.values(ui).some((item) => !item) || ui.steps.length !== 3) {
            console.error("KidneyVision: the page controls do not match script.js.");
            if (ui.status) ui.status.textContent = "Page setup failed. Use the matching index.html and script.js.";
            return;
        }

        const labels = ["normal", "stone", "cyst", "tumor"];
        const state = {csrf: null, scanId: null, ready: false, busy: null, file: null, previewUrl: null};
        const title = (value) => value[0].toUpperCase() + value.slice(1);

        function setProgress(stage) {
            const order = ["selected", "analyzing", "complete"];
            const current = order.indexOf(stage);
            ui.scanPanel.classList.toggle("is-analyzing", stage === "analyzing");
            ui.steps.forEach((step, index) => {
                step.classList.toggle("is-current", index === current);
                step.classList.toggle("is-done", current >= 0 && index < current);
            });
        }

        function controls() {
            const busy = Boolean(state.busy);
            ui.choose.disabled = busy;
            ui.uploadBox.disabled = busy;
            ui.file.disabled = busy;
            ui.analyze.disabled = busy || !state.ready || !state.file;
            ui.analyze.classList.toggle("primary", !ui.analyze.disabled);
            ui.analyze.classList.toggle("secondary", ui.analyze.disabled);
            ui.analyze.textContent = state.busy === "analyzing" ? "Analyzing..." : "Analyze image";
            ui.input.disabled = !state.ready || ["selecting", "analyzing", "connecting"].includes(state.busy);
            ui.send.disabled = busy || !state.ready || !ui.input.value.trim();
            ui.messages.setAttribute("aria-busy", String(busy));
        }

        function status(text, retry = false) {
            ui.status.replaceChildren(document.createTextNode(text));
            if (retry) {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "button secondary";
                button.textContent = "Reconnect";
                button.style.marginLeft = "8px";
                button.addEventListener("click", () => connect(true));
                ui.status.append(button);
            }
        }

        function bubble(text, role = "assistant") {
            const box = document.createElement("div");
            box.className = "message " + (role === "user" ? "outgoing" : "incoming");
            const paragraph = document.createElement("p");
            paragraph.textContent = String(text);
            paragraph.style.whiteSpace = "pre-wrap";
            const label = document.createElement("span");
            label.className = "message-label";
            label.textContent = role === "user" ? "You" : role === "system" ? "Notice" : "Kidney Assistant";
            box.append(paragraph, label);
            ui.messages.append(box);
            ui.messages.scrollTop = ui.messages.scrollHeight;
            return box;
        }

        function resetMessages(text) {
            ui.messages.replaceChildren();
            const date = document.createElement("div");
            date.className = "date-label";
            date.textContent = "Today";
            ui.messages.append(date);
            if (text) bubble(text);
        }

        function clearScores(text = "Choose an image to begin.") {
            ui.resultBody.hidden = true;
            delete ui.results.dataset.label;
            ui.resultClass.textContent = "\u2014";
            ui.resultScore.textContent = "\u2014";
            ui.result.textContent = text;
            setProgress(state.file ? "selected" : null);
        }

        function showPrediction(data, previous = false) {
            if (!labels.includes(data.label) || !data.probabilities ||
                !labels.every((name) => Number.isFinite(data.probabilities[name]) &&
                    data.probabilities[name] >= 0 && data.probabilities[name] <= 1)) {
                throw new Error("The server returned an invalid prediction.");
            }
            ui.results.dataset.label = data.label;
            ui.resultClass.textContent = title(data.label);
            ui.resultScore.textContent = (data.probabilities[data.label] * 100).toFixed(2) + "%";
            ui.resultBody.hidden = false;
            ui.result.textContent = previous ? "Previous scan result" : "The model has analyzed your CT image.";
            setProgress("complete");
        }

        async function api(path, body = null, multipart = false) {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), body === null ? 15000 : 150000);
            const options = {credentials: "same-origin", cache: "no-store", signal: controller.signal};
            if (body !== null) {
                options.method = "POST";
                options.headers = {"X-CSRF-Token": state.csrf || ""};
                if (multipart) options.body = body;
                else {
                    options.headers["Content-Type"] = "application/json";
                    options.body = JSON.stringify(body);
                }
            }
            try {
                const response = await fetch(path, options);
                let data;
                try { data = await response.json(); }
                catch {
                    throw new Error("Flask did not return API data. Save the updated app.py and restart Flask.");
                }
                if (!response.ok || data.ok !== true) {
                    const error = new Error(response.status === 404
                        ? "The Flask routes are missing. Save the updated file as app.py and restart Flask."
                        : data.error || "The request failed.");
                    error.status = response.status;
                    throw error;
                }
                return data;
            } catch (error) {
                if (error.name === "AbortError") throw new Error("The request timed out. Wait for it to finish before trying again.");
                if (error instanceof TypeError) throw new Error("Cannot reach Flask. Keep app.py running and refresh this page.");
                throw error;
            } finally { clearTimeout(timer); }
        }

        function needReconnect(error) {
            if ([403, 404, 409].includes(error.status)) {
                state.ready = false;
                status(error.message, true);
            } else status(error.message);
        }

        function requireMatchingReply(answer, scanId) {
            if (answer.scan_id !== scanId) {
                const error = new Error("The reply belongs to an older scan. Reconnect before continuing.");
                error.status = 409;
                throw error;
            }
            if (typeof answer.answer !== "string" || !answer.answer.trim()) {
                throw new Error("The chatbot returned an empty answer. Please try your question again.");
            }
        }

        async function connect(clearFile = false) {
            if (state.busy) return;
            state.busy = "connecting";
            state.ready = false;
            controls();
            status("Connecting...");
            if (clearFile) {
                state.file = null;
                ui.file.value = "";
                ui.input.value = "";
                if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
                state.previewUrl = null;
                ui.preview.removeAttribute("src");
                ui.preview.hidden = true;
                ui.placeholder.hidden = false;
                ui.changeHint.hidden = true;
                ui.filename.textContent = "No image selected";
            }
            try {
                const data = await api("/api/chat/session");
                if (typeof data.csrf_token !== "string" || typeof data.scan_id !== "string") {
                    throw new Error("The Flask backend is out of date. Save the updated app.py and restart Flask.");
                }
                if (data.busy) throw new Error("A previous request is still running. Wait, then reconnect.");
                state.csrf = data.csrf_token;
                state.scanId = data.scan_id;
                clearScores();
                if (data.prediction) showPrediction(data.prediction, true);
                resetMessages();
                const messages = Array.isArray(data.messages) ? data.messages : [];
                for (const message of messages) {
                    if (["user", "assistant"].includes(message.role) && typeof message.content === "string")
                        bubble(message.content, message.role);
                }
                if (!messages.length) bubble("Hello! You can ask kidney-health questions or upload a CT image for an explanation of its result.");
                state.ready = true;
                status(data.condition ? "Questions will use your " + title(data.condition) + " scan context."
                    : "Ask a kidney-health question, or choose a CT image.");
            } catch (error) { status(error.message, true); }
            finally { state.busy = null; controls(); }
        }

        function openImagePicker() {
            if (!state.busy) ui.file.click();
        }

        ui.choose.addEventListener("click", openImagePicker);
        ui.uploadBox.addEventListener("click", openImagePicker);
        ui.file.addEventListener("change", async () => {
            const file = ui.file.files[0];
            if (!file || state.busy) return;
            ui.uploadError.textContent = "";
            if (!/\.(png|jpe?g)$/i.test(file.name) || file.size === 0 || file.size > 10 * 1024 * 1024) {
                ui.uploadError.textContent = "Choose a nonempty PNG or JPG image smaller than 10 MB.";
                ui.file.value = "";
                return;
            }
            state.file = file;
            if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
            state.previewUrl = URL.createObjectURL(file);
            ui.preview.src = state.previewUrl;
            ui.preview.hidden = false;
            ui.placeholder.hidden = true;
            ui.changeHint.hidden = false;
            ui.filename.textContent = file.name;
            ui.input.value = "";
            clearScores("Image selected. Click Analyze image.");
            resetMessages("Image selected. Analyze it for a result, or ask a general kidney-health question.");
            if (!state.ready) { controls(); return; }
            state.busy = "selecting";
            controls();
            try {
                const data = await api("/api/chat/reset", {});
                state.scanId = data.scan_id;
                status("Ask a kidney-health question, or analyze the selected image.");
            } catch (error) {
                state.ready = false;
                ui.uploadError.textContent = error.message;
                status(error.message, true);
            } finally { state.busy = null; controls(); }
        });

        ui.analyze.addEventListener("click", async () => {
            if (state.busy || !state.ready || !state.file) return;
            state.busy = "analyzing";
            controls();
            ui.uploadError.textContent = "";
            clearScores("Analyzing your CT image...");
            setProgress("analyzing");
            resetMessages();
            status("Analyzing the image...");
            let typing = null;
            let predicted = false;
            try {
                const form = new FormData();
                form.append("image", state.file, state.file.name);
                const result = await api("/api/predict", form, true);
                if (typeof result.scan_id !== "string") throw new Error("The prediction did not include its scan context.");
                state.scanId = result.scan_id;
                showPrediction(result);
                predicted = true;
                state.busy = "advice";
                controls();
                status("Preparing an explanation of your " + title(result.label) + " result...");
                typing = bubble("Preparing your explanation...", "system");
                const answer = await api("/api/chat/start", {scan_id: state.scanId});
                requireMatchingReply(answer, state.scanId);
                typing.remove();
                typing = null;
                bubble(answer.answer);
                status("You can ask questions about your " + title(result.label) + " result.");
            } catch (error) {
                if (typing) typing.remove();
                if (predicted) {
                    bubble("Your scan result is shown. The explanation could not be completed: " + error.message, "system");
                    needReconnect(error);
                } else {
                    clearScores("Analysis could not be completed.");
                    ui.uploadError.textContent = error.message;
                    // Inference may already have invalidated the old scan ID.
                    state.ready = false;
                    status(error.message, true);
                }
            } finally { state.busy = null; controls(); }
        });

        async function send() {
            const question = ui.input.value.trim();
            if (!question || state.busy || !state.ready) return;
            const scanId = state.scanId;
            bubble(question, "user");
            ui.input.value = "";
            ui.input.style.height = "";
            state.busy = "message";
            controls();
            status("Preparing your answer...");
            const typing = bubble("Kidney Assistant is typing...", "system");
            try {
                const answer = await api("/api/chat/message", {message: question, scan_id: scanId});
                requireMatchingReply(answer, scanId);
                typing.remove();
                bubble(answer.answer);
                status("Ask another question whenever you are ready.");
            } catch (error) {
                typing.remove();
                bubble("Your question could not be answered: " + error.message, "system");
                if (!ui.input.value.trim()) ui.input.value = question;
                needReconnect(error);
            } finally { state.busy = null; controls(); }
        }

        ui.send.addEventListener("click", send);
        ui.input.addEventListener("input", () => {
            ui.input.style.height = "auto";
            ui.input.style.height = Math.min(ui.input.scrollHeight, 140) + "px";
            controls();
        });
        ui.input.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
                event.preventDefault();
                send();
            }
        });
        window.addEventListener("beforeunload", () => {
            if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
        });
        if (window.matchMedia("(hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)").matches) {
            document.querySelectorAll(".tilt-surface").forEach((surface) => {
                surface.addEventListener("pointermove", (event) => {
                    const bounds = surface.getBoundingClientRect();
                    const horizontal = (event.clientX - bounds.left) / bounds.width - 0.5;
                    const vertical = (event.clientY - bounds.top) / bounds.height - 0.5;
                    surface.style.setProperty("--rotate-x", (-vertical * 3).toFixed(2) + "deg");
                    surface.style.setProperty("--rotate-y", (horizontal * 3).toFixed(2) + "deg");
                });
                surface.addEventListener("pointerleave", () => {
                    surface.style.removeProperty("--rotate-x");
                    surface.style.removeProperty("--rotate-y");
                });
            });
        }
        controls();
        await connect();
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
    else init();
})();
