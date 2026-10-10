r"""Automatic scan-result explanation followed by patient questions.

Place beside chatbot.py and knowledge_base.py.
Manual terminal test: .\.venv\Scripts\python.exe scan_chat.py --condition stone

Flask integration contract:
    conversation = ScanConversation()  # one instance per patient/session
    first_reply = conversation.start(predicted_label)
    next_reply = conversation.ask(message, scan_id=first_reply['scan_id'])

Keep these objects on the server. Never share one conversation between patients.
This module supplies conversation logic; it does not create Flask routes or
classify images. The terminal condition is a manually selected test label.
"""

import argparse
from threading import RLock
from uuid import uuid4

from chatbot import CONDITIONS, ChatbotError, KidneyChatbot, initial_explanation_question


class ScanConversation:
    """A single patient's active scan and recent chat history."""

    def __init__(self, chatbot=None):
        self.bot = chatbot if chatbot is not None else KidneyChatbot()
        self._lock = RLock()
        self._condition = None
        self._scan_id = None
        self._history = []
        self._ready = False

    def start(self, predicted_label):
        """Call after the trained classifier returns a label for a new scan."""
        if not isinstance(predicted_label, str):
            raise ValueError('The prediction label must be text.')
        condition = predicted_label.strip().lower()
        if condition not in CONDITIONS:
            raise ValueError('Prediction must be stone, normal, cyst or tumor.')

        overview = next(
            (topic for topic in self.bot.kb.topics
             if topic['id'] == f'{condition.upper()}_001'), None
        )
        if overview is None:
            raise ValueError(f'The knowledge base is missing the {condition} overview.')

        # Including the real overview title helps retrieval find an explanation
        # of the result as well as the condition's food and follow-up guidance.
        question = initial_explanation_question(overview['title'], condition)

        with self._lock:
            # Invalidate the previous scan even if generating the new reply fails.
            self._scan_id = uuid4().hex
            self._condition = condition
            self._history = []
            self._ready = False
            result = self.bot.answer(question, condition=condition, history=[])
            self._remember(question, result['answer'])
            self._ready = True
            return self._response(result)

    def ask(self, question, *, scan_id):
        """Answer a follow-up for the same scan, preserving recent dialogue."""
        with self._lock:
            if not self._ready:
                raise ValueError('Start the conversation with a scan result first.')
            if scan_id != self._scan_id:
                raise ValueError('This question belongs to an older scan. Use the current scan.')
            result = self.bot.answer(
                question, condition=self._condition,
                history=[dict(message) for message in self._history],
            )
            self._remember(question.strip(), result['answer'])
            return self._response(result)

    def clear(self):
        with self._lock:
            self._condition = None
            self._scan_id = None
            self._history = []
            self._ready = False

    def _remember(self, question, answer):
        self._history.extend([
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer},
        ])
        self._history = self._history[-8:]

    def _response(self, result):
        return {**result, 'scan_id': self._scan_id, 'condition': self._condition}


def main():
    parser = argparse.ArgumentParser(description='Test automatic result-to-chat flow.')
    parser.add_argument('--condition', required=True, choices=CONDITIONS)
    args = parser.parse_args()
    try:
        conversation = ScanConversation()
        print(f'Manually selected test result: {args.condition.title()}')
        print('Generating the first explanation automatically...', flush=True)
        result = conversation.start(args.condition)
        print('\nBot:', result['answer'])
    except (ChatbotError, FileNotFoundError, ValueError) as error:
        print(f'Setup or initial reply error: {error}')
        return 1

    print('\nYou can now ask follow-up questions. Type exit to stop.')
    while True:
        try:
            question = input('\nYou: ').strip()
            if question.lower() in {'exit', 'quit'}:
                return 0
            if not question:
                continue
            print('Preparing your answer...', flush=True)
            reply = conversation.ask(question, scan_id=result['scan_id'])
            print('\nBot:', reply['answer'])
        except (ChatbotError, ValueError) as error:
            print(f'Chat error: {error}')
        except (EOFError, KeyboardInterrupt):
            print('\nChat stopped.')
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
