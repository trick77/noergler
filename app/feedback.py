import random

_DISAGREE = "disagree"

_FUN_RESPONSES = [
    "Ouch! I'll try harder next time 😅",
    "Noted — my AI feelings are only slightly hurt 🤖",
    "Fair enough, back to the drawing board 📝",
    "Well, even I have off days 🤷",
    "Message received — recalibrating snark levels 🔧",
    "I'll add that to my list of regrets 📋",
    "Tough crowd! Noted though 🎭",
    "Roger that — filing under 'room for improvement' 🗂️",
    "Oof. I'll do better, promise! 🫡",
    "Copy that — not my finest moment 😬",
]


def is_disagreed(text: str) -> bool:
    return _DISAGREE in text.strip().lower()


def random_response() -> str:
    return random.choice(_FUN_RESPONSES)


def classify_feedback(text: str) -> str:
    return "negative" if is_disagreed(text) else "positive"
