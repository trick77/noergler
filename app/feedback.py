NEGATIVE_SIGNALS = {
    "\U0001f44e",  # thumbsdown
    "-1",
    "false positive",
    "wrong",
    "not helpful",
    "disagree",
    "noise",
}


def classify_feedback(text: str) -> str:
    lowered = text.strip().lower()
    for signal in NEGATIVE_SIGNALS:
        if signal in lowered:
            return "negative"
    return "positive"
