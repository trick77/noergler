_DISAGREE = "disagree"

_DISAGREE_RESPONSE = (
    "This disagree has been recorded as a negative signal against my overall "
    "review-quality metric. Please reserve `disagree` for findings that are "
    "**technically wrong** or **hallucinated**. If that was the case here, "
    "sincere apologies 🙏"
)


def is_disagreed(text: str) -> bool:
    return _DISAGREE in text.strip().lower()


def disagree_response() -> str:
    return _DISAGREE_RESPONSE


def classify_feedback(text: str) -> str:
    return "negative" if is_disagreed(text) else "positive"
