_DISAGREE = "disagree"

_DISAGREE_RESPONSE = (
    "I'm truly sorry for the bad review comment — please forgive me! 🙏 "
    "This disagree has been recorded as a negative signal against my overall "
    "review-quality metric. Kindly only reply `disagree` when the finding is "
    "actually incorrect or hallucinated; use a regular reply for other concerns."
)


def is_disagreed(text: str) -> bool:
    return _DISAGREE in text.strip().lower()


def disagree_response() -> str:
    return _DISAGREE_RESPONSE


def classify_feedback(text: str) -> str:
    return "negative" if is_disagreed(text) else "positive"
