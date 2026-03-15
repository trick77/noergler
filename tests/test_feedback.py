from app.feedback import classify_feedback


class TestClassifyFeedback:
    def test_thumbsdown_emoji(self):
        assert classify_feedback("\U0001f44e") == "negative"

    def test_minus_one(self):
        assert classify_feedback("-1") == "negative"

    def test_false_positive(self):
        assert classify_feedback("this is a false positive") == "negative"

    def test_not_helpful(self):
        assert classify_feedback("not helpful") == "negative"

    def test_disagree(self):
        assert classify_feedback("disagree") == "negative"

    def test_noise(self):
        assert classify_feedback("this is just noise") == "negative"

    def test_case_insensitive(self):
        assert classify_feedback("FALSE POSITIVE") == "negative"

    def test_no_signal(self):
        assert classify_feedback("I'll fix this later") == "positive"

    def test_empty_string(self):
        assert classify_feedback("") == "positive"

    def test_whitespace_only(self):
        assert classify_feedback("   ") == "positive"

    def test_mixed_negative_wins(self):
        # Negative signals are checked first (more specific)
        assert classify_feedback("good catch but wrong") == "negative"

    def test_arbitrary_reply_is_positive(self):
        assert classify_feedback("I'll fix this") == "positive"
