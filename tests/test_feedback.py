from app.feedback import classify_feedback, disagree_response, is_disagreed


class TestIsDisagreed:
    def test_exact_match(self):
        assert is_disagreed("disagree") is True

    def test_case_insensitive(self):
        assert is_disagreed("DISAGREE") is True

    def test_with_exclamation(self):
        assert is_disagreed("disagree!") is True

    def test_embedded_in_sentence(self):
        assert is_disagreed("I disagree with this") is True

    def test_unrelated_text(self):
        assert is_disagreed("great catch") is False

    def test_empty_string(self):
        assert is_disagreed("") is False

    def test_whitespace_only(self):
        assert is_disagreed("   ") is False


class TestClassifyFeedback:
    def test_disagree(self):
        assert classify_feedback("disagree") == "negative"

    def test_case_insensitive(self):
        assert classify_feedback("DISAGREE") == "negative"

    def test_no_signal(self):
        assert classify_feedback("I'll fix this later") == "positive"

    def test_empty_string(self):
        assert classify_feedback("") == "positive"

    def test_whitespace_only(self):
        assert classify_feedback("   ") == "positive"

    def test_arbitrary_reply_is_positive(self):
        assert classify_feedback("I'll fix this") == "positive"


class TestDisagreeResponse:
    def test_mentions_quality_metric(self):
        response = disagree_response()
        assert response
        assert "review-quality metric" in response
        assert "disagree" in response.lower()
