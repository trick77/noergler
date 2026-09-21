package inference

import "testing"

// line: null drops the finding, confidence: null keeps it. line is required
// and confidence is optional, and Go would unmarshal null into an int as 0
// without erroring, so both need an explicit null check.
func TestNullHandlingInFindings(t *testing.T) {
	dropped := ParseReview(`{"findings":[{"file":"a.py","line":null,"severity":"issue","comment":"c"}]}`)
	if len(dropped.Findings) != 0 {
		t.Errorf("a null line must drop the finding, got %+v", dropped.Findings)
	}

	kept := ParseReview(`{"findings":[{"file":"a.py","line":1,"severity":"issue","comment":"c","confidence":null}]}`)
	if len(kept.Findings) != 1 {
		t.Fatalf("a null confidence must keep the finding, got %d", len(kept.Findings))
	}
	if kept.Findings[0].Confidence != nil {
		t.Errorf("Confidence = %v, want nil for an explicit null", kept.Findings[0].Confidence)
	}
}
