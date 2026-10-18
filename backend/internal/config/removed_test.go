package config

import (
	"slices"
	"testing"
)

func TestRemovedPromptSettingsNamesWhatIsStillSet(t *testing.T) {
	env := map[string]string{"REVIEW_PROMPT_TEMPLATE": "/app/prompts/review.txt", "REVIEW_MAX_COMMENTS": "5"}
	got := RemovedPromptSettings(func(k string) (string, bool) { v, ok := env[k]; return v, ok })
	if !slices.Equal(got, []string{"REVIEW_PROMPT_TEMPLATE"}) {
		t.Errorf("got %v", got)
	}
	if got := RemovedPromptSettings(func(string) (string, bool) { return "", false }); got != nil {
		t.Errorf("nothing set: %v", got)
	}
}
