package config

import "slices"

// The prompt-template settings removed when the prompts were compiled in. A
// deployment still setting one would otherwise believe its own prompt is
// live, so each is named: a warning for an env var, the disable reason for a
// teams.yaml key.
var (
	removedPromptEnv = []string{
		"REVIEW_PROMPT_TEMPLATE", "REVIEW_MENTION_PROMPT_TEMPLATE", "REVIEW_FINDING_REPLY_PROMPT_TEMPLATE",
	}
	removedPromptKeys = []string{
		"review_prompt_template", "mention_prompt_template", "finding_reply_prompt_template",
	}
)

// RemovedPromptReason is what a removed prompt setting is told.
const RemovedPromptReason = "no longer read: the prompts are compiled into the binary"

// RemovedPromptSettings returns the removed prompt-template variables that
// are still set, for the startup warning.
func RemovedPromptSettings(lookup func(string) (string, bool)) []string {
	var set []string
	for _, name := range removedPromptEnv {
		if _, ok := lookup(name); ok {
			set = append(set, name)
		}
	}
	return set
}

func isRemovedPromptKey(k string) bool { return slices.Contains(removedPromptKeys, k) }
