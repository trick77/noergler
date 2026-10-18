package inference

import "github.com/trick77/noergler/internal/lang"

// PlaceholderOutputLanguage is the language block, the same in all three
// templates. It sits right before "## Untrusted input": it is our text, not
// the author's, and the prefix up to it stays the same for every PR in that
// language, so the gateway's prefix cache forks once per language, not per PR.
const PlaceholderOutputLanguage = "{output_language}"

// The prompts stay English; only what a human reads follows the language.
// The enums and keys are what the code branches on, so they never move.
const outputLanguageKeep = "Keep verbatim, never translated: code, identifiers, file paths, JSON keys, " +
	"every enum value (`severity`, `intent`, the verdict `decision` and the like) and the `suggestion` field. " +
	"`agents_md_line` is written in the language of the project guidelines (English when there are none). " +
	"Fixed sentences the rules quote in English are written in this language too. " +
	"Earlier findings, replies or disputes shown to you may be in another language: " +
	"the same issue in another language is the same finding, never a new one."

const (
	outputLanguageDE = "## Output language\n\n" +
		"Write every free-text value a human reads (`headline`, `comment`, the summary texts, " +
		"`answer`, `reply`) in Swiss Standard German. Swiss spelling: always \"ss\", never \"ß\". " +
		"Use the generic masculine: no gendered double forms, no \":innen\" or \"*innen\". " +
		"Technical terms developers say in English stay English (Review, Diff, Commit, Token). " +
		outputLanguageKeep
	outputLanguageEN = "## Output language\n\n" +
		"Write every free-text value a human reads (`headline`, `comment`, the summary texts, " +
		"`answer`, `reply`) in English. " +
		outputLanguageKeep
)

// OutputLanguageBlock is what {output_language} becomes.
func OutputLanguageBlock(l lang.Lang) string {
	if l == lang.EN {
		return outputLanguageEN
	}
	return outputLanguageDE
}
