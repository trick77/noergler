package inference

import (
	"encoding/json"
	"regexp"
	"strings"

	"github.com/trick77/noergler/internal/diff"
)

// PreExistingBug is a real bug the model noticed in code the PR did not
// change. It is never posted to the PR (decided): the PR's author did not
// write it. The run page lists it with its explanation and a link to the
// line, so somebody can pick it up.
type PreExistingBug struct {
	File        string
	Line        int
	Headline    string
	Explanation string
	Evidence    []string
}

// Limits on what a pre-existing bug may carry onto the page: a few, short.
const (
	maxPreExisting         = 5
	maxPreExistingHeadline = 200
	maxPreExistingProse    = 1000
)

// parsePreExisting reads one `pre_existing` item, reporting whether it is
// usable at all: a file, a line, a headline and an explanation.
func parsePreExisting(item json.RawMessage) (PreExistingBug, bool) {
	var probe struct {
		File        *string          `json:"file"`
		Line        *json.RawMessage `json:"line"`
		Headline    *string          `json:"headline"`
		Explanation *string          `json:"explanation"`
		Evidence    *json.RawMessage `json:"evidence"`
	}
	if err := json.Unmarshal(item, &probe); err != nil {
		return PreExistingBug{}, false
	}
	if probe.File == nil || probe.Line == nil || probe.Headline == nil || probe.Explanation == nil {
		return PreExistingBug{}, false
	}
	line, ok := coerceInt(*probe.Line)
	if !ok {
		return PreExistingBug{}, false
	}
	b := PreExistingBug{
		File:        *probe.File,
		Line:        line,
		Headline:    TruncateRunes(strings.TrimSpace(*probe.Headline), maxPreExistingHeadline),
		Explanation: TruncateRunes(strings.TrimSpace(*probe.Explanation), maxPreExistingProse),
	}
	if b.Headline == "" || b.Explanation == "" {
		return PreExistingBug{}, false
	}
	if probe.Evidence != nil {
		b.Evidence = coerceEvidence(*probe.Evidence)
	}
	return b, true
}

// DroppedPreExisting is a pre-existing claim validation refused, and why.
type DroppedPreExisting struct {
	Bug    PreExistingBug
	Reason string
}

// ValidatePreExisting keeps the claims whose evidence is real: in the file
// the prompt showed (its diff, its context or its full content), and none of
// it on a line the PR added or removed, which would make it the PR's own bug
// and a finding, not this. There is no in-diff rule: unchanged code is the
// point.
//
// prAdded is every line the whole PR added, by file, read off the full PR
// diff on an incremental review (the focused diff is one push there, and an
// earlier push's lines read as unchanged); nil on a full review, whose diff
// is the whole PR. A file missing from it is one the PR adds nothing to. The
// caller drops every claim when that diff could not be read.
//
// Known limit, accepted: lines compare by normalised text, so a real old bug
// whose line reads exactly like a line the PR added is dropped. The other
// way round, an author's own bug shown as old, is the worse mistake.
//
// The line is moved to the evidence when the evidence is a shown line and
// the cited one is not near it; the final line must be 1 or more and not a
// line the PR added. Evidence found only in the full content has no line
// number in the index (hashes only), so the cited line stands.
//
// Code in the headline or explanation is fine (the page is reached only by
// its link token); one whose text or evidence carries a secret-looking token
// is dropped:
// blunt rules, like ValidateFindings, never a judge model. At most
// maxPreExisting are kept, in the model's order.
func ValidatePreExisting(bugs []PreExistingBug, idx diff.AnchorIndex, prAdded map[string]map[uint64]bool) (kept []PreExistingBug, dropped []DroppedPreExisting) {
	for _, b := range bugs {
		b.File = ShownPath(b.File, idx)
		reason, line := validatePreExisting(b, idx, prAdded)
		if reason == "" && len(kept) >= maxPreExisting {
			reason = "over_limit"
		}
		if reason != "" {
			dropped = append(dropped, DroppedPreExisting{Bug: b, Reason: reason})
			continue
		}
		b.Line = line
		kept = append(kept, b)
	}
	return kept, dropped
}

// tokenRunRE is a run of 20 or more token characters with no space or dot:
// a dot ends a run, so dotted code (`base64.RawURLEncoding.EncodeToString`)
// is judged by its names, each a word. ASCII on purpose, against the
// adapters rule: credentials are ASCII, and a run that a non-ASCII letter
// breaks is a word, not a key.
var tokenRunRE = regexp.MustCompile(`[A-Za-z0-9_+/=-]{20,}`)

// knownSecretRE is a credential by its shape: an AWS key id, a GitHub or
// Slack token, an OpenAI or Stripe-style key, a PEM private key.
var knownSecretRE = regexp.MustCompile(`AKIA[0-9A-Z]{16}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bxox[abprs]-[A-Za-z0-9-]{10,}|\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}|\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY`)

// credentialRE is a credential by where it sits: a secret-named key given a
// quoted literal (`password = "admin123"`, `"token": "x"`), or a whole
// properties/YAML line with an unquoted value (`db.password=admin123`). The
// shapes above miss a short default password, the commonest committed one.
// A value read at runtime (`getenv(...)`, `${...}` lines) or an empty one
// does not match.
var credentialRE = regexp.MustCompile(`(?i)(?:passw(?:or)?d|pwd|secret|token|api_?key)[\w.-]*["']?\s*(?::=|[:=])\s*(?:"[^"\s]+"|'[^'\s]+')|(?im)^\s*[\w.-]*(?:passw(?:or)?d|pwd|secret|token|api_?key)[\w.-]*\s*[:=]\s*[^\s"'$<{#;()]+\s*$`)

// minSecretAlnum and minSecretShifts make a long run secret-like: random
// text (a key, a hash, base64) flips between upper case, lower case and
// digits at almost every other character; an identifier, a path or a
// German compound flips at its word boundaries only
// ("handleConnectionRecoveryTimeout": 6 in 31, "Sha256DigestCalculator": 6 in
// 22). Characters that are neither are skipped, so a path's slashes do not
// count.
const (
	minSecretAlnum  = 20
	minSecretShifts = 0.3
)

// HasSecret reports a secret-like token in model or author text: a known
// credential shape, a literal given to a secret-named key, or a long run
// that shifts character class as random text does. Blunt, like every
// validation rule: a commit hash or a UUID counts too, and a short or
// unusually regular key outside a secret-named assignment passes.
func HasSecret(s string) bool {
	if knownSecretRE.MatchString(s) || credentialRE.MatchString(s) {
		return true
	}
	for _, run := range tokenRunRE.FindAllString(s, -1) {
		alnum, shifts, prev := 0, 0, 0
		for _, r := range run {
			c := 0
			switch {
			case r >= 'a' && r <= 'z':
				c = 1
			case r >= 'A' && r <= 'Z':
				c = 2
			case r >= '0' && r <= '9':
				c = 3
			default:
				continue
			}
			alnum++
			if prev != 0 && c != prev {
				shifts++
			}
			prev = c
		}
		if alnum >= minSecretAlnum && float64(shifts) >= minSecretShifts*float64(alnum) {
			return true
		}
	}
	return false
}

// AnySecret is HasSecret over several texts.
func AnySecret(texts ...string) bool {
	for _, t := range texts {
		if HasSecret(t) {
			return true
		}
	}
	return false
}

func validatePreExisting(b PreExistingBug, idx diff.AnchorIndex, prAdded map[string]map[uint64]bool) (string, int) {
	fa, ok := idx[b.File]
	if !ok {
		return string(DropUnknownFile), 0
	}
	if AnySecret(append([]string{b.Headline, b.Explanation}, b.Evidence...)...) {
		return "secret_in_text", 0
	}
	history := prAdded[b.File]
	var shown []uint64
	quoted := 0
	for _, e := range b.Evidence {
		if strings.TrimSpace(e) == "" {
			continue
		}
		h, where := matchEvidence(e, fa)
		if where == onRemovedSide && fa.InContent(h) {
			// The PR removed a line that reads the same, but this one is
			// still in the file.
			where = inContentOnly
		}
		switch {
		case where == notFound && isAnnotation(e):
			// The model's own note or an elision, as in ValidateFindings.
			continue
		case where == notFound:
			return string(DropEvidenceNotFound), 0
		case where == onRemovedSide:
			// The PR deleted it: whatever was wrong there is gone.
			return "evidence_removed_by_pr", 0
		case addedLine(h, fa.Shown) || history[h]:
			return "evidence_on_added_line", 0
		case where == onNewSide:
			shown = append(shown, h)
		}
		quoted++
	}
	if quoted == 0 {
		return string(DropNoEvidence), 0
	}
	line := b.Line
	if len(shown) > 0 && !nearAny(line, shown, fa.Shown) {
		if at := linesWithHash(shown[0], fa.Shown); len(at) == 1 {
			line = at[0]
		}
	}
	if line < 1 {
		return "bad_line", 0
	}
	if s, ok := fa.Shown[line]; ok && (s.Added || history[s.Hash]) {
		return "line_added_by_pr", 0
	}
	return "", line
}
