package inference

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// VerdictDecisions is the accepted verdict enum. Anything else is dropped.
var VerdictDecisions = []string{"approve", "approve_with_followups", "request_changes"}

// ReviewSummary is the non-finding half of a review response.
//
// Every field is populated on every review: sentinel strings carry the
// "nothing to report" case rather than empty values, so the posted comment can
// render a stable section list.
//
// VerdictDecision defaults to "approve", not empty, and keeps that default
// when the model returns an unrecognised decision or nothing at all. Build a
// summary with NewReviewSummary, never a bare struct literal, or an
// unparseable response silently reports "request_changes"-shaped emptiness
// where Python reported approval.
type ReviewSummary struct {
	Overview            string
	Strengths           []string
	SecurityPerformance string
	TestCoverage        string
	VerdictDecision     string
	VerdictRationale    string
}

// DefaultVerdictDecision is the value a summary carries until the model
// supplies a recognised one.
const DefaultVerdictDecision = "approve"

// NewReviewSummary returns a summary with the Python dataclass defaults.
func NewReviewSummary() ReviewSummary {
	return ReviewSummary{VerdictDecision: DefaultVerdictDecision}
}

// ComplianceRequirement is one acceptance-criteria check. Both fields must be
// present and correctly typed or the item is skipped.
type ComplianceRequirement struct {
	Requirement string `json:"requirement"`
	Met         bool   `json:"met"`
}

// ParsedReview is the result of parsing a model review response.
//
// ComplianceRequirements is nil when the response could not be parsed at all,
// which is distinct from an empty slice meaning the model legitimately returned
// none. The timeout and 413 paths also emit the nil sentinel, so ParseFailed is
// what identifies a refusal or unparseable output specifically.
type ParsedReview struct {
	Findings               []ReviewFinding
	ComplianceRequirements []ComplianceRequirement
	Summary                ReviewSummary
	ParseFailed            bool
	// Diagnostics is what Python logged from inside the parser. ParseReview
	// stays a pure function, so it returns them instead and the caller, which
	// holds the request context, emits them bound to team and pr_tag.
	Diagnostics []ParseDiagnostic
}

// ParseDiagnostic is one operator-facing line the parser produced.
type ParseDiagnostic struct {
	Level   slog.Level
	Message string
}

// warn and fail append a diagnostic at Python's level for the same event.
func (p *ParsedReview) warn(format string, args ...any) {
	p.Diagnostics = append(p.Diagnostics,
		ParseDiagnostic{Level: slog.LevelWarn, Message: fmt.Sprintf(format, args...)})
}

func (p *ParsedReview) info(format string, args ...any) {
	p.Diagnostics = append(p.Diagnostics,
		ParseDiagnostic{Level: slog.LevelInfo, Message: fmt.Sprintf(format, args...)})
}

// parseErrorDiagnostic splits a decode failure the way Python's
// json.JSONDecodeError / isinstance(data, dict) pair does: malformed JSON
// reports the content prefix, while a well-formed non-object (an array, a
// scalar, or null) reports only that it is not an object.
func parseErrorDiagnostic(err error, content string) ParseDiagnostic {
	var syntax *json.SyntaxError
	if errors.As(err, &syntax) {
		return ParseDiagnostic{
			Level:   slog.LevelError,
			Message: "Failed to parse review response as JSON: " + truncateRunes(content, 200),
		}
	}
	return ParseDiagnostic{Level: slog.LevelError, Message: "Review response is not a JSON object"}
}

// decodesAs reports whether raw is present and holds a value of dst's type.
//
// It is Python's isinstance check. json.Unmarshal alone is not: a missing key
// yields a nil RawMessage and an explicit null both decode as a silent no-op,
// leaving the zero value in place, so `{"met": null}` would read as a real
// "not met" rather than a malformed item.
func decodesAs(raw json.RawMessage, dst any) bool {
	if len(raw) == 0 || isJSONNull(raw) {
		return false
	}
	return json.Unmarshal(raw, dst) == nil
}

// truncateRunes cuts s to at most n runes. Python's content[:200] slices
// characters, so the prefix is measured the same way.
func truncateRunes(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n])
}

// Severities are the only values a finding may carry. The JSON schema already
// constrains them; this is the belt-and-braces check, so downstream stats and
// label rendering can trust the value even if the model regresses past the
// enum.
var Severities = []string{"issue", "suggestion"}

// ReviewFinding is one finding from the model, mirroring the Python Pydantic
// model. File, Line, Severity and Comment are required; a finding missing any
// of them, or carrying an unknown severity, is skipped rather than failing the
// batch.
type ReviewFinding struct {
	File     string
	Line     int
	Severity string
	Comment  string

	// Confidence, Headline and Suggestion are optional. Nil means absent,
	// which is distinct from an empty value.
	Confidence *int
	Headline   *string
	Suggestion *string
}

// parseFinding validates one finding, reporting whether it survives.
//
// Python is Pydantic in lax mode, so a numeric string coerces to an int
// ("1" becomes 1) while a float with a fractional part does not. Both are
// reproduced: the first is a model quirk worth tolerating, the second would
// silently move a finding to the wrong line.
func parseFinding(raw json.RawMessage) (ReviewFinding, bool) {
	var probe struct {
		File     *string          `json:"file"`
		Line     *json.RawMessage `json:"line"`
		Severity *string          `json:"severity"`
		Comment  *string          `json:"comment"`
		// Confidence goes through coerceInt too: it is a REQUIRED field in the
		// response schema, so a model emitting it as 95.0 would otherwise drop
		// every finding in the batch and report a silent empty review.
		Confidence *json.RawMessage `json:"confidence"`
		Headline   *string          `json:"headline"`
		Suggestion *string          `json:"suggestion"`
	}
	if json.Unmarshal(raw, &probe) != nil {
		return ReviewFinding{}, false
	}
	if probe.File == nil || probe.Severity == nil || probe.Comment == nil || probe.Line == nil {
		return ReviewFinding{}, false
	}
	if !validSeverity(*probe.Severity) {
		return ReviewFinding{}, false
	}
	line, ok := coerceInt(*probe.Line)
	if !ok {
		return ReviewFinding{}, false
	}
	out := ReviewFinding{
		File:       *probe.File,
		Line:       line,
		Severity:   *probe.Severity,
		Comment:    *probe.Comment,
		Headline:   probe.Headline,
		Suggestion: probe.Suggestion,
	}
	// confidence is Optional, so an explicit null is absent rather than
	// invalid. Any other unusable value drops the finding, as Pydantic does.
	if probe.Confidence != nil && !isJSONNull(*probe.Confidence) {
		n, ok := coerceInt(*probe.Confidence)
		if !ok {
			return ReviewFinding{}, false
		}
		out.Confidence = &n
	}
	return out, true
}

// isJSONNull reports whether raw is the JSON literal null. Needed because Go
// unmarshals null into most types without error, leaving the zero value, where
// Python's type checks reject it.
func isJSONNull(raw json.RawMessage) bool {
	return string(bytes.TrimSpace(raw)) == "null"
}

func validSeverity(s string) bool {
	for _, allowed := range Severities {
		if s == allowed {
			return true
		}
	}
	return false
}

// coerceInt accepts an integer, a string holding one, or a float equal to its
// own truncation, matching Pydantic's lax coercion. Verified against the
// running Python: 95.0 becomes 95, "95" becomes 95, and 95.5 is refused
// because silently truncating it would move a finding to the wrong line.
func coerceInt(raw json.RawMessage) (int, bool) {
	// null unmarshals into an int without error, leaving 0, where Python
	// rejects it for a required field. Same trap as the map case.
	if isJSONNull(raw) {
		return 0, false
	}
	// A bool coerces in Pydantic lax mode: true becomes 1.
	var b bool
	if json.Unmarshal(raw, &b) == nil {
		if b {
			return 1, true
		}
		return 0, true
	}
	var n int
	if json.Unmarshal(raw, &n) == nil {
		return n, true
	}
	var f float64
	if json.Unmarshal(raw, &f) == nil {
		if trunc := math.Trunc(f); trunc == f {
			return int(trunc), true
		}
		return 0, false
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
			return n, true
		}
	}
	return 0, false
}

// stripFence removes a leading ``` line and, when the final line is exactly a
// fence, that too. Mirrors Python: the opening line is dropped unconditionally
// (including its info string), the closing one only when it is exactly "```"
// after trimming.
func stripFence(content string) string {
	if !strings.HasPrefix(content, "```") {
		return content
	}
	lines := splitLines(content)
	if len(lines) > 0 {
		lines = lines[1:]
	}
	if len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "```" {
		lines = lines[:len(lines)-1]
	}
	return strings.Join(lines, "\n")
}

// ParseReview parses a model review response.
func ParseReview(content string) ParsedReview {
	content = stripFence(strings.TrimSpace(content))

	var raw map[string]json.RawMessage
	// A JSON array or scalar fails to unmarshal into a map. `null` does NOT:
	// Go accepts it and leaves the map nil, where Python's
	// isinstance(data, dict) rejected it. The nil check is what covers that.
	if err := json.Unmarshal([]byte(content), &raw); err != nil || raw == nil {
		// The summary still carries the default verdict.
		return ParsedReview{
			Summary:     NewReviewSummary(),
			ParseFailed: true,
			Diagnostics: []ParseDiagnostic{parseErrorDiagnostic(err, content)},
		}
	}

	out := ParsedReview{
		ComplianceRequirements: []ComplianceRequirement{},
		Summary:                NewReviewSummary(),
	}

	if rawReqs, ok := raw["compliance_requirements"]; ok {
		var items []json.RawMessage
		if err := json.Unmarshal(rawReqs, &items); err == nil {
			for _, item := range items {
				// Python requires isinstance(requirement, str) and
				// isinstance(met, bool), which is a TYPE check: a present key
				// holding null fails it. Probing for presence alone is not
				// enough, because encoding/json decodes a null into a string
				// or bool field without error, turning {"met": null} into a
				// silent "not met" and {"requirement": null} into the "???"
				// placeholder the summary renders.
				var probe map[string]json.RawMessage
				if json.Unmarshal(item, &probe) != nil {
					out.warn("Skipping malformed compliance requirement: %s", item)
					continue
				}
				var reqStr string
				if !decodesAs(probe["requirement"], &reqStr) {
					out.warn("Skipping malformed compliance requirement: %s", item)
					continue
				}
				var metBool bool
				if !decodesAs(probe["met"], &metBool) {
					out.warn("Skipping malformed compliance requirement: %s", item)
					continue
				}
				req := ComplianceRequirement{Requirement: reqStr, Met: metBool}
				out.ComplianceRequirements = append(out.ComplianceRequirements, req)
			}
		}
	}

	out.Summary.Overview = strings.TrimSpace(decodeString(raw["overview"]))
	// A blank overview parses fine but means the model returned no summary at
	// all, which the operator should see.
	if out.Summary.Overview == "" {
		out.warn("overview empty after parse")
	}
	out.Summary.SecurityPerformance = strings.TrimSpace(decodeString(raw["security_performance"]))
	out.Summary.TestCoverage = strings.TrimSpace(decodeString(raw["test_coverage"]))

	// Non-string and blank entries are dropped; the rest keep their original
	// spacing, since Python filters on s.strip() but appends s.
	if rawStrengths, ok := raw["strengths"]; ok {
		var items []json.RawMessage
		if err := json.Unmarshal(rawStrengths, &items); err == nil {
			for _, item := range items {
				var s string
				if json.Unmarshal(item, &s) != nil {
					continue
				}
				if strings.TrimSpace(s) == "" {
					continue
				}
				out.Summary.Strengths = append(out.Summary.Strengths, s)
			}
		}
	}

	if rawVerdict, ok := raw["verdict"]; ok {
		var verdict map[string]json.RawMessage
		if err := json.Unmarshal(rawVerdict, &verdict); err == nil {
			// An unrecognised decision leaves the default in place rather than
			// clearing it, and the rationale is kept either way.
			decision := decodeString(verdict["decision"])
			for _, allowed := range VerdictDecisions {
				if decision == allowed {
					out.Summary.VerdictDecision = decision
					break
				}
			}
			out.Summary.VerdictRationale = strings.TrimSpace(decodeString(verdict["rationale"]))
		}
	}

	if rawFindings, ok := raw["findings"]; ok {
		var items []json.RawMessage
		if err := json.Unmarshal(rawFindings, &items); err == nil {
			for _, item := range items {
				f, ok := parseFinding(item)
				if !ok {
					out.warn("Skipping malformed finding: %s", item)
					continue
				}
				// A finding whose suggestion says there is nothing to do is
				// not a finding.
				if f.Suggestion != nil && IsVacuousSuggestion(*f.Suggestion) {
					out.info("Dropping no-issue finding (vacuous suggestion): %s", item)
					continue
				}
				out.Findings = append(out.Findings, f)
			}
		}
	}

	return out
}

// decodeString returns raw as a string, or "" when it is absent or not a
// string. Mirrors Python's isinstance(x, str) guards.
func decodeString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return ""
	}
	return s
}

// vacuousSuggestionPatterns drop findings whose suggestion says there is
// nothing to do.
//
// Python's \b and \s are Unicode, RE2's are ASCII, and this pattern set
// diverges in BOTH directions. Verified against the running Python:
//   - "no fix needed" (non-breaking spaces) matches in Python
//     because \s covers them; plain RE2 \s would not.
//   - "üno fix needed" does NOT match in Python, because ü is a word character
//     so \b fails; plain RE2 \b would match.
//
// So \s becomes [\s\p{Zs}] and \b is spelled out against the Unicode word set.
var vacuousSuggestionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)` + wordStart + `no` + uniSpace + `+fix(es)?` + uniSpace + `+(needed|required)` + wordEnd),
	regexp.MustCompile(`(?i)` + wordStart + `no` + uniSpace + `+changes?` + uniSpace + `+(needed|required)` + wordEnd),
	regexp.MustCompile(`(?i)` + wordStart + `nothing` + uniSpace + `+to` + uniSpace + `+(fix|change)` + wordEnd),
	regexp.MustCompile(`(?i)` + wordStart + `code` + uniSpace + `+is` + uniSpace + `+(actually` + uniSpace + `+)?correct` + wordEnd),
	regexp.MustCompile(`(?i)` + wordStart + `this` + uniSpace + `+is` + uniSpace + `+correct` + wordEnd),
	regexp.MustCompile(`(?i)^n/?a$`),
}

const (
	// uniSpace matches what Python's \s matches, including non-breaking and
	// other Unicode spaces that RE2's ASCII \s misses.
	uniSpace = `[\s\p{Zs}]`
	// wordStart and wordEnd stand in for Python's Unicode-aware \b.
	wordStart = `(?:^|[^\pL\pN_])`
	wordEnd   = `(?:$|[^\pL\pN_])`
)

// maxVacuousSuggestionLen bounds what is considered vacuous: a long suggestion
// containing one of these phrases is still a real suggestion.
const maxVacuousSuggestionLen = 120

// IsVacuousSuggestion reports whether a suggestion says there is nothing to do.
func IsVacuousSuggestion(suggestion string) bool {
	stripped := strings.TrimSpace(suggestion)
	if stripped == "" {
		return false
	}
	// Python measures len() in characters, not bytes.
	if len([]rune(stripped)) > maxVacuousSuggestionLen {
		return false
	}
	for _, p := range vacuousSuggestionPatterns {
		if p.MatchString(stripped) {
			return true
		}
	}
	return false
}

// ParseMention turns a mention.txt JSON envelope into a rendered markdown
// answer, falling back to the raw content when parsing fails. Models that
// disregard the envelope still produce a usable reply.
func ParseMention(content string) string {
	text := strings.TrimSpace(content)
	if text == "" {
		return ""
	}
	stripped := text
	if strings.HasPrefix(stripped, "```") {
		stripped = strings.TrimSpace(stripFence(stripped))
	}

	var raw map[string]json.RawMessage
	// As in ParseReview, `null` unmarshals into a nil map rather than failing.
	if err := json.Unmarshal([]byte(stripped), &raw); err != nil || raw == nil {
		return text
	}
	rawAnswer, ok := raw["answer"]
	if !ok {
		return text
	}
	var answer string
	if json.Unmarshal(rawAnswer, &answer) != nil {
		return text
	}
	answer = strings.TrimSpace(answer)

	var refLines []string
	if rawRefs, ok := raw["refs"]; ok {
		var items []json.RawMessage
		if json.Unmarshal(rawRefs, &items) == nil {
			for _, item := range items {
				var ref struct {
					File string           `json:"file"`
					Line *json.RawMessage `json:"line"`
				}
				if json.Unmarshal(item, &ref) != nil || ref.File == "" {
					continue
				}
				// A line is rendered only when it is an integer. Python's
				// isinstance(line, int) rejects a float or a string, and also
				// accepts a bool, which json.Unmarshal into int would not; a
				// bool line is vanishingly unlikely and renders without it.
				if ref.Line != nil {
					var line int
					if json.Unmarshal(*ref.Line, &line) == nil {
						refLines = append(refLines, "- `"+ref.File+"`:"+itoa(line))
						continue
					}
				}
				refLines = append(refLines, "- `"+ref.File+"`")
			}
		}
	}
	if len(refLines) > 0 {
		answer = answer + "\n\n**References:**\n" + strings.Join(refLines, "\n")
	}
	return answer
}
