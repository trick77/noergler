package inference

import (
	"encoding/json"

	"github.com/trick77/llmwire"
)

// ReviewSchemaName is the strict JSON schema's name on the wire.
const ReviewSchemaName = "review_response"

// ReviewResponseSchema is the strict JSON schema bound to every review call.
//
// It is also token-counted as part of the pre-flight fit check, because the
// gateway bills it as input like everything else. The counted form uses ", "
// and ": " separators, which brings the serialization to 1492 bytes. Go's
// json.Marshal is compact and would count fewer tokens for the same schema,
// moving every pinned total in testdata/assemble_golden.json, so SchemaJSON
// re-inserts the separators rather than marshalling.
func ReviewResponseSchema() map[string]any {
	decisions := make([]any, len(VerdictDecisions))
	for i, d := range VerdictDecisions {
		decisions[i] = d
	}
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required": []any{
			"overview", "strengths", "security_performance", "test_coverage",
			"verdict", "findings", "compliance_requirements",
		},
		"properties": map[string]any{
			"overview":             map[string]any{"type": "string", "minLength": 1},
			"strengths":            map[string]any{"type": "array", "maxItems": 4, "items": map[string]any{"type": "string"}},
			"security_performance": map[string]any{"type": "string", "minLength": 1},
			"test_coverage":        map[string]any{"type": "string", "minLength": 1},
			"verdict": map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"decision", "rationale"},
				"properties": map[string]any{
					"decision":  map[string]any{"type": "string", "enum": decisions},
					"rationale": map[string]any{"type": "string", "minLength": 1},
				},
			},
			"findings": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"required":             []any{"file", "line", "severity", "confidence", "headline", "comment", "suggestion"},
					"properties": map[string]any{
						"file":       map[string]any{"type": "string"},
						"line":       map[string]any{"type": "integer"},
						"severity":   map[string]any{"type": "string", "enum": []any{"issue", "suggestion"}},
						"confidence": map[string]any{"type": "integer", "minimum": 80, "maximum": 100},
						"headline":   map[string]any{"type": "string", "minLength": 1},
						"comment":    map[string]any{"type": "string"},
						"suggestion": map[string]any{"type": []any{"string", "null"}},
					},
				},
			},
			"compliance_requirements": map[string]any{
				"type": []any{"array", "null"},
				"items": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"required":             []any{"requirement", "met", "evidence"},
					"properties": map[string]any{
						"requirement": map[string]any{"type": "string"},
						"met":         map[string]any{"type": "boolean"},
						"evidence":    map[string]any{"type": []any{"string", "null"}},
					},
				},
			},
		},
	}
}

// ReviewResponseFormat is the schema as llmwire sends it.
func ReviewResponseFormat() *llmwire.ResponseFormat {
	return &llmwire.ResponseFormat{
		Kind:   llmwire.FormatJSONSchema,
		Name:   ReviewSchemaName,
		Schema: ReviewResponseSchema(),
		Strict: true,
	}
}

// schemaJSON is the exact byte form the schema is counted in: the key order
// written below, with ", " and ": " separators.
//
// It is a literal rather than a marshal of ReviewResponseSchema because Go's
// encoder sorts map keys, and the tokenizer is sensitive to that: the same
// 1492 bytes tokenize to 432 in sorted order against 435 in this order. The
// fit check compares against the model's real ceiling, so a three-token
// under-count is a (small) permissive drift; pinning the string keeps the
// counted form stable. TestSchemaStringAndMapAgree asserts this stays in
// sync with ReviewResponseSchema.
const schemaJSON = `{"type": "object", "additionalProperties": false, "required": ` +
	`["overview", "strengths", "security_performance", "test_coverage", "verdict", "findings", ` +
	`"compliance_requirements"], "properties": {"overview": {"type": "string", "minLength": 1}, ` +
	`"strengths": {"type": "array", "maxItems": 4, "items": {"type": "string"}}, ` +
	`"security_performance": {"type": "string", "minLength": 1}, ` +
	`"test_coverage": {"type": "string", "minLength": 1}, ` +
	`"verdict": {"type": "object", "additionalProperties": false, "required": ["decision", "rationale"], ` +
	`"properties": {"decision": {"type": "string", "enum": ["approve", "approve_with_followups", ` +
	`"request_changes"]}, "rationale": {"type": "string", "minLength": 1}}}, ` +
	`"findings": {"type": "array", "items": {"type": "object", "additionalProperties": false, ` +
	`"required": ["file", "line", "severity", "confidence", "headline", "comment", "suggestion"], ` +
	`"properties": {"file": {"type": "string"}, "line": {"type": "integer"}, ` +
	`"severity": {"type": "string", "enum": ["issue", "suggestion"]}, ` +
	`"confidence": {"type": "integer", "minimum": 80, "maximum": 100}, ` +
	`"headline": {"type": "string", "minLength": 1}, "comment": {"type": "string"}, ` +
	`"suggestion": {"type": ["string", "null"]}}}}, ` +
	`"compliance_requirements": {"type": ["array", "null"], "items": {"type": "object", ` +
	`"additionalProperties": false, "required": ["requirement", "met", "evidence"], ` +
	`"properties": {"requirement": {"type": "string"}, "met": {"type": "boolean"}, ` +
	`"evidence": {"type": ["string", "null"]}}}}}}`

// SchemaJSON is the schema in its pinned byte form, for token counting.
func SchemaJSON() string { return schemaJSON }

// schemaEquivalent reports whether the pinned string and the map describe the
// same schema, ignoring key order. Used by the test that keeps them in sync.
func schemaEquivalent() (bool, error) {
	var fromString any
	if err := json.Unmarshal([]byte(schemaJSON), &fromString); err != nil {
		return false, err
	}
	viaMap, err := json.Marshal(ReviewResponseSchema())
	if err != nil {
		return false, err
	}
	var fromMap any
	if err := json.Unmarshal(viaMap, &fromMap); err != nil {
		return false, err
	}
	a, err := json.Marshal(fromString)
	if err != nil {
		return false, err
	}
	b, err := json.Marshal(fromMap)
	if err != nil {
		return false, err
	}
	return string(a) == string(b), nil
}
