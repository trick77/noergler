package inference

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/tokens"
)

type assembleGolden struct {
	PromptSHA256 string `json:"prompt_sha256"`
	Breakdown    struct {
		Template         int `json:"template"`
		RepoInstructions int `json:"repo_instructions"`
		Files            int `json:"files"`
	} `json:"breakdown"`
	PromptTokens int `json:"prompt_tokens"`
	PromptLen    int `json:"prompt_len"`
}

type schemaGolden struct {
	JSONLen      int `json:"json_len"`
	Tokens       int `json:"tokens"`
	SystemTokens int `json:"system_tokens"`
}

func loadAssembleGolden(t *testing.T) (map[string]assembleGolden, schemaGolden) {
	t.Helper()
	blob, err := os.ReadFile("testdata/assemble_golden.json")
	if err != nil {
		t.Fatalf("read assemble golden: %v", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(blob, &raw); err != nil {
		t.Fatalf("decode assemble golden: %v", err)
	}
	var schema schemaGolden
	if err := json.Unmarshal(raw["_schema"], &schema); err != nil {
		t.Fatalf("decode schema golden: %v", err)
	}
	delete(raw, "_schema")

	cases := make(map[string]assembleGolden, len(raw))
	for name, r := range raw {
		var g assembleGolden
		if err := json.Unmarshal(r, &g); err != nil {
			t.Fatalf("decode case %q: %v", name, err)
		}
		cases[name] = g
	}
	return cases, schema
}

func testTemplate(t *testing.T) string {
	t.Helper()
	// prompts/ lives at the REPO root, beside backend/, not inside the module:
	// the image mounts it and a deployment may swap it. Hence the third level.
	blob, err := os.ReadFile("../../../prompts/review.txt")
	if err != nil {
		t.Fatalf("read template: %v", err)
	}
	return string(blob)
}

func counter(t *testing.T) CountFunc {
	t.Helper()
	c, err := tokens.New()
	if err != nil {
		t.Fatalf("tokens.New: %v", err)
	}
	return c.Count
}

func assembleCases(template string) map[string]AssembleRequest {
	content := "package a\n"
	files1 := []diff.FileReviewData{
		{Path: "a.go", Diff: "@@ -1 +1 @@\n-x\n+y", Content: content, ContentFetched: true},
	}
	files2 := []diff.FileReviewData{
		{Path: "a.go", Diff: "@@ -1 +1 @@\n-x\n+y", Content: content, ContentFetched: true},
		{Path: "b/c.py", Diff: "@@ -2 +2 @@\n-p\n+q"},
	}
	base := func(files []diff.FileReviewData) AssembleRequest {
		return AssembleRequest{Template: template, Files: files, TicketComplianceCheck: true}
	}

	cases := map[string]AssembleRequest{}

	cases["minimal"] = base(files1)

	withRepo := base(files1)
	withRepo.RepoInstructions = "# Rules\nBe terse.\n"
	cases["with_repo_instructions"] = withRepo

	withTicket := base(files1)
	withTicket.TicketContext = "### Jira ticket: [ABC-1](u)\n**Title:** T"
	cases["with_ticket"] = withTicket

	complianceOff := base(files1)
	complianceOff.TicketContext = "### Jira ticket: [ABC-1](u)"
	complianceOff.TicketComplianceCheck = false
	cases["ticket_compliance_off"] = complianceOff

	cases["ticket_absent_compliance_on"] = base(files1)

	supp := base(files2)
	supp.OtherModifiedPaths = []string{"x/y.go"}
	supp.DeletedFilePaths = []string{"gone.go"}
	supp.RenamedFilePaths = []string{"old.go -> new.go"}
	cases["with_supplementary"] = supp

	cross := base(files2)
	cross.CrossFileContext = "## Cross-file\n- Foo used in a.go"
	cases["with_cross_file"] = cross

	cum := base(files1)
	cum.CumulativePRDiff = "diff --git a/z b/z\n@@ -1 +1 @@\n-1\n+2"
	cases["with_cumulative"] = cum

	every := base(files2)
	every.RepoInstructions = "# Rules\nBe terse.\n"
	every.OtherModifiedPaths = []string{"x/y.go"}
	every.DeletedFilePaths = []string{"gone.go"}
	every.RenamedFilePaths = []string{"old.go -> new.go"}
	every.TicketContext = "### Jira ticket: [ABC-1](u)\n**Title:** T"
	every.CrossFileContext = "## Cross-file\n- Foo used in a.go"
	every.CumulativePRDiff = "diff --git a/z b/z\n@@ -1 +1 @@\n-1\n+2"
	line := 3
	every.PreviouslyPosted = []PostedFinding{
		{FilePath: "a.go", LineNumber: &line, Severity: "issue", CommentText: "old finding"},
	}
	cases["everything"] = every

	return cases
}

// TestAssembleReviewPromptIsPinned pins the whole assembled prompt by
// SHA-256 against the Python review_diff assembly, plus the breakdown counts
// and the fit-check token total.
func TestAssembleReviewPromptIsPinned(t *testing.T) {
	golden, _ := loadAssembleGolden(t)
	template := testTemplate(t)
	count := counter(t)
	cases := assembleCases(template)

	for name, want := range golden {
		req, ok := cases[name]
		if !ok {
			t.Errorf("golden case %q has no Go input", name)
			continue
		}
		t.Run(name, func(t *testing.T) {
			got := AssembleReviewPrompt(req, count)

			sum := sha256.Sum256([]byte(got.Prompt))
			if hex.EncodeToString(sum[:]) != want.PromptSHA256 {
				t.Errorf("prompt differs from Python (len %d, want %d)", len(got.Prompt), want.PromptLen)
			}
			if got.Breakdown.Template != want.Breakdown.Template {
				t.Errorf("breakdown.template = %d, want %d", got.Breakdown.Template, want.Breakdown.Template)
			}
			if got.Breakdown.RepoInstructions != want.Breakdown.RepoInstructions {
				t.Errorf("breakdown.repo_instructions = %d, want %d", got.Breakdown.RepoInstructions, want.Breakdown.RepoInstructions)
			}
			if got.Breakdown.Files != want.Breakdown.Files {
				t.Errorf("breakdown.files = %d, want %d", got.Breakdown.Files, want.Breakdown.Files)
			}
			if got.PromptTokens != want.PromptTokens {
				t.Errorf("PromptTokens = %d, want %d", got.PromptTokens, want.PromptTokens)
			}
		})
	}
	for name := range cases {
		if _, ok := golden[name]; !ok {
			t.Errorf("Go case %q has no golden value", name)
		}
	}
}

// The fit check weighs system + prompt + schema, because the gateway bills
// all three as input. Counting the prompt alone would under-count by roughly
// the schema and let a PR through that does not fit.
func TestSchemaSerializationIsPinned(t *testing.T) {
	_, schema := loadAssembleGolden(t)
	count := counter(t)

	if got := len(SchemaJSON()); got != schema.JSONLen {
		t.Errorf("SchemaJSON length = %d, want %d: Go's compact encoder must be spaced like json.dumps", got, schema.JSONLen)
	}
	if got := count(SchemaJSON()); got != schema.Tokens {
		t.Errorf("schema tokens = %d, want %d", got, schema.Tokens)
	}
	if got := count(ReviewSystemMessage); got != schema.SystemTokens {
		t.Errorf("system message tokens = %d, want %d", got, schema.SystemTokens)
	}
}

// SchemaJSON is a pinned string because key order changes the token count,
// so it can drift from the map that actually goes on the wire. This is the
// guard: the two must describe the same schema.
func TestSchemaStringAndMapAgree(t *testing.T) {
	ok, err := schemaEquivalent()
	if err != nil {
		t.Fatalf("comparing schema forms: %v", err)
	}
	if !ok {
		t.Error("SchemaJSON and ReviewResponseSchema describe different schemas")
	}
}

// The schema on the wire must carry the real fields, not the test stub that
// stood in for it before.
func TestReviewResponseFormatIsTheRealSchema(t *testing.T) {
	rf := ReviewResponseFormat()
	if rf.Name != ReviewSchemaName || !rf.Strict {
		t.Errorf("name = %q, strict = %v", rf.Name, rf.Strict)
	}
	props, ok := rf.Schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("schema has no properties object")
	}
	for _, key := range []string{"overview", "strengths", "security_performance", "test_coverage",
		"verdict", "findings", "compliance_requirements"} {
		if _, ok := props[key]; !ok {
			t.Errorf("schema is missing the %q property", key)
		}
	}
}

// One substitution pass, so a block that itself contains a placeholder is
// never expanded. Python substitutes {files} last, which protects file
// content but leaves repo_instructions and ticket_context expandable; the
// single pass protects every block.
func TestPlaceholderInABlockStaysLiteral(t *testing.T) {
	count := counter(t)
	req := AssembleRequest{
		Template:              "FILES:{files}\nRULES:{repo_instructions}\nTICKET:{ticket_context}",
		Files:                 []diff.FileReviewData{{Path: "a.go", Diff: "@@"}},
		RepoInstructions:      "do not expand {files} here",
		TicketContext:         "nor {previously_posted_findings} here",
		TicketComplianceCheck: true,
	}
	got := AssembleReviewPrompt(req, count).Prompt

	if !strings.Contains(got, "do not expand {files} here") {
		t.Errorf("a {files} inside repo_instructions was expanded:\n%s", got)
	}
	if !strings.Contains(got, "nor {previously_posted_findings} here") {
		t.Errorf("a placeholder inside ticket_context was expanded:\n%s", got)
	}
}

// Both halves of the compliance rule: instructions only when a ticket is
// present AND the check is on, and the ticket default when no ticket linked.
func TestComplianceAndTicketDefaults(t *testing.T) {
	count := counter(t)
	tmpl := "T:{ticket_context}|C:{compliance_instructions}|F:{files}"
	base := AssembleRequest{Template: tmpl, TicketComplianceCheck: true}

	got := AssembleReviewPrompt(base, count).Prompt
	if !strings.Contains(got, "T:"+NoTicketContext) {
		t.Errorf("missing the no-ticket default: %q", got)
	}
	if !strings.Contains(got, "|C:|") {
		t.Errorf("compliance instructions present without a ticket: %q", got)
	}

	withTicket := base
	withTicket.TicketContext = "ABC-1"
	if got := AssembleReviewPrompt(withTicket, count).Prompt; !strings.Contains(got, ComplianceInstructions) {
		t.Error("compliance instructions missing when a ticket is present and the check is on")
	}

	off := withTicket
	off.TicketComplianceCheck = false
	if got := AssembleReviewPrompt(off, count).Prompt; strings.Contains(got, ComplianceInstructions) {
		t.Error("compliance instructions present when the check is off")
	}
}
