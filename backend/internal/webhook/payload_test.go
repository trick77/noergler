package webhook

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
)

// A real Bitbucket Server delivery, captured verbatim.
func TestDecodeRealWebhookPayload(t *testing.T) {
	blob, err := os.ReadFile("testdata/sample_webhook.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	p, err := Decode(blob)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}

	if p.EventKey != EventOpened {
		t.Errorf("eventKey = %q, want %q", p.EventKey, EventOpened)
	}
	if p.PullRequest.ID != 42 {
		t.Errorf("pr id = %d, want 42", p.PullRequest.ID)
	}
	if p.PullRequest.Title != "Add new feature" {
		t.Errorf("title = %q", p.PullRequest.Title)
	}
	if p.PullRequest.Author.User.Name != "jan.username" {
		t.Errorf("author = %q", p.PullRequest.Author.User.Name)
	}
	if p.PullRequest.FromRef.LatestCommit != "abc123def456" {
		t.Errorf("latestCommit = %q", p.PullRequest.FromRef.LatestCommit)
	}
	if p.Actor == nil || p.Actor.Name != "jan.username" {
		t.Errorf("actor = %+v", p.Actor)
	}
	// Absent in the fixture: optional, so it decodes to a zero value.
	if p.PullRequest.State != "" {
		t.Errorf("state = %q, want empty", p.PullRequest.State)
	}
	if p.PullRequest.CreatedDate != 0 {
		t.Errorf("createdDate = %d, want 0", p.PullRequest.CreatedDate)
	}

	project, repo := p.ProjectRepo()
	if project != "PROJ" || repo != "my-repo" {
		t.Errorf("ProjectRepo() = (%q, %q), want (PROJ, my-repo)", project, repo)
	}
}

// A payload missing a required field is refused at the edge. Without
// Validate the struct would accept these and fail deep in the review path.
func TestValidateRejectsMissingRequiredFields(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"no eventKey", `{"pullRequest":{"id":1,"title":"t","fromRef":{"id":"a","displayId":"a"},"toRef":{"id":"b","displayId":"b"},"author":{"user":{"name":"u"}}}}`},
		{"no pr id", `{"eventKey":"pr:opened","pullRequest":{"title":"t","fromRef":{"id":"a","displayId":"a"},"toRef":{"id":"b","displayId":"b"},"author":{"user":{"name":"u"}}}}`},
		{"no title", `{"eventKey":"pr:opened","pullRequest":{"id":1,"fromRef":{"id":"a","displayId":"a"},"toRef":{"id":"b","displayId":"b"},"author":{"user":{"name":"u"}}}}`},
		{"no author name", `{"eventKey":"pr:opened","pullRequest":{"id":1,"title":"t","fromRef":{"id":"a","displayId":"a"},"toRef":{"id":"b","displayId":"b"},"author":{"user":{}}}}`},
		{"no fromRef", `{"eventKey":"pr:opened","pullRequest":{"id":1,"title":"t","toRef":{"id":"b","displayId":"b"},"author":{"user":{"name":"u"}}}}`},
		{"no toRef", `{"eventKey":"pr:opened","pullRequest":{"id":1,"title":"t","fromRef":{"id":"a","displayId":"a"},"author":{"user":{"name":"u"}}}}`},
		{"comment without author", `{"eventKey":"pr:comment:added","pullRequest":{"id":1,"title":"t","fromRef":{"id":"a","displayId":"a"},"toRef":{"id":"b","displayId":"b"},"author":{"user":{"name":"u"}}},"comment":{"id":7,"text":"hi","author":{}}}`},
		{"comment without id", `{"eventKey":"pr:comment:added","pullRequest":{"id":1,"title":"t","fromRef":{"id":"a","displayId":"a"},"toRef":{"id":"b","displayId":"b"},"author":{"user":{"name":"u"}}},"comment":{"text":"hi","author":{"name":"u"}}}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Decode([]byte(c.json)); !errors.Is(err, ErrInvalidPayload) {
				t.Errorf("Decode accepted an invalid payload, err = %v", err)
			}
		})
	}
}

// Bitbucket adds fields across versions, so an unknown key must not refuse a
// delivery the service can handle. This is the opposite of teams.yaml, where
// an unknown key is an operator typo.
func TestUnknownFieldsAreAllowed(t *testing.T) {
	js := `{"eventKey":"pr:opened","somethingNew":{"a":1},"pullRequest":{"id":1,"title":"t","brandNew":true,
		"fromRef":{"id":"a","displayId":"a"},"toRef":{"id":"b","displayId":"b"},"author":{"user":{"name":"u"}}}}`
	if _, err := Decode([]byte(js)); err != nil {
		t.Errorf("unknown fields rejected: %v", err)
	}
}

// Either side may carry the repository; toRef wins when both do.
func TestProjectRepoFallsBackToFromRef(t *testing.T) {
	p := &Payload{}
	p.PullRequest.FromRef.Repository = &Repository{Slug: "from-repo", Project: Project{Key: "FROM"}}
	project, repo := p.ProjectRepo()
	if project != "FROM" || repo != "from-repo" {
		t.Errorf("fallback to fromRef failed: (%q, %q)", project, repo)
	}

	p.PullRequest.ToRef.Repository = &Repository{Slug: "to-repo", Project: Project{Key: "TO"}}
	project, repo = p.ProjectRepo()
	if project != "TO" || repo != "to-repo" {
		t.Errorf("toRef should win: (%q, %q)", project, repo)
	}

	empty := &Payload{}
	if project, repo := empty.ProjectRepo(); project != "" || repo != "" {
		t.Errorf("no repository anywhere should give empty strings, got (%q, %q)", project, repo)
	}
}

// An empty string is treated as missing: some deployments populate the field
// as "" rather than omitting it.
func TestMergeCommitSHA(t *testing.T) {
	p := &Payload{}
	if got := p.MergeCommitSHA(); got != "" {
		t.Errorf("no properties = %q, want empty", got)
	}

	p.PullRequest.Properties = &Properties{}
	if got := p.MergeCommitSHA(); got != "" {
		t.Errorf("no mergeCommit = %q, want empty", got)
	}

	p.PullRequest.Properties.MergeCommit = &MergeCommit{ID: ""}
	if got := p.MergeCommitSHA(); got != "" {
		t.Errorf("empty id = %q, want empty", got)
	}

	p.PullRequest.Properties.MergeCommit = &MergeCommit{ID: "deadbeef"}
	if got := p.MergeCommitSHA(); got != "deadbeef" {
		t.Errorf("= %q, want deadbeef", got)
	}
}

// json.Unmarshal accepts an explicit null into a pointer or map and leaves it
// nil, so every reader of these fields must be nil-safe.
func TestNullsDecodeSafely(t *testing.T) {
	js := `{"eventKey":"pr:merged","actor":null,"comment":null,
		"pullRequest":{"id":1,"title":"t","state":null,"properties":null,
		"fromRef":{"id":"a","displayId":"a","repository":null},
		"toRef":{"id":"b","displayId":"b","repository":null},
		"author":{"user":{"name":"u"}}}}`
	p, err := Decode([]byte(js))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if p.Actor != nil || p.Comment != nil || p.PullRequest.Properties != nil {
		t.Error("explicit nulls should decode as nil")
	}
	if project, repo := p.ProjectRepo(); project != "" || repo != "" {
		t.Errorf("null repositories = (%q, %q), want empty", project, repo)
	}
	if got := p.MergeCommitSHA(); got != "" {
		t.Errorf("null properties = %q, want empty", got)
	}
}

// An empty comment.text validates, so the route answers 200 "comment without
// mention"; rejecting it here would answer 400 instead.
func TestEmptyCommentTextIsAccepted(t *testing.T) {
	js := `{"eventKey":"pr:comment:added",
		"pullRequest":{"id":1,"title":"t",
		"fromRef":{"id":"a","displayId":"a"},
		"toRef":{"id":"b","displayId":"b"},
		"author":{"user":{"name":"u"}}},
		"comment":{"id":5,"text":"","author":{"name":"x"}}}`
	p, err := Decode([]byte(js))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if p.Comment == nil || p.Comment.Text != "" {
		t.Errorf("comment = %+v, want one with empty text", p.Comment)
	}
}

// Decode has two failure paths and both must carry ErrInvalidPayload: the
// json.Unmarshal one (malformed or wrongly-typed JSON) and the Validate one
// (well-formed JSON, missing required fields).
//
// TestValidateRejectsMissingRequiredFields only covers the second: every case
// there is structurally valid and fails in Validate. So when the Unmarshal
// branch stopped wrapping the sentinel, nothing noticed. These cases fail in
// Unmarshal, which is the gap.
func TestDecodeWrapsSentinelOnMalformedJSON(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"not json at all", `not json`},
		{"truncated", `{"eventKey":"pr:opened"`},
		// Passes the eventKey peek in api/webhook.go, then fails on the type.
		{"wrong type for pr id", `{"eventKey":"pr:opened","pullRequest":{"id":"abc"}}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Decode([]byte(c.json))
			if err == nil {
				t.Fatal("Decode accepted malformed JSON")
			}
			if !errors.Is(err, ErrInvalidPayload) {
				t.Errorf("errors.Is(err, ErrInvalidPayload) = false, err = %v", err)
			}
			// The underlying json error must survive too, or the reason is lost.
			var typeErr *json.UnmarshalTypeError
			var syntaxErr *json.SyntaxError
			if !errors.As(err, &typeErr) && !errors.As(err, &syntaxErr) {
				t.Errorf("the json error was dropped, err = %v", err)
			}
		})
	}
}
