package webhook

import (
	"errors"
	"os"
	"testing"
)

// The fixture is the one the Python suite replays
// (tests/fixtures/sample_webhook.json), copied verbatim.
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
	// Absent in the fixture; Optional in Pydantic, zero here.
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

// Pydantic refuses a payload missing a required field before any handler
// sees it. A plain Go struct would accept these and fail deep in the review
// path instead.
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

// json.Unmarshal accepts null into a pointer or map and leaves it nil where
// Python's isinstance(x, dict) rejected it, so the nil cases must be safe.
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

// An empty comment.text validates: Pydantic's required str is satisfied by
// "", so Python answers 200 "comment without mention" where a rejection here
// would answer 400. Probed against the venv on 2026-09-20.
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
