package review

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"sync"

	"github.com/trick77/noergler-go/internal/bitbucket"
	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/inference"
	"github.com/trick77/noergler-go/internal/jira"
	"github.com/trick77/noergler-go/internal/riptide"
	"github.com/trick77/noergler-go/internal/store"
)

func quietLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// postedComment is one comment the fake Bitbucket recorded.
type postedComment struct {
	PRID int
	Text string
	File string
	Line int
}

// fakeBitbucket records what was posted and serves canned reads. Every field
// is a hook the test sets when it cares and leaves nil when it does not.
type fakeBitbucket struct {
	mu sync.Mutex

	bot string

	prDiff     string
	prDiffErr  error
	commitDiff string
	commitErr  error

	files    map[string]string // "commit:path" -> content
	fileErr  map[string]error
	comments map[int]*bitbucket.Comment

	nextCommentID int
	updateErr     error
	postErr       error
	inlineErr     error

	Inline   []postedComment
	Posted   []postedComment
	Updates  []postedComment
	Replies  []postedComment
	FetchedC []int
}

func newFakeBitbucket() *fakeBitbucket {
	return &fakeBitbucket{
		bot:           "noergler",
		files:         map[string]string{},
		fileErr:       map[string]error{},
		comments:      map[int]*bitbucket.Comment{},
		nextCommentID: 100,
	}
}

func (f *fakeBitbucket) BotUsername() string { return f.bot }

func (f *fakeBitbucket) FetchPRDiff(_ context.Context, _, _ string, _, _ int) (string, error) {
	return f.prDiff, f.prDiffErr
}

func (f *fakeBitbucket) FetchCommitDiff(_ context.Context, _, _, _, _ string) (string, error) {
	return f.commitDiff, f.commitErr
}

func (f *fakeBitbucket) FetchFileContent(_ context.Context, _, _, commit, path string) (string, error) {
	key := commit + ":" + path
	if err, ok := f.fileErr[key]; ok {
		return "", err
	}
	if content, ok := f.files[key]; ok {
		return content, nil
	}
	return "", fmt.Errorf("no such file %q", key)
}

func (f *fakeBitbucket) PostInlineComment(_ context.Context, _, _ string, prID int, file string, line int, body string) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.inlineErr != nil {
		return 0, f.inlineErr
	}
	f.nextCommentID++
	f.Inline = append(f.Inline, postedComment{PRID: prID, Text: body, File: file, Line: line})
	return f.nextCommentID, nil
}

func (f *fakeBitbucket) PostPRComment(_ context.Context, _, _ string, prID int, text string) (int, int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.postErr != nil {
		return 0, 0, f.postErr
	}
	f.nextCommentID++
	f.Posted = append(f.Posted, postedComment{PRID: prID, Text: text})
	f.comments[f.nextCommentID] = &bitbucket.Comment{ID: f.nextCommentID, Text: text, Version: 1}
	return f.nextCommentID, 1, nil
}

func (f *fakeBitbucket) ReplyToComment(_ context.Context, _, _ string, prID, parent int, text string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Replies = append(f.Replies, postedComment{PRID: prID, Text: text, Line: parent})
	return nil
}

func (f *fakeBitbucket) FetchPRComment(_ context.Context, _, _ string, _, commentID int) (*bitbucket.Comment, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.FetchedC = append(f.FetchedC, commentID)
	if c, ok := f.comments[commentID]; ok {
		return c, nil
	}
	return nil, &bitbucket.StatusError{Method: "GET", Path: "/comments", Status: 404}
}

func (f *fakeBitbucket) UpdatePRComment(_ context.Context, _, _ string, prID, commentID, _ int, text string) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.updateErr != nil {
		return 0, f.updateErr
	}
	f.Updates = append(f.Updates, postedComment{PRID: prID, Text: text, Line: commentID})
	if c, ok := f.comments[commentID]; ok {
		c.Text = text
		c.Version++
		return c.Version, nil
	}
	f.comments[commentID] = &bitbucket.Comment{ID: commentID, Text: text, Version: 2}
	return 2, nil
}

// setComment seeds an existing comment body, as a prior summary would be.
func (f *fakeBitbucket) setComment(id int, text string, version int) {
	f.comments[id] = &bitbucket.Comment{ID: id, Text: text, Version: version}
}

// fakeStore records writes and serves canned reads.
type fakeStore struct {
	mu sync.Mutex

	prID       int64
	skipState  *store.SkipState
	lastCommit string
	hasLast    bool
	summary    *store.SummaryComment
	existing   []store.Finding
	prCost     *int64
	rollup     *store.RollupSnapshot

	upsertErr error
	failAll   bool

	Upserts   []store.PRUpsert
	Runs      []store.Run
	Findings  []store.Finding
	Ignored   int
	Reactived int
	Merged    int
	Declined  int
	Deleted   int
	Summaries []store.SummaryComment
	Claims    []store.RollupFinal
	Frozen    int
}

func newFakeStore() *fakeStore {
	return &fakeStore{prID: 7}
}

func (f *fakeStore) err() error {
	if f.failAll {
		return fmt.Errorf("db is down")
	}
	return nil
}

func (f *fakeStore) UpsertPullRequest(_ context.Context, u store.PRUpsert) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.upsertErr != nil {
		return 0, f.upsertErr
	}
	if err := f.err(); err != nil {
		return 0, err
	}
	f.Upserts = append(f.Upserts, u)
	return f.prID, nil
}

func (f *fakeStore) GetLastReviewedCommit(context.Context, store.PRKey) (string, bool, error) {
	if err := f.err(); err != nil {
		return "", false, err
	}
	return f.lastCommit, f.hasLast, nil
}

func (f *fakeStore) SetSummaryComment(_ context.Context, _ int64, commentID, version int) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return err
	}
	f.Summaries = append(f.Summaries, store.SummaryComment{ID: commentID, Version: version})
	f.summary = &store.SummaryComment{ID: commentID, Version: version}
	return nil
}

func (f *fakeStore) GetSummaryComment(context.Context, int64) (*store.SummaryComment, error) {
	if err := f.err(); err != nil {
		return nil, err
	}
	return f.summary, nil
}

func (f *fakeStore) GetSkipState(context.Context, store.PRKey) (*store.SkipState, error) {
	if err := f.err(); err != nil {
		return nil, err
	}
	return f.skipState, nil
}

func (f *fakeStore) MarkIgnored(context.Context, store.PRKey) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Ignored++
	return f.err()
}

func (f *fakeStore) Reactivate(context.Context, store.PRKey) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Reactived++
	return f.err()
}

func (f *fakeStore) MarkMerged(context.Context, store.PRKey) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Merged++
	return f.err()
}

func (f *fakeStore) MarkDeclined(context.Context, store.PRKey) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Declined++
	return f.err()
}

func (f *fakeStore) MarkDeleted(context.Context, store.PRKey) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Deleted++
	return f.err()
}

func (f *fakeStore) InsertRun(_ context.Context, r store.Run) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return 0, err
	}
	f.Runs = append(f.Runs, r)
	return int64(len(f.Runs)), nil
}

func (f *fakeStore) InsertFinding(_ context.Context, fi store.Finding) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return err
	}
	f.Findings = append(f.Findings, fi)
	return nil
}

func (f *fakeStore) ExistingFindings(context.Context, store.PRKey) ([]store.Finding, error) {
	if err := f.err(); err != nil {
		return nil, err
	}
	return f.existing, nil
}

// PRCost is the real store's aggregate, not a static value: it sums the runs
// recorded so far on top of whatever the test seeded. Returning a fixed
// number would hide the ordering the pipeline depends on, where the run row
// must be written before the PR total is read.
func (f *fakeStore) PRCost(context.Context, store.PRKey) (*int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return nil, err
	}
	var total int64
	priced := false
	if f.prCost != nil {
		total, priced = *f.prCost, true
	}
	for _, r := range f.Runs {
		if r.CostNanoUSD != nil {
			total += *r.CostNanoUSD
			priced = true
		}
	}
	if !priced {
		return nil, nil
	}
	return &total, nil
}

func (f *fakeStore) FreezeFinalCost(ctx context.Context, k store.PRKey) (*int64, error) {
	f.mu.Lock()
	f.Frozen++
	f.mu.Unlock()
	return f.PRCost(ctx, k)
}

func (f *fakeStore) ClaimRollup(_ context.Context, _ store.PRKey, final store.RollupFinal) (*store.RollupSnapshot, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Claims = append(f.Claims, final)
	if err := f.err(); err != nil {
		return nil, err
	}
	return f.rollup, nil
}

// fakeLLM returns canned review and mention results.
type fakeLLM struct {
	model   string
	effort  string
	ready   bool
	window  int
	budget  int
	review  inference.ReviewResult
	mention inference.MentionResult

	Reviews  []inference.ReviewRequest
	Mentions []inference.MentionRequest
}

func newFakeLLM() *fakeLLM {
	return &fakeLLM{
		model:  "gpt-5.5",
		effort: "high",
		ready:  true,
		window: 1_000_000,
		budget: 628_000,
		review: inference.ReviewResult{
			Outcome: inference.OutcomeOK,
			Review:  inference.ParsedReview{Summary: inference.NewReviewSummary()},
		},
	}
}

func (f *fakeLLM) Model() string { return f.model }

// Label mirrors the real client rather than returning a canned string: a fake
// that formats the label its own way would keep passing if the real one
// changed.
func (f *fakeLLM) Label() string         { return config.ModelLabel(f.model, f.effort) }
func (f *fakeLLM) Ready() bool           { return f.ready }
func (f *fakeLLM) ContextWindow() int    { return f.window }
func (f *fakeLLM) InputTokenBudget() int { return f.budget }

func (f *fakeLLM) Review(_ context.Context, req inference.ReviewRequest) inference.ReviewResult {
	f.Reviews = append(f.Reviews, req)
	return f.review
}

func (f *fakeLLM) Mention(_ context.Context, req inference.MentionRequest) inference.MentionResult {
	f.Mentions = append(f.Mentions, req)
	return f.mention
}

// fakeJira serves one ticket and optionally its parent.
type fakeJira struct {
	ticket *jira.Ticket
	parent *jira.Ticket
	err    error
	Calls  []string
}

func (f *fakeJira) FetchTicketWithParent(_ context.Context, key string) (*jira.Ticket, *jira.Ticket, error) {
	f.Calls = append(f.Calls, key)
	return f.ticket, f.parent, f.err
}

// fakeRiptide records emitted rollups.
type fakeRiptide struct {
	enabled bool
	Emitted []riptide.Rollup
}

func (f *fakeRiptide) Enabled() bool { return f.enabled }
func (f *fakeRiptide) EmitPRCompleted(_ context.Context, r riptide.Rollup) {
	f.Emitted = append(f.Emitted, r)
}

// fakeTokens counts one token per four bytes, which is close enough to real
// text for budget arithmetic and is exact enough to assert on.
type fakeTokens struct{ perToken int }

func (f *fakeTokens) Count(text string) int {
	n := f.perToken
	if n <= 0 {
		n = 4
	}
	return (len(text) + n - 1) / n
}
