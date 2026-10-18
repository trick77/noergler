package review

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/riptide"
	"github.com/trick77/noergler/internal/store"
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
	onPRDiff   func() // runs inside FetchPRDiff, e.g. a shutdown landing mid-fetch
	commitDiff string
	commitErr  error
	changes    []string
	changesErr error

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
	// Calls is every reply and resolve in order, "reply:<id>" and
	// "resolve:<id>", for the tests that pin which comes first.
	Calls      []string
	replyErr   error
	resolveErr error

	// UpdateVersions records the version each UpdatePRComment was sent
	// with; the optimistic-locking version is otherwise invisible to a
	// test that only reads Updates.
	UpdateVersions []int
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
	if f.onPRDiff != nil {
		f.onPRDiff()
	}
	return f.prDiff, f.prDiffErr
}

func (f *fakeBitbucket) FetchCommitDiff(_ context.Context, _, _, _, _ string) (string, error) {
	return f.commitDiff, f.commitErr
}

func (f *fakeBitbucket) FetchPRChanges(_ context.Context, _, _ string, _ int) ([]string, error) {
	return f.changes, f.changesErr
}

func (f *fakeBitbucket) FetchFileContent(_ context.Context, _, _, commit, path string) (string, error) {
	key := commit + ":" + path
	if err, ok := f.fileErr[key]; ok {
		return "", err
	}
	if content, ok := f.files[key]; ok {
		return content, nil
	}
	// A 404, as the real client reports a missing file.
	return "", &bitbucket.StatusError{Method: "GET", Path: "/raw/" + path, Status: 404}
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
	f.Calls = append(f.Calls, fmt.Sprintf("reply:%d", parent))
	if f.replyErr != nil {
		return f.replyErr
	}
	f.Replies = append(f.Replies, postedComment{PRID: prID, Text: text, Line: parent})
	return nil
}

func (f *fakeBitbucket) ResolveCommentThread(_ context.Context, _, _ string, _, commentID int) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Calls = append(f.Calls, fmt.Sprintf("resolve:%d", commentID))
	return f.resolveErr
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

func (f *fakeBitbucket) UpdatePRComment(_ context.Context, _, _ string, prID, commentID, version int, text string) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.updateErr != nil {
		return 0, f.updateErr
	}
	f.UpdateVersions = append(f.UpdateVersions, version)
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
	lastErr    error
	summary    *store.SummaryComment
	existing   []store.Finding
	prCost     *int64
	rollup     *store.RollupSnapshot
	// languages is user_settings, keyed by lower-cased user name.
	languages map[string]lang.Lang
	status    *store.PRStatus

	upsertErr  error
	attemptErr error
	runErr     error
	failAll    bool

	Upserts   []store.PRUpsert
	Runs      []store.Run
	Attempts  []store.Attempt
	Findings  []store.Finding
	Ignored   int
	Reactived int
	Merged    int
	Declined  int
	Deleted   int
	Summaries []store.SummaryComment
	Claims    []store.RollupFinal
	Frozen    int
	// Fixed is every MarkFindingFixed call as [findingID, runID].
	Fixed [][2]int64
	// byComment serves FindingByComment; Replied records MarkFindingReplied.
	byComment map[int]*store.Finding
	Replied   []repliedCall
}

type repliedCall struct {
	FindingID int64
	Dispute   *string
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
	if f.lastErr != nil {
		return "", false, f.lastErr
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

// Reactivate clears the skip state and the tracked summary, as the real
// store's UPDATE does. A fake that only counted the call would leave the PR
// ignored, so a review following a reactivating mention would still skip and
// the seam the mention exists for would be invisible.
//
// The failure check comes first: the real statement is a single UPDATE, so a
// refused write commits nothing and leaves ignored_at and the summary pointer
// standing. Mutating before the check would let a reviewer bug that ignores a
// failed Reactivate pass.
func (f *fakeStore) Reactivate(context.Context, store.PRKey) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return err
	}
	f.Reactived++
	f.skipState = nil
	f.summary = nil
	return nil
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
	if f.runErr != nil {
		return 0, f.runErr
	}
	f.Runs = append(f.Runs, r)
	return int64(len(f.Runs)), nil
}

func (f *fakeStore) InsertAttempt(_ context.Context, a store.Attempt) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	// attemptErr is separate from failAll: the point of the fail-open test
	// is a store where ONLY the attempt write is broken, so everything else
	// must still succeed around it.
	if f.attemptErr != nil {
		return f.attemptErr
	}
	if err := f.err(); err != nil {
		return err
	}
	f.Attempts = append(f.Attempts, a)
	return nil
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

func (f *fakeStore) MarkFindingFixed(_ context.Context, findingID, runID int64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return err
	}
	f.Fixed = append(f.Fixed, [2]int64{findingID, runID})
	return nil
}

func (f *fakeStore) FindingByComment(_ context.Context, _ store.PRKey, commentID int) (*store.Finding, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return nil, err
	}
	return f.byComment[commentID], nil
}

func (f *fakeStore) PRStatus(context.Context, store.PRKey) (*store.PRStatus, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return nil, err
	}
	if f.status == nil {
		return &store.PRStatus{}, nil
	}
	return f.status, nil
}

func (f *fakeStore) UserLanguage(_ context.Context, userName string) (lang.Lang, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return "", err
	}
	return f.languages[strings.ToLower(userName)], nil
}

func (f *fakeStore) SetUserLanguage(_ context.Context, userName string, l lang.Lang) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return err
	}
	if f.languages == nil {
		f.languages = map[string]lang.Lang{}
	}
	f.languages[strings.ToLower(userName)] = l
	return nil
}

func (f *fakeStore) MarkFindingReplied(_ context.Context, findingID int64, dispute *string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return err
	}
	f.Replied = append(f.Replied, repliedCall{findingID, dispute})
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
	reply   inference.FindingReplyResult
	// onReview runs during the gateway call, between prepare and post.
	onReview func()
	// onMention plays the model in a tool loop: it may run the offered
	// tools and returns the answer.
	onMention func(ctx context.Context, tools []string, run func(name, args string) string) string

	Reviews  []inference.ReviewRequest
	Mentions []inference.MentionRequest
	Replies  []inference.FindingReplyRequest
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
	if f.onReview != nil {
		f.onReview()
	}
	return f.review
}

func (f *fakeLLM) Mention(ctx context.Context, req inference.MentionRequest) inference.MentionResult {
	f.Mentions = append(f.Mentions, req)
	if f.onMention != nil {
		var names []string
		for _, t := range req.Tools {
			names = append(names, t.Name)
		}
		run := func(name, args string) string {
			for _, t := range req.Tools {
				if t.Name == name {
					return t.Run(ctx, args)
				}
			}
			return ""
		}
		return inference.MentionResult{Outcome: inference.OutcomeOK, Answer: f.onMention(ctx, names, run)}
	}
	return f.mention
}

func (f *fakeLLM) FindingReply(_ context.Context, req inference.FindingReplyRequest) inference.FindingReplyResult {
	f.Replies = append(f.Replies, req)
	return f.reply
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
type fakeTokens struct {
	mu       sync.Mutex
	perToken int

	// Counted records the byte length of every text handed to Count. The
	// byte pre-check in fetchCumulativeDiff is only observable as an
	// absence, so the sizes must be recorded to assert one.
	Counted []int
}

func (f *fakeTokens) Count(text string) int {
	f.mu.Lock()
	f.Counted = append(f.Counted, len(text))
	n := f.perToken
	f.mu.Unlock()
	if n <= 0 {
		n = 4
	}
	return (len(text) + n - 1) / n
}

// counted reports whether a text of exactly n bytes was tokenized.
func (f *fakeTokens) counted(n int) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, got := range f.Counted {
		if got == n {
			return true
		}
	}
	return false
}
