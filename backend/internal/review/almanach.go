package review

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/trick77/noergler/internal/almanach"
	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
	"github.com/trick77/ship/kit/logging"
)

// inflightSet counts the note calls on the pool per PR. A count, not a
// presence: two rounds of one PR can be in flight at once (the note runs
// unheld), and the first to finish must not clear the second.
type inflightSet struct {
	mu   sync.Mutex
	keys map[store.PRKey]int
}

func (s *inflightSet) add(k store.PRKey) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.keys == nil {
		s.keys = map[store.PRKey]int{}
	}
	s.keys[k]++
}

func (s *inflightSet) remove(k store.PRKey) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.keys[k] <= 1 {
		delete(s.keys, k)
		return
	}
	s.keys[k]--
}

func (s *inflightSet) has(k store.PRKey) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.keys[k] > 0
}

// AlmanachEmitter is the part of the almanach adapter the pipeline uses.
type AlmanachEmitter interface {
	Enabled() bool
	EmitNote(ctx context.Context, n almanach.Note) bool
}

// almanachDrain bounds how many pending notes one merge sends on: enough to
// catch up after an Almanach outage over a few merges, small enough to
// keep the worker turn short.
const almanachDrain = 10

// almanachJob is what the spawned note call carries: the fields it needs
// and nothing of the plan. The prompt and the review answer are the
// conversation it continues, so they stay resident until the call ends:
// one prompt per spawned call, the documented cost of the piggyback.
type almanachJob struct {
	key          store.PRKey
	prTag        string
	team         string
	runID        int64
	sourceCommit string
	prTitle      string
	ticketKey    string
	ticketType   string
	prompt       string
	answer       string
	instruction  string
	promptTokens int
}

// almanachOn reports whether this team feeds Almanach: an emitter that is
// enabled. No block in teams.yaml, no note call, no row.
func (r *Reviewer) almanachOn() bool { return r.almanach != nil && r.almanach.Enabled() }

// spawnAlmanachNote hands the note call to the pool after a successful
// review posted. Nothing to do without a run row (the review did not
// complete), without the model's answer (nothing to continue) or when the
// conversation plus the note turn would not fit.
func (r *Reviewer) spawnAlmanachNote(ctx context.Context, plan *reviewPlan, team string, sched Scheduler) {
	if !r.almanachOn() {
		return
	}
	// A skip means this round writes no note; without an earlier round's
	// note the merge sends Almanach nothing. Say which gate held.
	var skip string
	switch {
	case plan.result.Outcome != inference.OutcomeOK:
		skip = "review outcome " + plan.result.Outcome.String()
	case plan.runID == 0:
		skip = "no run row"
	case plan.result.Raw == "":
		skip = "empty review answer"
	}
	if skip != "" {
		r.log.InfoContext(ctx, plan.prTag+": almanach note skipped, "+skip)
		return
	}
	job := almanachJob{
		key: plan.key, prTag: plan.prTag, team: team, runID: plan.runID,
		sourceCommit: plan.sourceCommit, prTitle: deref(plan.upsert.Title),
		prompt: plan.prompt, answer: plan.result.Raw,
	}
	if plan.ticket != nil {
		job.ticketKey, job.ticketType = plan.ticket.Key, plan.ticket.IssueType
	}
	job.instruction = inference.RenderAlmanachNotePrompt(inference.AlmanachNotePromptRequest{
		Template: r.almanachTmpl, Repo: plan.repo, Project: plan.project,
		TicketKey: job.ticketKey, TicketType: job.ticketType, MaxChars: r.almanachMaxChars,
	})
	// The review's count already holds the system message, the prompt and
	// the review schema; the turn adds the answer, the instruction and the
	// note schema. Over the ceiling the call is not made at all.
	job.promptTokens = plan.promptTokens + r.tokens.Count(job.answer) + r.tokens.Count(job.instruction) +
		r.almanachSchemaTokens()
	if r.llm.Ready() && job.promptTokens > r.llm.FitCeiling() {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: almanach note skipped, %d tokens exceed the fit ceiling of %d",
			plan.prTag, job.promptTokens, r.llm.FitCeiling()))
		return
	}
	r.almanachInflight.add(plan.key)
	sched.Spawn(ctx, plan.key, team, "almanach note", func(c context.Context) { r.almanachNote(c, job) })
}

// almanachSchemaTokens counts the note schema once: it is a constant, and
// the count runs on the worker.
func (r *Reviewer) almanachSchemaTokens() int {
	r.almanachSchemaOnce.Do(func() { r.almanachSchemaCount = r.tokens.Count(inference.AlmanachNoteSchemaJSON()) })
	return r.almanachSchemaCount
}

// almanachNote is the spawned call: the gateway turn, the blunt checks, the
// row, the call record, and the emit when the merge already happened. It
// runs unheld, so every write re-asks ownership first and the store orders
// the rounds by run id. Its HTTP count is its own: the review's totals
// line is already out.
func (r *Reviewer) almanachNote(ctx context.Context, job almanachJob) {
	ctx, counter := httpstats.WithScope(ctx)
	defer func() {
		if n := counter.Summarize()["inference"]; n > 0 {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: almanach note HTTP totals - inference=%d", job.prTag, n))
		}
	}()

	res := r.llm.AlmanachNote(ctx, inference.AlmanachNoteRequest{
		ReviewPrompt: job.prompt, ReviewAnswer: job.answer, Instruction: job.instruction, PromptTokens: job.promptTokens,
	})
	if res.Cost.Answered {
		r.logCost(ctx, job.prTag+": almanach note", res.Cost)
	}

	status := inference.AlmanachNoAnswer
	switch res.Outcome {
	case inference.OutcomeOK:
		status = r.storeAlmanachNote(ctx, job, res)
	case inference.OutcomeUnparseable:
		status = inference.AlmanachInvalid
		if res.Cut {
			// The model's output limit ended the answer: the note did not
			// fit, which the ledger should say as a length drop.
			status = inference.AlmanachTooLong
		}
	default:
		r.log.InfoContext(ctx, fmt.Sprintf("%s: almanach note not written: %s (%v)", job.prTag, res.Outcome, res.Err))
	}
	runID := job.runID
	safeDBErr(ctx, r.log, "InsertAlmanachCall", func() error {
		return r.store.InsertAlmanachCall(ctx, store.AlmanachCall{
			Key: job.key, TeamSlug: r.TeamSlug, RunID: &runID, Outcome: res.Outcome.String(),
			NoteStatus: status, Call: r.llmCall(res.Cost),
		})
	})
	r.log.InfoContext(ctx, fmt.Sprintf("%s: almanach note %s (run %d)", job.prTag, status, job.runID))

	// The merge may have come and gone while the call ran: finalize then
	// left the row for this round to fill, and the emit is this round's.
	// The count drops before the status read: a merge landing between the
	// two sees no flight and emits itself, and if both paths then send the
	// same note, Almanach dedupes it. The other order would lose the note.
	r.almanachInflight.remove(job.key)
	st := safeDB(ctx, r.log, "AlmanachNoteStatus", func() (store.AlmanachNoteState, error) {
		st, _, err := r.store.AlmanachNoteStatus(ctx, job.key)
		return st, err
	})
	switch {
	case st.Finalized && st.HasNote && !st.Emitted:
		r.emitAlmanach(ctx, r.TeamSlug, &job.key)
	case st.Finalized && !st.HasNote && !r.almanachInflight.has(job.key):
		// The merge deferred its emit to the rounds; this last one wrote
		// nothing. With another still in flight, that one may.
		r.log.InfoContext(ctx, job.prTag+": merged without an almanach note, nothing to send")
	}
}

// storeAlmanachNote runs the checks and the upsert and returns the note's
// status. Spelling first (it changes lengths), then the shape, the secret
// scan and the caps, then the empty case, then ownership, then the row.
func (r *Reviewer) storeAlmanachNote(ctx context.Context, job almanachJob, res inference.AlmanachNoteResult) string {
	note := res.Note
	chars, reason := inference.ValidateAlmanachNote(&note, r.almanachMaxChars)
	if reason != "" {
		return reason
	}
	if note.Empty() {
		return inference.AlmanachNothingLearned
	}
	if r.owns != nil && !r.owns(job.key.Project, job.key.Repo) {
		// The release purged the PR's rows; the note has no row to land on.
		r.log.InfoContext(ctx, job.prTag+": almanach note not written, repo no longer claimed by the team")
		return store.NoteNoRow
	}
	raw, err := json.Marshal(note)
	if err != nil {
		return inference.AlmanachInvalid
	}
	written := safeDB(ctx, r.log, "UpsertAlmanachNote", func() (string, error) {
		return r.store.UpsertAlmanachNote(ctx, store.AlmanachNote{
			Key: job.key, TeamSlug: r.TeamSlug, RunID: job.runID, ReviewedCommit: job.sourceCommit,
			PRTitle: job.prTitle, TicketKey: job.ticketKey, JiraIssueType: job.ticketType, ChangeKind: note.ChangeKind,
			Note: raw, Chars: chars, ModelLabel: r.llm.Label(),
		})
	})
	if written == "" {
		// The store failed open: the warn log has the error, the call row
		// says the note never reached the table.
		return store.NoteStoreFailed
	}
	return written
}

// finalizeAlmanach stamps the PR's note at merge and, unless this PR's
// round is still in flight, sends the team's pending notes on. No model
// call: the merge only ships what the rounds prepared.
func (r *Reviewer) finalizeAlmanach(ctx context.Context, key store.PRKey, prTag string, payload *webhook.Payload) {
	if !r.almanachOn() {
		return
	}
	stamped, err := r.store.FinalizeAlmanachNote(ctx, key, r.TeamSlug, store.AlmanachFinal{
		MergedAt: time.Now().UTC(), MergeCommit: payload.MergeCommitSHA(),
		SourceCommit: payload.PullRequest.FromRef.LatestCommit, PRTitle: payload.PullRequest.Title,
	})
	if err != nil {
		// Not final, so never pending: this PR's note is not sent by this
		// merge or any later drain. The team's other notes still go.
		r.log.WarnContext(ctx, prTag+": almanach note not final, FinalizeAlmanachNote failed", logging.Err(err))
		r.emitAlmanach(ctx, r.TeamSlug, nil)
		return
	}
	if stamped {
		r.log.InfoContext(ctx, prTag+": almanach note final")
	}
	if r.almanachInflight.has(key) {
		// This PR's own note lands in a moment and emits itself, alone; the
		// team's other pending notes wait for the next merge's drain.
		r.log.InfoContext(ctx, prTag+": almanach note still in flight, emit deferred to it")
		return
	}
	// Read before the drain, so Emitted means sent by an earlier merge.
	st, found, err := r.store.AlmanachNoteStatus(ctx, key)
	if err != nil {
		r.log.WarnContext(ctx, prTag+": almanach note state unread, AlmanachNoteStatus failed", logging.Err(err))
		r.emitAlmanach(ctx, r.TeamSlug, nil)
		return
	}
	sent := r.emitAlmanach(ctx, r.TeamSlug, nil)
	// What this merge had for Almanach: an empty Almanach after a merge
	// must be readable from noergler's log, not guessed.
	switch {
	case !found:
		// The store writes no stub without a pull_requests row.
		r.log.InfoContext(ctx, prTag+": no almanach note, no PR row (never reviewed, or purged)")
	case st.Emitted:
		r.log.InfoContext(ctx, prTag+": almanach note already sent")
	case !st.HasNote:
		// A stub: no round wrote a note (its "almanach note <status>" or
		// "skipped" line says why), or none has run yet and a late one
		// fills it and sends itself.
		r.log.InfoContext(ctx, prTag+": almanach note final without a note, nothing to send")
	case !slices.Contains(sent, key) && !r.emittedSince(ctx, key):
		// Sent, the emitter logged "taken"; sent by a round that finished
		// beside this drain, its own line says so; else this one.
		r.log.InfoContext(ctx, prTag+": almanach note not sent (refused or behind older notes), left for the next merge")
	}
}

// emittedSince reports whether the PR's note is marked sent now: a round
// that finished beside the merge's drain may have sent it. A failed read
// says no, so the "not sent" line errs towards being written.
func (r *Reviewer) emittedSince(ctx context.Context, key store.PRKey) bool {
	st, _, err := r.store.AlmanachNoteStatus(ctx, key)
	return err == nil && st.Emitted
}

// dropAlmanachNote forgets a declined or deleted PR's note; the store keeps
// a final one.
func (r *Reviewer) dropAlmanachNote(ctx context.Context, key store.PRKey) {
	if !r.almanachOn() {
		return
	}
	safeDBErr(ctx, r.log, "DeleteAlmanachNote", func() error { return r.store.DeleteAlmanachNote(ctx, key) })
}

// emitAlmanach sends the team's final notes Almanach has not taken yet,
// oldest first, up to almanachDrain, and marks each one Almanach answered
// 2xx for. A note it refuses stays for the next merge. With only set the
// caller is a round on the pool: it sends its own PR's note and nothing
// else, so the catch-up drain stays a worker job. Returns the PRs whose
// note Almanach took.
func (r *Reviewer) emitAlmanach(ctx context.Context, team string, only *store.PRKey) []store.PRKey {
	pending, err := r.store.PendingAlmanachNotes(ctx, team, almanachDrain)
	if err != nil {
		what := "the team's pending almanach notes"
		if only != nil {
			what = only.Tag() + "'s almanach note"
		}
		r.log.WarnContext(ctx, what+" not sent, PendingAlmanachNotes failed", logging.Err(err))
		return nil
	}
	if only != nil {
		i := slices.IndexFunc(pending, func(p store.AlmanachPending) bool { return p.Key == *only })
		if i < 0 {
			// Behind the oldest almanachDrain (the next merge's drain sends
			// it), or a concurrent merge already sent it.
			r.log.InfoContext(ctx, only.Tag()+": almanach note not in this drain (behind older notes or already sent), nothing sent")
			return nil
		}
		pending = pending[i : i+1]
	}
	var sent []store.PRKey
	for _, p := range pending {
		n := almanachNoteOf(p, r.claimsFor(p.Key.Project))
		if !r.almanach.EmitNote(ctx, n) {
			// The emitter logged why; the rest of the list would meet the
			// same Almanach.
			return sent
		}
		sent = append(sent, p.Key)
		runID := p.RunID
		safeDBErr(ctx, r.log, "MarkAlmanachEmitted", func() error { return r.store.MarkAlmanachEmitted(ctx, p.Key, runID) })
	}
	return sent
}

// claimsFor is the team's ownership of one project, from the live snapshot:
// what Almanach places the note by.
func (r *Reviewer) claimsFor(project string) []almanach.Claim {
	if r.team == nil {
		return nil
	}
	t := r.team()
	if t == nil {
		return nil
	}
	var out []almanach.Claim
	for _, sc := range t.Projects {
		if sc.Key != project {
			continue
		}
		if len(sc.Repos) == 0 {
			out = append(out, almanach.Claim{ProjectKey: sc.Key})
			continue
		}
		for _, repo := range sc.Repos {
			out = append(out, almanach.Claim{ProjectKey: sc.Key, RepoSlug: repo})
		}
	}
	return out
}

// almanachNoteOf is the wire note for one pending row.
func almanachNoteOf(p store.AlmanachPending, claims []almanach.Claim) almanach.Note {
	n := almanach.Note{
		ProjectKey: p.Key.Project, RepoSlug: p.Key.Repo, PRID: p.Key.PRID, PRKey: p.Key.Tag(),
		PRTitle: p.PRTitle, TicketKey: p.TicketKey, JiraIssueType: p.JiraIssueType, ChangeKind: p.ChangeKind,
		ReviewedCommit: p.ReviewedCommit, SourceCommitSHA: p.SourceCommit, MergeCommitSHA: p.MergeCommit,
		MergedAt: p.MergedAt, RunID: p.RunID, ModelLabel: p.ModelLabel,
		IssuesPosted: p.IssuesPosted, SuggestionsPosted: p.SuggestionsPosted,
		Note: p.Note, Claims: claims,
		Rounds: almanach.Rounds{Count: p.Rounds, InputTokens: p.RoundsInputTokens, OutputTokens: p.RoundsOutputTokens, CostNanoUSD: p.RoundsCostNanoUSD},
		Review: almanach.ReviewTotals{Runs: p.Review.Runs, PromptTokens: p.Review.PromptTokens, CachedTokens: p.Review.CachedTokens,
			CompletionTokens: p.Review.CompletionTokens, ElapsedMS: p.Review.ElapsedMS, CostNanoUSD: p.Review.CostNanoUSD, Models: p.Review.Models},
	}
	if p.NoteCallFound {
		n.NoteCall = &almanach.Call{Outcome: p.NoteCallOutcome, NoteStatus: p.NoteStatus, LLMCall: p.NoteCall}
	}
	return n
}
