package review

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
	"github.com/trick77/ship/kit/logging"
)

// contextExpansionRatio decides when a PR counts as small: its files plus
// room to expand them still fit the budget.
const contextExpansionRatio = 1.5

// nanoPerUSD converts the DB's BIGINT nano-USD to the USD used at the edges.
const nanoPerUSD = 1_000_000_000.0

// prepare is everything up to and including prompt assembly: the guards, the
// Bitbucket fetches, Jira, and the token budget. It returns the plan the
// inference and posting stages need, and the ctx carrying pr_tag and the
// httpstats scope. The one entry is ReviewPullRequestStaged (stage.go):
// prepare, then infer and post through the Scheduler.
//
// skipAuthorCheck is set by an @mention: it bypasses the author and actor
// gates and the cost cap, because a human asked for this review explicitly.
//
// The order of the guards below is deliberate and fixed, not incidental.
// The "Guard order" block in review_test.go pins it guard by guard
// (TestReviewRequiresAProjectAndRepo, TestSkipsDisallowedAuthorAndIgnoredActor,
// TestIgnoredPRIsSkippedWithoutAnyAPICall,
// TestDeletedSummaryIgnoresButTransientErrorProceeds, TestOptOutBranchKeyword,
// TestAgentsMDGates, TestCostCap). Two rules run through all of it:
// every store call goes through safeDB, and every skip that did not review
// the new commit writes the PRIOR commit back, so raising a limit later
// re-reviews the accumulated range instead of skipping it.
//
// ok is false when the review is over: the caller must not continue. Every
// such exit has already logged its own HTTP totals.
func (r *Reviewer) prepare(ctx context.Context, payload *webhook.Payload, skipAuthorCheck bool) (*reviewPlan, context.Context, bool) {
	pr := payload.PullRequest
	ctx, key, prTag, ok := prContext(ctx, payload)
	if !ok {
		r.log.ErrorContext(ctx, "Could not extract project/repo from webhook payload")
		return nil, ctx, false
	}
	project, repo := key.Project, key.Repo
	kind := runKind(skipAuthorCheck)

	// Per-review HTTP accounting. The bitbucket and jira transports already
	// record into the scope; without one opened here every count was dropped
	// and the totals line never existed.
	// NOT deferred: with the inference call staged off the worker, a defer
	// here fires before the gateway call and before posting, so the totals
	// line would report inference=0 and none of the post stage's Bitbucket
	// calls. Each exit below logs its own; the full path logs in post.
	ctx, httpCounter := httpstats.WithScope(ctx)

	author := pr.Author.User.Name

	// 1b. The team may have given the repo up while this sat in the queue.
	if r.owns != nil && !r.owns(project, repo) {
		r.log.InfoContext(ctx, prTag+": repo no longer claimed by the team, skipping")
		return nil, ctx, r.abort(ctx, key, kind, SkipRepoNotClaimed, prTag, httpCounter)
	}

	// 2. Author gate. The ignore list wins over the allow list inside
	// IsAutoReviewAuthor, so a bare false cannot say which list decided.
	// Name the ignore list when it is the reason: reporting every skip as an
	// allow-list miss would misstate why an ignored bot was skipped.
	if !skipAuthorCheck {
		if autoReview, ignored := r.autoReviewDecision(author); !autoReview {
			// The log wording and the stored reason come from the SAME
			// decision: a dashboard that groups an ignored bot under
			// "not in auto-review authors" while the log says otherwise
			// reintroduces exactly the confusion this branch exists to end.
			reason, why := "not in auto-review authors", SkipNotAutoAuthor
			if ignored {
				reason, why = r.ignoredWhy(author), SkipIgnoredAuthor
			}
			r.log.InfoContext(ctx, fmt.Sprintf("Skipping %s by %s (%s)", prTag, author, reason))
			return nil, ctx, r.abort(ctx, key, kind, why, prTag, httpCounter)
		}
	}
	// 3. A push by an ignored account (CI amending someone's PR) is not a
	// reason to re-review; the author's next push is.
	if !skipAuthorCheck && payload.Actor != nil && r.isIgnoredAuthor(payload.Actor.Name) {
		r.log.InfoContext(ctx, fmt.Sprintf("Skipping %s: pushed by %s (%s)", prTag, payload.Actor.Name, r.ignoredWhy(payload.Actor.Name)))
		return nil, ctx, r.abort(ctx, key, kind, SkipIgnoredAuthor, prTag, httpCounter)
	}

	// 4. Skip state. If the user removed our summary comment, or told
	// noergler to stop, they want it to leave this PR alone.
	if why := r.checkSkipState(ctx, key, prTag); why != SkipNone {
		return nil, ctx, r.abort(ctx, key, kind, why, prTag, httpCounter)
	}

	// Everything this run posts is in the author's language, whoever
	// triggered it: the summary and the findings are the author's.
	l := r.languageFor(ctx, author)

	upsert := store.PRUpsert{
		Key: key, TeamSlug: r.TeamSlug,
		Author: &author, Title: &pr.Title, OpenedAt: epochMSToTime(pr.CreatedDate),
	}

	// 5. Opt-out keyword in the branch name.
	if keyword := r.cfg.OptOutBranchKeyword; keyword != "" &&
		strings.Contains(strings.ToLower(pr.FromRef.Branch()), strings.ToLower(keyword)) {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: branch %q contains opt-out keyword %q, skipping review",
			prTag, pr.FromRef.Branch(), keyword))
		// The pointer advances here: an opt-out branch is a stable state of
		// the PR, not a transient failure.
		prReviewID := r.upsert(ctx, upsert, pr.FromRef.LatestCommit)
		r.postOrUpdateSummary(ctx, project, repo, pr.ID, prReviewID,
			render.OptOutBranchSummary(l, keyword, pr.FromRef.Branch()))
		return nil, ctx, r.abort(ctx, key, kind, SkipBranchOptOut, prTag, httpCounter)
	}

	// 5b. The SNAPSHOT task: no model, so before every guard that can end
	// the run without a review (snapshottask.go).
	r.checkSnapshotTask(ctx, pr, key, upsert, prTag, l)

	md := r.fetchRepoInstructions(ctx, project, repo, pr.FromRef.LatestCommit, pr.ToRef.Branch())
	repoInstructions := md.text

	// 6a. AGENTS.md over the file cap: it exists and is far over any token
	// limit (the cap is 1 MiB, the limit thousands of tokens), so guard 7's
	// notice, its size estimated at four bytes a token (the notice says ~).
	// Without a token limit it is merely unreadable: 6b.
	if repoInstructions == "" && md.oversize > 0 && r.cfg.AgentsMDMaxTokens > 0 {
		n := md.oversize / 4
		r.log.InfoContext(ctx, fmt.Sprintf(
			"%s: AGENTS.md over the file cap (%d bytes, ~%d tokens > %d), skipping review",
			prTag, md.oversize, n, r.cfg.AgentsMDMaxTokens))
		prReviewID, _ := r.upsertKeepingPrior(ctx, upsert, key)
		r.postOrUpdateSummary(ctx, project, repo, pr.ID, prReviewID,
			render.AgentsMDTooLargeSummary(l, n, r.cfg.AgentsMDMaxTokens, r.cfg.AgentsMDCustomLink))
		return nil, ctx, r.abort(ctx, key, kind, SkipAgentsMDTooLarge, prTag, httpCounter)
	}

	// 6b. AGENTS.md required and unreadable on every branch: a fault, like a
	// diff that would not fetch. Not "missing": the summary would say so
	// wrongly. No notice, the pointer stays, the next push reviews this one
	// too.
	if repoInstructions == "" && (md.unknown || md.oversize > 0) && r.cfg.RequireAgentsMD {
		if ctx.Err() == nil {
			r.log.WarnContext(ctx, prTag+": AGENTS.md unreadable, not reviewed (the next push reviews this range too)")
			r.recordAttempt(ctx, store.Attempt{
				Key: key, TeamSlug: r.TeamSlug, Kind: kind,
				Outcome: inference.OutcomeError.String(),
			})
		}
		return nil, ctx, r.abort(ctx, key, kind, SkipNone, prTag, httpCounter)
	}

	// 6. AGENTS.md required and absent. The prior commit stands: nothing of
	// this push was reviewed, so the review that follows a fix covers it.
	if repoInstructions == "" && r.cfg.RequireAgentsMD {
		r.log.InfoContext(ctx, fmt.Sprintf(
			"%s: no AGENTS.md found on PR or target branch, skipping review (set REVIEW_REQUIRE_AGENTS_MD=false to override)", prTag))
		prReviewID, _ := r.upsertKeepingPrior(ctx, upsert, key)
		r.postOrUpdateSummary(ctx, project, repo, pr.ID, prReviewID, render.AgentsMDMissingSummary(l))
		return nil, ctx, r.abort(ctx, key, kind, SkipNoAgentsMD, prTag, httpCounter)
	}

	// 7. AGENTS.md over the hard token limit.
	if repoInstructions != "" && r.cfg.AgentsMDMaxTokens > 0 {
		if n := r.tokens.Count(repoInstructions); n > r.cfg.AgentsMDMaxTokens {
			r.log.InfoContext(ctx, fmt.Sprintf(
				"%s: AGENTS.md too large (%d > %d tokens), skipping review (set REVIEW_AGENTS_MD_MAX_TOKENS=0 to disable the hard limit)",
				prTag, n, r.cfg.AgentsMDMaxTokens))
			prReviewID, _ := r.upsertKeepingPrior(ctx, upsert, key)
			r.postOrUpdateSummary(ctx, project, repo, pr.ID, prReviewID,
				render.AgentsMDTooLargeSummary(l, n, r.cfg.AgentsMDMaxTokens, r.cfg.AgentsMDCustomLink))
			return nil, ctx, r.abort(ctx, key, kind, SkipAgentsMDTooLarge, prTag, httpCounter)
		}
	}

	// 8. Per-PR cost cap, mention or not (decided: a mention no longer
	// overrides it; raising the limit is the way on). Fail open: an
	// unpriced PR never blocks.
	if _, cumulative, over := r.overCap(ctx, key); over {
		// 2 decimals, not the 3 the cost log lines use: these are the
		// same two figures costLimitNotice renders into the banner
		// below, and a log that disagrees with the posted comment
		// about the number that paused reviews is worse than a log
		// that is less precise than its neighbours.
		r.log.InfoContext(ctx, fmt.Sprintf(
			"%s: PR cost $%.2f >= limit $%.2f - skipping review (raise REVIEW_MAX_PR_COST_USD to go on)",
			prTag, cumulative, r.cfg.MaxPRCostUSD))
		prReviewID, _ := r.upsertKeepingPrior(ctx, upsert, key)
		r.costLimitNotice(ctx, l, project, repo, pr.ID, prReviewID, cumulative, r.cfg.MaxPRCostUSD, true, false)
		return nil, ctx, r.abort(ctx, key, kind, SkipCostCap, prTag, httpCounter)
	}

	r.log.InfoContext(ctx, fmt.Sprintf("Starting review of %s by %s (branch: %s)", prTag, author, pr.FromRef.Branch()))
	started := time.Now()
	sw := newStopwatch(started)

	sourceCommit := pr.FromRef.LatestCommit
	lastReviewed, _ := r.priorCommit(ctx, key)

	// 9. Incremental review when the event is a push and we have a pointer.
	rawDiff, cumulativePR, incrementalFrom, diffWhy, err := r.resolveDiff(ctx, payload, key, prTag, sourceCommit, lastReviewed, upsert, l)
	sw.lap(stageDiff)
	if err != nil || diffWhy != SkipNone {
		// A diff that would not fetch is a fault, recorded as an error like
		// an inference one, so an outage is visible. A cancelled ctx is a
		// shutdown, not an outage.
		if err != nil && ctx.Err() == nil {
			r.recordAttempt(ctx, store.Attempt{
				Key: key, TeamSlug: r.TeamSlug, Kind: kind,
				Outcome: inference.OutcomeError.String(),
			})
		}
		return nil, ctx, r.abort(ctx, key, kind, diffWhy, prTag, httpCounter)
	}

	// The line counts are the whole PR's scope, taken before compression.
	prep := r.prepareFiles(ctx, project, repo, rawDiff, sourceCommit, prTag)
	files, contentSkipped, oversized, diffAdded, diffRemoved := prep.files, prep.contentSkipped, prep.oversized, prep.added, prep.removed
	// 13. Nothing reviewable after the content fetch. A diff that changes
	// test screenshots and nothing else (a baseline refresh, the drift the
	// Visual changes section exists for) still goes on, to a model that can
	// see them.
	diffHasShot := diff.HasScreenshot(rawDiff)
	baselinesOnly := diffHasShot && r.llm.Vision()
	if len(files) == 0 && !baselinesOnly {
		r.log.InfoContext(ctx, prTag+" has no reviewable files after content fetch, skipping")
		return nil, ctx, r.abort(ctx, key, kind, SkipNoReviewable, prTag, httpCounter)
	}

	// Every reviewable changed file, size-gated ones included: the set the
	// +/- lines cover and the merge-time rollup counts
	// (diff.Scope), so the run row, the footer and the rollup
	// agree. What was not reviewed is a subset, named on its own lines.
	totalFiles := len(files) + len(oversized)

	// 13b. AGENTS.md below the root, for the whole PR's paths: the full
	// cumulative diff's on an incremental review (unfiltered, never the
	// budgeted copy), this diff's on a full one (it is the PR). A cumulative
	// diff that could not be read falls back to the push, logged: the set
	// then follows the push for that run. After the root's gates: those
	// decide whether a review runs, a nested file never does.
	paths := prPaths(rawDiff)
	if incrementalFrom != "" {
		if cumulativePR.paths != nil {
			paths = cumulativePR.paths
		} else {
			r.log.InfoContext(ctx, prTag+": nested AGENTS.md looked up for the push's paths only: the PR's diff was not read")
		}
	}
	rootInstructions := repoInstructions
	nested, nestedLeft, nestedIncomplete := r.fetchNestedAgentsMD(ctx, project, repo, sourceCommit, prTag, paths)
	repoInstructions, nestedKept, nestedDropped := withNested(repoInstructions, nested, r.cfg.AgentsMDMaxTokens, r.tokens.Count)
	nestedLeft = leftOver(nestedLeft, nestedDropped)
	if len(nestedDropped) > 0 {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: left out over the AGENTS.md limit of %d tokens: %s",
			prTag, r.cfg.AgentsMDMaxTokens, strings.Join(nestedDropped, ", ")))
	}

	// 13c. Callers in files the PR does not change: one search, at most.
	// Before the budget, which the block's tokens come off. Names from the
	// whole diff as fetched: compression drops deleted files and
	// removal-only hunks, the names whose callers break first. The PR's own
	// files, new and old paths: on an incremental review only the full
	// cumulative diff names them all; without it there is no search, since
	// an earlier push's file would read as unchanged.
	var prFiles []string
	prFilesKnown := true
	switch {
	case incrementalFrom == "":
		prFiles = append(prPaths(rawDiff), diff.SourcePaths(rawDiff)...)
	case cumulativePR.paths != nil:
		prFiles = append(append([]string{}, cumulativePR.paths...), cumulativePR.sources...)
	default:
		prFilesKnown = false
	}
	unchangedRefs := r.unchangedReferences(ctx, pr, key, prTag, diffParts(rawDiff), prFiles, prFilesKnown)
	unchangedBlock := diff.RenderUnchangedReferences(unchangedRefs)

	// 14. Small PRs get context expansion; large ones are compressed first.
	var otherModified, deletedPaths, renamedPaths []string
	budget := r.llm.InputTokenBudget()
	if unchangedBlock != "" {
		budget = max(0, budget-r.tokens.Count(unchangedBlock))
	}
	if diff.IsSmall(files, budget, r.template, r.tokens.Count, inference.FormatFileEntry, contextExpansionRatio) {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: small PR (%d files) - expanding context (before=%d, after=%d, dynamic=%v)",
			prTag, len(files), r.cfg.DiffExtraLinesBefore, r.cfg.DiffExtraLinesAfter, r.cfg.DiffAllowDynamicContext))
	} else {
		compression := diff.Compress(files, budget, r.template, r.tokens.Count, inference.FormatFileEntry)
		files = compression.IncludedFiles
		otherModified = compression.OtherModifiedPaths
		deletedPaths = compression.DeletedFilePaths
		renamedPaths = compression.RenamedFilePaths
		r.log.InfoContext(ctx, fmt.Sprintf(
			"%s: large PR - compression applied: %d included, %d other_modified, %d deleted, %d renamed",
			prTag, len(files), len(otherModified), len(deletedPaths), len(renamedPaths)))
		if len(otherModified) > 0 {
			// Named only in the prompt, so not reviewed: a count hides which.
			r.log.WarnContext(ctx, fmt.Sprintf("%s: %d file(s) over the %d-token budget are NOT reviewed, named to the model only: %s",
				prTag, len(otherModified), budget, capPaths(otherModified)))
		}
	}
	// Before expansion: it adds context lines the count must not read.
	sentAdded, sentRemoved := sentLines(files)
	files = diff.ExpandAllFiles(files, r.cfg.DiffExtraLinesBefore, r.cfg.DiffExtraLinesAfter,
		r.cfg.DiffMaxExtraLinesDynamicContext, r.cfg.DiffAllowDynamicContext)

	// 15. Compression can drop everything.
	if len(files) == 0 && !baselinesOnly {
		r.log.InfoContext(ctx, prTag+" has no reviewable files after compression, skipping")
		return nil, ctx, r.abort(ctx, key, kind, SkipNoReviewable, prTag, httpCounter)
	}
	// What the prompt carries, file by file, after compression and
	// expansion: the one place the per-file cost is visible.
	sentLine, fileTokens := renderFilesSent(files, r.tokens.Count)
	r.log.InfoContext(ctx, prTag+": "+sentLine)
	sw.lap(stageFiles)

	// 16. Cross-file context, Jira, previously-posted findings.
	relationships := diff.BuildRelationships(files)
	crossFile := diff.RenderRelationships(relationships)
	if crossFile != "" {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: cross-file context: %d relationship(s)", prTag, len(relationships)))
	}

	ticket, parentTicket, jiraDown := r.fetchTicket(ctx, pr.FromRef.Branch(), pr.Title, prTag)
	ticketContext := ticketContextFor(ticket, parentTicket, jiraDown)
	prDescription := r.prDescription(ctx, prTag, pr)

	// Not safeDB: its fallback "none" would post a PR summary with no
	// earlier findings and an approve over open ones. Unreadable, the
	// review goes on and the summary says what it cannot show.
	existing, err := r.store.ExistingFindings(ctx, key)
	existingUnknown := err != nil
	if existingUnknown {
		r.log.WarnContext(ctx, prTag+": earlier findings unreadable, the PR summary lists this run's only", logging.Err(err))
	}
	// Thread state, once, for everything after. A finding whose thread was
	// resolved or deleted on the PR is settled (user, 2026-10-28): the
	// prompt shows it closed and without an id (not raised again, never
	// claimed fixed), dedupe still holds its repeat back, and the summary's
	// list and verdict leave it out.
	existing, _ = r.closeThreads(ctx, key, prTag, existing, maxThreadChecks, false)
	// The root's rules from the target branch, never the PR's own: the
	// branch a rule gates must not lift it. A rule no longer marked blocks
	// nothing: its open tasks go.
	rulesMD, rulesKnown := r.targetAgentsMD(ctx, project, repo, pr.ToRef.Branch())
	// The nested files' rules join them, from the PR head (blockingRules):
	// every file found, one left out of the prompt over the token limit too,
	// since its rules still hold.
	blockingRules, markedRules := r.blockingRules(ctx, prTag, rulesMD, nested)
	existing = r.retireRemovedRules(ctx, key, prTag, sourceCommit, existing, markedRules, rulesKnown)
	// Only an incremental review may retire a finding: on a full one every
	// PR line is an added line, the buggy one too, so nothing would prove a
	// fix. No id listed, nothing to claim.
	posted := r.trimPreviouslyPosted(ctx, prTag, existing, budget, incrementalFrom != "")
	// Incremental only: a full review's subject is the whole PR, so a draft
	// recovers nothing and its stale parts win (mimo kept a reverted part).
	previousOverview := ""
	if incrementalFrom != "" {
		previousOverview = r.previousOverview(ctx, key, prTag)
	}
	sw.lap(stageContext)
	listed := map[int64]bool{}
	for _, p := range posted {
		if p.ID > 0 {
			listed[p.ID] = true
		}
	}

	assembled := inference.AssembleReviewPrompt(inference.AssembleRequest{
		Template:              r.template,
		Files:                 files,
		RepoInstructions:      repoInstructions,
		BlockingRules:         blockingRules,
		OtherModifiedPaths:    otherModified,
		DeletedFilePaths:      deletedPaths,
		RenamedFilePaths:      renamedPaths,
		TicketContext:         ticketContext,
		TicketComplianceCheck: r.cfg.TicketComplianceCheck,
		PRDescription:         prDescription,
		CrossFileContext:      crossFile,
		UnchangedReferences:   unchangedBlock,
		CumulativePRDiff:      cumulativePR.diff,
		CumulativePartial:     cumulativePR.partial,
		CumulativeOmitted:     cumulativePR.omitted,
		Incremental:           incrementalFrom != "",
		PreviouslyPosted:      posted,
		PreviousOverview:      previousOverview,
		Language:              l,
	}, r.tokens.Count)
	r.logPromptComposition(ctx, prTag, assembled, incrementalFrom != "")

	// The whole PR's screenshots, read from its changes only when this diff
	// or the whole PR's names one: a PR without any costs no call.
	// A baseline-only diff goes on even when nothing is left to send (all
	// stored, or deleted): the call is what refreshes the summary, whose
	// section would otherwise keep the line of a screenshot this push
	// removed or reverted.
	var visualPlan visualPlan
	if diffHasShot || (incrementalFrom != "" && (cumulativePR.scope == nil || cumulativePR.hasScreenshot)) {
		visualPlan = r.prepareVisual(ctx, key, prTag, sourceCommit, r.visualRoom(assembled.PromptTokens))
	}

	detail := &store.Pipeline{
		Prompt:    promptPartsOf(assembled, budget),
		Files:     filesSent(prep, files, deletedPaths, renamedPaths, otherModified, fileTokens),
		CrossFile: []string{},
		// Found and Tokens are the root file's: the footer's tip is about
		// the root AGENTS.md, which nested files never replace. The bloat
		// line adds the nested ones' tokens to it.
		AgentsMD: store.AgentsMDFacts{Found: rootInstructions != "", Tokens: rootTokens(rootInstructions, repoInstructions,
			assembled.Breakdown.RepoInstructions, r.tokens.Count), WarnTokens: r.cfg.AgentsMDWarnTokens,
			Nested: nestedKept, LeftOut: nestedLeft, Incomplete: nestedIncomplete},
		PreviouslyPosted: store.PreviouslyPosted{Shown: len(posted), Total: len(existing)},
		Decisions:        []store.FindingDecision{},
		Moved:            []store.MovedFinding{},
		PreExisting:      []store.PreExistingBug{},
		Commit:           sourceCommit,
	}
	// Names only: BuildRelationships emits one relationship per defining
	// file, and the card lists a name once even when two files define it.
	symbols := map[string]bool{}
	for _, rel := range relationships {
		if !symbols[rel.Symbol] {
			symbols[rel.Symbol] = true
			detail.CrossFile = append(detail.CrossFile, rel.Symbol)
		}
	}
	if ticket != nil {
		detail.Ticket = ticket.Key
	}
	if parentTicket != nil {
		detail.ParentTicket = parentTicket.Key
	}
	// The whole PR at this commit: a full review's own counts, an incremental
	// one's from the cumulative diff (nil when that fetch failed).
	if incrementalFrom == "" {
		detail.PRScope = &store.PRScope{Added: diffAdded, Removed: diffRemoved, Files: totalFiles}
	} else {
		detail.PRScope = cumulativePR.scope
	}

	// The cumulative diff is quotable too: on an incremental review it is
	// the only place an earlier push's change is shown.
	anchors := diff.BuildAnchorIndex(files)
	anchors.AddContextDiff(cumulativePR.diff)
	// The unchanged files' lines are quotable too, never anchors.
	anchors.AddContextLines(diff.ReferenceLines(unchangedRefs))
	var prAdded map[string]map[uint64]bool
	var history diff.AnchorIndex
	preUnknown := false
	if incrementalFrom != "" {
		// No history read (a failed fetch): every pre-existing claim drops,
		// rather than passing on one push's diff alone.
		prAdded, preUnknown = cumulativePR.added, cumulativePR.added == nil
		// The cumulative diff as shown, by file: a ticket requirement an
		// earlier push implemented is proven by its lines there.
		history = diff.BuildAnchorIndex(diff.FilesOf(cumulativePR.diff))
	}
	// The last of prepare's own work: what follows is the wait for a slot.
	sw.lap(stagePrompt)

	return &reviewPlan{
		sw:              sw,
		detail:          detail,
		key:             key,
		project:         project,
		repo:            repo,
		prID:            pr.ID,
		prTag:           prTag,
		upsert:          upsert,
		sourceCommit:    sourceCommit,
		incrementalFrom: incrementalFrom,
		mention:         skipAuthorCheck,
		blockingRules:   blockingRules,
		lang:            l,
		offer:           offerPROf(pr, key),
		started:         started,
		counter:         httpCounter,
		prompt:          assembled.Prompt,
		promptTokens:    assembled.PromptTokens,
		visual:          visualPlan,
		anchors:         anchors,
		prAdded:         prAdded,
		history:         history,
		preUnknown:      preUnknown,
		existing:        existing,
		existingUnknown: existingUnknown,
		listed:          listed,
		// Read from what the prompt carries (notFullContext): a deleted file
		// is never fetched, so it is never content-skipped to begin with.
		contentSkipped: notFullContext(contentSkipped, files),
		oversized:      oversized,
		overBudget:     otherModified,
		ticket:         ticket,
		parentTicket:   parentTicket,
		jiraDown:       jiraDown,
		// One that could not be read is no reason for the "add an
		// AGENTS.md" tip: it may well exist.
		agentsMDFound: rootInstructions != "" || md.unknown || md.oversize > 0,
		totalFiles:    totalFiles,
		diffAdded:     diffAdded,
		diffRemoved:   diffRemoved,
		sentAdded:     sentAdded,
		sentRemoved:   sentRemoved,
	}, ctx, true
}

// infer is the gateway call, and nothing else. It is the only stage that may
// run off the review worker.
func (r *Reviewer) infer(ctx context.Context, plan *reviewPlan) inference.ReviewResult {
	// Since prepare: the wait for a slot on the inference pool (near zero
	// when the call runs inline).
	plan.sw.lap(stageQueued)
	defer plan.sw.lap(stageGateway)
	return r.llm.Review(ctx, inference.ReviewRequest{
		Prompt:         plan.prompt,
		PromptTokens:   plan.promptTokens + plan.visual.tokens,
		ResponseSchema: inference.ReviewResponseFormat(),
		Images:         plan.visual.images,
		VisualKnown:    plan.visual.known,
	})
}

// post is everything after the gateway answers: the outcome branches, the
// comments, the rows and the summary. It runs on the review worker, so every
// Bitbucket call here is still one at a time.
func (r *Reviewer) post(ctx context.Context, plan *reviewPlan) {
	defer r.logHTTPTotals(ctx, plan.prTag, plan.counter)
	result := plan.result
	// Since the call: the wait to get back onto the review worker.
	plan.sw.lap(stageToPost)

	key, prTag, upsert := plan.key, plan.prTag, plan.upsert
	project, repo, sourceCommit := plan.project, plan.repo, plan.sourceCommit
	started, existing := plan.started, plan.existing

	// One cost record per call, and only when the gateway reported one: an
	// answer, or a stream cut after its usage, which was billed. The too-large
	// branch decides before any request, and a call that never got that far
	// would otherwise log an absent cost for a call nobody answered.
	if result.Cost.Answered {
		r.logCost(ctx, prTag, result.Cost)
	}

	// Given up during inference: posting would comment on a repo the team
	// left and re-create the PR row its removal purged. The call was paid
	// for, so the attempt carries its cost.
	if r.owns != nil && !r.owns(project, repo) {
		r.log.InfoContext(ctx, prTag+": repo no longer claimed by the team, not posting")
		r.recordAttempt(ctx, store.Attempt{
			Key: key, TeamSlug: r.TeamSlug, Kind: runKind(plan.mention),
			Outcome: "skipped", Reason: string(SkipRepoNotClaimed),
			CostNanoUSD: result.Cost.NanoUSD,
			Call:        r.llmCall(result.Cost),
		})
		return
	}

	// 17-19. Terminal branches. Each preserves the prior commit, posts a
	// notice and writes no run row.
	//
	// 19b. OutcomeError is NOT one of them: a non-overflow API error is only
	// logged. No upsert, no notice, no run row.
	if result.Outcome != inference.OutcomeOK {
		attemptID := r.handleNonOK(ctx, plan.lang, result, key, runKind(plan.mention), prTag, sourceCommit, upsert)
		r.finishDetail(ctx, plan, attemptID, stageRecordSum)
		return
	}

	// 20. Drop findings whose evidence or line the prompt's diffs do not
	// bear out, then dedup against what earlier runs already posted.
	v := r.validateFindings(ctx, prTag, result.Review.Findings, plan.anchors)
	// Fixed claims first: a finding this run retires is no reason to drop a
	// new one on its line as a repeat. A half fix then posts what is left
	// and resolves the old thread; a response that re-reports the very bug
	// it claims fixed posts it again, so the bug stays flagged either way.
	fixed := r.validateFixed(ctx, plan, result.Review.Fixed)
	// unfixed holds back repeats, settled ones included; the summary lists
	// and counts only what is still open on the PR (prepare's closeThreads).
	unfixed := withoutFixed(existing, fixed)
	earlierOpen := asFindings(openOnly(unfixed))
	for _, d := range v.Dropped {
		plan.detail.Decisions = append(plan.detail.Decisions, decision(d.Finding, "dropped", string(d.Reason)))
	}
	// Bugs in code the PR did not change: validated like findings, minus
	// the in-diff rule, and only ever shown on the run page.
	pre, preDropped := inference.ValidatePreExisting(result.Review.PreExisting, plan.anchors, plan.prAdded)
	if plan.preUnknown && len(pre) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d pre-existing bug(s) dropped: the PR's history could not be read", prTag, len(pre)))
		pre = nil
	}
	for _, d := range preDropped {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: pre-existing bug on %s:%d dropped: %s",
			prTag, d.Bug.File, d.Bug.Line, d.Reason))
	}
	for _, b := range pre {
		plan.detail.PreExisting = append(plan.detail.PreExisting, store.PreExistingBug{
			File: b.File, Line: b.Line, Headline: b.Headline, Explanation: b.Explanation, Evidence: b.Evidence})
	}
	if len(pre) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d pre-existing bug(s) recorded for the run page, not posted", prTag, len(pre)))
	}
	// Where to read first: summary only, never a finding, never the
	// verdict. A place the diff does not show is dropped with its reason.
	focus, focusDropped := inference.ValidateFocus(result.Review.Focus, plan.anchors, plan.history)
	for _, d := range focusDropped {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: review focus on %s:%d dropped: %s", prTag, d.Item.File, d.Item.Line, d.Reason))
	}
	// The model's blocking claims, checked by code; capTasks settles the
	// cap and the switches on what is no repeat, before the sort, so only a
	// real task sorts first. Then a plain finding on a task resolved for its
	// rule's sake is a repeat after all (holdBackRetired): only the task
	// may come back.
	deduped, repeats, retired := dedupe(decideBlocking(v.Kept, plan.blockingRules), unfixed, plan.anchors)
	capTasks(deduped, r.tasksOn())
	deduped, held := holdBackRetired(deduped, retired)
	repeats = append(repeats, held...)
	for _, f := range repeats {
		plan.detail.Decisions = append(plan.detail.Decisions, decision(f, "repeat", ""))
		r.log.InfoContext(ctx, fmt.Sprintf("%s: finding on %s:%d (%s) already posted by an earlier run, not posted again",
			prTag, f.File, f.Line, f.Severity))
	}
	r.logBlockingNotPosted(ctx, prTag, repeats, "a repeat of an earlier finding")
	// 21. Sort by severity and cap.
	findings, truncated := sortAndLimit(deduped, r.cfg.MaxComments)
	if truncated {
		// Sorting with no cap gives the same order, so the tail is the cut.
		all, _ := sortAndLimit(deduped, len(deduped))
		for _, f := range all[len(findings):] {
			plan.detail.Decisions = append(plan.detail.Decisions, decision(f, "cut", ""))
			r.log.InfoContext(ctx, fmt.Sprintf("%s: finding on %s:%d (%s) over the %d-comment cap, not posted",
				prTag, f.File, f.Line, f.Severity, r.cfg.MaxComments))
		}
		r.logBlockingNotPosted(ctx, prTag, all[len(findings):], "over the comment cap")
	}
	// The verdict follows the PR summary's findings, both halves, once. The
	// new ones uncapped: a finding over the comment cap is still a finding
	// (max_comments 0 posts none and must not approve over it). This run
	// only: a cut or refused one is never stored (a row needs a comment on
	// the PR), so a later run counts it only if the model reports it again.
	dropped := make([]inference.ReviewFinding, 0, len(v.Dropped))
	for _, d := range v.Dropped {
		dropped = append(dropped, d.Finding)
	}
	droppedClaims := decideBlocking(dropped, plan.blockingRules)
	capTasks(droppedClaims, r.tasksOn()) // logged as a kept claim would be
	r.logBlockingNotPosted(ctx, prTag, droppedClaims, "dropped by validation")
	reviewSummary := inference.SettleVerdict(result.Review.Summary, earlierOpen, deduped, dropped,
		!plan.existingUnknown, plan.lang)
	plan.detail.Verdict = &store.VerdictChange{Model: result.Review.Summary.VerdictDecision, Posted: reviewSummary.VerdictDecision}
	if was := result.Review.Summary.VerdictDecision; was != reviewSummary.VerdictDecision {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: verdict %s -> %s to match the PR summary's list: %d earlier open, %d new, %d dropped",
			prTag, was, reviewSummary.VerdictDecision, len(earlierOpen), len(deduped), len(v.Dropped)))
	}

	plan.sw.lap(stageValidate)

	// 22. The pointer finally advances.
	prReviewID := r.upsert(ctx, upsert, sourceCommit)

	// 23. Post the inline comments. postedIDs is parallel to findings, with
	// a zero where the post failed, so the count of successes is separate
	// from the slice length.
	labels := render.CriterionLabels(plan.ticket, plan.parentTicket)
	postedIDs, postedCount, failed := r.postInlineComments(ctx, plan.lang, labels, project, repo, plan.prID, findings)
	r.logBlocking(ctx, prTag, findings, postedIDs)
	// The summary counts and marks only tasks known to be on the PR: a
	// failed post may or may not have left one (postInlineComments), so it
	// is listed as a plain finding there. Its outcome stays as logged.
	summaryFindings := slices.Clone(findings)
	for i, id := range postedIDs {
		if id == 0 && summaryFindings[i].IsTask() {
			summaryFindings[i].Blocking = ""
		}
		if id == 0 {
			plan.detail.Decisions = append(plan.detail.Decisions, decision(findings[i], "failed", ""))
		}
	}
	plan.detail.Moved = postedMoves(findings, postedIDs)
	plan.sw.lap(stageComments)

	elapsed := time.Since(started)

	// 24. Run row, then cost, then the summary.
	//
	// The run row must exist before any finding row: store.Finding.RunID is
	// required and comes from InsertRun, so posting and inserting findings
	// cannot be interleaved ahead of it. The comments are already on the PR
	// by this point, so a crash before InsertRun leaves them there with no
	// finding row.
	tasks := postedTasks(findings, postedIDs)
	runID := r.recordRun(ctx, prReviewID, result, sourceCommit, plan, elapsed, postedCount-tasks, tasks)
	plan.runID = runID
	r.recordFindings(ctx, prReviewID, runID, findings, postedIDs)
	r.retireFixed(ctx, plan, fixed, runID)

	// The attempt row carries the run's id, so the feed can show this run's
	// findings and cost without the dashboard having to guess which run a
	// successful attempt produced. Written after InsertRun for that reason;
	// a zero runID (the insert failed) still records that the attempt
	// succeeded, with the figures missing rather than wrong.
	attempt := store.Attempt{
		Key: key, TeamSlug: r.TeamSlug, Kind: runKind(plan.mention),
		Outcome: inference.OutcomeOK.String(),
		Call:    r.llmCall(result.Cost),
	}
	ms := elapsed.Milliseconds()
	attempt.ElapsedMS = &ms
	if runID != 0 {
		attempt.RunID = &runID
	}
	attemptID := r.recordAttempt(ctx, attempt)

	runCost, cumulativeCost := r.resolveCost(ctx, key, result)
	compliance := r.confirmCompliance(ctx, plan, result.Review.ComplianceRequirements)

	summary := render.Summary(render.SummaryInput{
		Lang:                       plan.lang,
		Findings:                   summaryFindings,
		Cut:                        len(deduped) - len(findings),
		EarlierOpen:                earlierOpen,
		EarlierUnknown:             plan.existingUnknown,
		FixedEarlier:               len(fixed),
		Dropped:                    len(v.Dropped),
		Summary:                    reviewSummary,
		Focus:                      focus,
		AgentsMDFound:              plan.agentsMDFound,
		AgentsMDNested:             renderNested(plan.detail.AgentsMD.Nested),
		AgentsMDLeftOut:            renderLeft(plan.detail.AgentsMD.LeftOut),
		AgentsMDIncomplete:         plan.detail.AgentsMD.Incomplete,
		SkippedFiles:               plan.oversized,
		OverBudgetFiles:            plan.overBudget,
		ContentSkippedFiles:        plan.contentSkipped,
		AgentsMDTokens:             plan.detail.AgentsMD.Tokens,
		Ticket:                     plan.ticket,
		ParentTicket:               plan.parentTicket,
		ComplianceRequirements:     compliance,
		ComplianceExtractionFailed: result.Review.ComplianceRequirements == nil,
		ComplianceHistoryUnknown:   plan.preUnknown,
		TicketComplianceCheck:      r.cfg.TicketComplianceCheck,
		JiraEnabled:                r.jiraEnabled(),
		JiraUnavailable:            plan.jiraDown,
		ElapsedSeconds:             elapsed.Seconds(),
		ElapsedPresent:             true,
		ReviewedCommit:             sourceCommit,
		IncrementalFrom:            plan.incrementalFrom,
		TotalFiles:                 plan.totalFiles,
		FilesCountsSet:             true,
		DiffAdded:                  plan.diffAdded,
		DiffRemoved:                plan.diffRemoved,
		ModelLabel:                 r.llm.Label(),
		PRURL:                      r.prURL(ctx, prReviewID),
		RunCostUSD:                 runCost,
		CumulativeCostUSD:          cumulativeCost,
		MaxPRCostUSD:               r.cfg.MaxPRCostUSD,
		AgentsMDWarnTokens:         r.cfg.AgentsMDWarnTokens,
		// After the upsert above: a first review's lines need its PR row.
		Visual:       r.visualLines(ctx, plan, result.Review.Visual),
		VisualUnread: plan.visual.unread,
	})

	// A completed run at or over the cap keeps the banner on top, so the
	// over-budget state stays loud. The summary is freshly built each run, so
	// no strip is needed.
	if cumulativeCost != nil && *cumulativeCost >= r.cfg.MaxPRCostUSD {
		summary = render.CostLimitBanner(plan.lang, *cumulativeCost, r.cfg.MaxPRCostUSD, false) +
			"\n\n" + summary
	}

	r.postOrUpdateSummary(ctx, project, repo, plan.prID, prReviewID, summary)
	r.finishDetail(ctx, plan, attemptID, stageRecordSum)

	// A conflict noergler could resolve is offered, in a comment of its
	// own; after the run's record, so its fetch is no stage of the review.
	// At or over the cost cap nothing is checked and an open offer is
	// withdrawn (user, 2026-11-14: stop there; the model one would be
	// refused).
	if prReviewID != 0 {
		kind := offerNone
		if cumulativeCost == nil || *cumulativeCost < r.cfg.MaxPRCostUSD {
			kind = r.conflictOffer(ctx, key, prTag, plan.offer)
		}
		r.noteConflictOffer(ctx, key, prReviewID, prTag, plan.lang, plan.offer, kind)
	}

	parts := []string{
		fmt.Sprintf("Review of %s completed in %.1fs", prTag, elapsed.Seconds()),
		render.Plural(len(findings), "issue"),
		render.Plural(postedCount, "comment") + " posted",
	}
	if failed > 0 {
		parts = append(parts, fmt.Sprintf("%d failed", failed))
	}
	// Token accounting on the completion line. An endpoint that reports no
	// usage leaves these zero, which is worth seeing as such rather than
	// omitting.
	c := result.Cost
	parts = append(parts, fmt.Sprintf("%d in (%d cached) + %d out = %d tokens",
		c.PromptTokens, c.CachedTokens, c.CompletionTokens,
		c.PromptTokens+c.CompletionTokens))
	r.log.InfoContext(ctx, strings.Join(parts, " - "))
}

// checkSkipState reports why the review may not proceed, SkipNone when it
// may.
//
// An ignored PR stops here: SkipStoppedPR when someone stopped it, else
// SkipIgnoredPR. When a summary comment is tracked, a 404 on refetch means
// the user deleted it, which is the backstop for repos not subscribed to
// pr:comment:deleted. Any other status, or a network failure, falls THROUGH
// to the review: never silence a live PR on a transient fault.
func (r *Reviewer) checkSkipState(ctx context.Context, key store.PRKey, prTag string) SkipReason {
	state := safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
		return r.store.GetSkipState(ctx, key)
	})
	if state == nil {
		return SkipNone
	}
	if state.Stopped() {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: PR stopped by %s - skipping", prTag, state.StoppedBy))
		return SkipStoppedPR
	}
	if state.IgnoredAt != nil {
		r.log.InfoContext(ctx, prTag+": PR ignored (summary comment removed) - skipping")
		return SkipIgnoredPR
	}
	if state.Summary == nil {
		return SkipNone
	}

	_, err := r.bitbucket.FetchPRComment(ctx, key.Project, key.Repo, key.PRID, state.Summary.ID)
	if err == nil {
		return SkipNone
	}
	if status := bitbucket.Status(err); status == http.StatusNotFound {
		safeDBErr(ctx, r.log, "MarkIgnored", func() error { return r.store.MarkIgnored(ctx, key) })
		r.log.InfoContext(ctx, fmt.Sprintf("%s: summary comment %d was deleted - ignoring PR from now on",
			prTag, state.Summary.ID))
		return SkipIgnoredPR
	} else if status != 0 {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: could not verify summary comment %d (HTTP %d) - proceeding",
			prTag, state.Summary.ID, status))
		return SkipNone
	}
	r.log.WarnContext(ctx, prTag+": summary-comment check failed - proceeding")
	return SkipNone
}

// resolveDiff decides between an incremental and a full review and fetches
// the diff. The review stops on a why other than SkipNone or on err.
//
// why names which of the three decisions it was (nothing changed, nothing to
// review, too big); err is the fault (the diff would not fetch). A single
// bare false told the caller only that it had to return, so the two
// commonest skips in the whole pipeline were invisible to anything but the
// log, and a zero why could not tell a fault from any other stop.
func (r *Reviewer) resolveDiff(ctx context.Context, payload *webhook.Payload, key store.PRKey, prTag, sourceCommit, lastReviewed string, upsert store.PRUpsert, l lang.Lang) (rawDiff string, cumulative cumulativeContext, incrementalFrom string, why SkipReason, err error) {
	project, repo, prID := key.Project, key.Repo, key.PRID
	isIncremental := false

	if payload.EventKey == webhook.EventFromRefUpdated && lastReviewed != "" && sourceCommit != "" {
		// 9. Same SHA: nothing changed (a retrigger or a no-op force push).
		if sourceCommit == lastReviewed {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: HEAD unchanged since last review (%s), skipping",
				prTag, render.ShortSHA(sourceCommit, 10)))
			return "", cumulativeContext{}, "", SkipHeadUnchanged, nil
		}
		incDiff, err := r.bitbucket.FetchCommitDiff(ctx, project, repo, prID, lastReviewed, sourceCommit)
		switch {
		case errors.Is(err, bitbucket.ErrIncrementalDiffUnavailable):
			// Expected control flow: a rebase, force push or merge in the range.
			// INFO, no traceback, and fall through to the full review. We
			// must NOT skip just because the optimization did not apply.
			r.log.InfoContext(ctx, fmt.Sprintf("%s: incremental diff unavailable (%v) - running full review", prTag, err))
		case errors.As(err, new(*bitbucket.ContentTooLarge)):
			// A decision, not a fault. The full diff is usually larger and
			// takes the too-large path below, but not always: a push that
			// deletes a big file the PR added nets out in the full diff.
			r.log.InfoContext(ctx, fmt.Sprintf("%s: incremental diff too large (%v) - running full review", prTag, err))
		case err != nil:
			r.log.WarnContext(ctx, fmt.Sprintf("%s: incremental diff failed unexpectedly, falling back to full review: %v", prTag, err))
		case strings.TrimSpace(incDiff) != "":
			rawDiff = incDiff
			isIncremental = true
			incrementalFrom = lastReviewed
			r.log.InfoContext(ctx, fmt.Sprintf("%s: incremental review %s..%s",
				prTag, render.ShortSHA(lastReviewed, 10), render.ShortSHA(sourceCommit, 10)))
		default:
			// HEAD moved but the range diff is empty: a rebase to an identical
			// tree, or a Bitbucket edge case. Never silently skip when the
			// SHA actually changed.
			r.log.WarnContext(ctx, fmt.Sprintf(
				"%s: HEAD moved %s -> %s but the incremental diff is empty - falling back to full review to avoid missing changes",
				prTag, render.ShortSHA(lastReviewed, 10), render.ShortSHA(sourceCommit, 10)))
		}
	}

	// A fallback above has said why already. Without this line a push
	// reviewed in full for want of a pointer (a first run, or a failed one
	// that kept the prior commit) looks like an incremental run that cost
	// as much as the whole PR.
	switch {
	case payload.EventKey != webhook.EventFromRefUpdated:
		r.log.InfoContext(ctx, fmt.Sprintf("%s: full review (event %s is not a push)", prTag, payload.EventKey))
	case lastReviewed == "":
		r.log.InfoContext(ctx, prTag+": full review (no prior reviewed commit)")
	case sourceCommit == "":
		r.log.InfoContext(ctx, prTag+": full review (payload carries no source commit)")
	}

	if !isIncremental {
		full, err := r.bitbucket.FetchPRDiff(ctx, project, repo, prID, 0)
		var tooLarge *bitbucket.ContentTooLarge
		if errors.As(err, &tooLarge) {
			// 10. Nothing in this push was reviewed, so the next push must
			// not go incremental from it.
			r.logDiffTooLarge(ctx, prTag, project, repo, prID, err, tooLarge)
			prReviewID, _ := r.upsertKeepingPrior(ctx, upsert, key)
			r.postOrUpdateSummary(ctx, project, repo, prID, prReviewID, render.DiffTooLargeSummary(l, tooLarge.Limit))
			return "", cumulativeContext{}, "", SkipDiffTooLarge, nil
		}
		if err != nil {
			r.log.ErrorContext(ctx, fmt.Sprintf("%s: failed to fetch PR diff: %v", prTag, err))
			return "", cumulativeContext{}, "", SkipNone, err
		}
		// 11. An empty diff is nothing to review.
		if strings.TrimSpace(full) == "" {
			r.log.InfoContext(ctx, prTag+" has empty diff, skipping")
			return "", cumulativeContext{}, "", SkipEmptyDiff, nil
		}
		return full, cumulativeContext{}, "", SkipNone, nil
	}

	// 12. The cumulative PR diff is cross-file context for an incremental
	// review, so the model can check invariants split across commits. Best
	// effort: a failure here must not block the review.
	cumulative = r.fetchCumulativeDiff(ctx, key, prTag, rawDiff)
	return rawDiff, cumulative, incrementalFrom, SkipNone, nil
}

// handleNonOK posts the notice for a non-ok outcome.
//
// timed_out, unparseable and too_large each preserve the prior commit, post
// their notice and write no RUN row. OutcomeError posts NOTHING, writes no
// run row and does not upsert: it is only logged (TestTerminalOutcomes).
//
// All four write a review_attempts row, which is a separate table precisely
// so those pins stay true: the dashboard needs to show a failure, and
// review_runs must keep meaning "a review that produced a result".
//
// Returns the attempt id, for the run's detail record.
func (r *Reviewer) handleNonOK(ctx context.Context, l lang.Lang, result inference.ReviewResult, key store.PRKey, kind store.RunKind, prTag, sourceCommit string, upsert store.PRUpsert) int64 {
	short := shortOrUnknown(sourceCommit, lang.EN) // logs stay English
	project, repo, prID := key.Project, key.Repo, key.PRID

	if result.Outcome == inference.OutcomeError {
		r.log.ErrorContext(ctx, fmt.Sprintf("Review of %s failed: %v", prTag, result.Err))
		// A stream cut after its usage was billed: its cost rides the
		// attempt like any other billed call without a run row.
		return r.recordAttempt(ctx, store.Attempt{
			Key: key, TeamSlug: r.TeamSlug, Kind: kind,
			Outcome:     result.Outcome.String(),
			CostNanoUSD: result.Cost.NanoUSD,
			Call:        r.llmCall(result.Cost),
		})
	}

	// Read the prior commit BEFORE the upsert, which would otherwise
	// overwrite it with this failed commit.
	prReviewID, prior := r.upsertKeepingPrior(ctx, upsert, key)

	timeout := r.llm.TimeoutOf(result.Err)
	switch result.Outcome {
	case inference.OutcomeTimedOut:
		r.log.ErrorContext(ctx, fmt.Sprintf("Review of %s aborted - no response within %.0fs (commit %s)",
			prTag, timeout.Seconds(), short))
	case inference.OutcomeUnparseable:
		r.log.ErrorContext(ctx, fmt.Sprintf(
			"Review of %s aborted - model returned an unparseable/refused response (commit %s)", prTag, short))
	case inference.OutcomeTooLarge:
		r.log.ErrorContext(ctx, fmt.Sprintf(
			"Review of %s skipped - PR too large for the model's context window (commit %s)", prTag, short))
	}
	r.failureNoticeFor(ctx, result.Outcome, timeout, l, project, repo, prID, prReviewID, sourceCommit, prior)

	// No run row, so a billed call's cost rides here, where PRCost reads it.
	// The call itself is kept too: an unparseable answer is exactly the run
	// someone opens the detail page for.
	return r.recordAttempt(ctx, store.Attempt{
		Key: key, TeamSlug: r.TeamSlug, Kind: kind,
		Outcome:     result.Outcome.String(),
		CostNanoUSD: result.Cost.NanoUSD,
		Call:        r.llmCall(result.Cost),
	})
}

// jiraUnavailableNote tells the model the PR names a ticket Jira did not
// serve, so neither the review nor a Q&A answer reads it as "no ticket". It
// sits in the untrusted ticket block, so it is a fact, never an instruction.
const jiraUnavailableNote = "Jira was unavailable: the ticket this PR names (or its parent) could not be read."

// ticketContextFor is the {ticket_context} value for both prompts.
func ticketContextFor(ticket, parent *jira.Ticket, jiraDown bool) string {
	ctx := renderTicketContext(ticket, parent)
	if jiraDown {
		ctx = strings.TrimSpace(ctx + "\n\n" + jiraUnavailableNote)
	}
	return ctx
}

// jiraLookupTimeout bounds one review's ticket lookup, child and parent
// together. It runs on the team's Bitbucket worker: a blackholed Jira must
// not hold every review for the client's 30 s per call. No retry.
const jiraLookupTimeout = 10 * time.Second

// fetchTicket resolves the Jira ticket for a PR, or nil when there is none.
// jiraDown reports a key whose lookup failed: Jira unreachable, rejecting the
// token, timing out, or the team's own token unusable. The review goes on
// without the ticket check (or with the child alone when only its parent
// failed) and the summary says so; no retry, the next review asks Jira again.
func (r *Reviewer) fetchTicket(ctx context.Context, branch, title, prTag string) (ticket, parent *jira.Ticket, jiraDown bool) {
	if !r.jiraEnabled() {
		return nil, nil, false
	}
	id := extractTicketID(branch, title)
	if id == "" {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: no Jira key in branch %q or title, reviewing without ticket context", prTag, branch))
		return nil, nil, false
	}
	lctx, cancel := context.WithTimeout(ctx, jiraLookupTimeout)
	defer cancel()
	ticket, parent, err := r.jira.FetchTicketWithParent(lctx, id)
	switch {
	case err != nil && ctx.Err() != nil:
		// The review was stopped mid-call, not Jira: no outage to report.
		return nil, nil, false
	case err != nil && ticket != nil:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: linked Jira ticket %s, its parent failed, reviewing without it: %v", prTag, id, err))
		return ticket, nil, true
	case err != nil:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: Jira lookup for %s failed, reviewing without the ticket check: %v", prTag, id, err))
		return nil, nil, true
	}
	if ticket != nil {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: linked Jira ticket %s", prTag, id))
	} else {
		// Not an error: the key came off a branch name and may be noise.
		r.log.InfoContext(ctx, fmt.Sprintf("%s: Jira ticket %s not readable, reviewing without ticket context", prTag, id))
	}
	return ticket, parent, false
}

// trimPreviouslyPosted caps the previously-posted block by count and then by
// rendered token size.
//
// The tail is the most recent (the store returns oldest first). Dropping the
// oldest in 25% chunks avoids re-rendering once per item.
//
// withIDs says whether the ids are listed. They are dropped before
// measuring: the block is trimmed as it is sent, never to a size it is not.
func (r *Reviewer) trimPreviouslyPosted(ctx context.Context, prTag string, existing []store.Finding, budget int, withIDs bool) []inference.PostedFinding {
	posted := make([]inference.PostedFinding, 0, len(existing))
	for _, f := range existing {
		p := asPosted(f)
		if !withIDs {
			p.ID = 0
		}
		posted = append(posted, p)
	}
	// Settled ones first, so the trims below drop them before any open one:
	// an open finding carries the id a `fixed` claim needs and the verdict
	// counts it; a closed one only keeps the model from raising it again.
	slices.SortStableFunc(posted, func(a, b inference.PostedFinding) int {
		if a.Closed == b.Closed {
			return 0
		}
		if a.Closed {
			return -1
		}
		return 1
	})
	if len(posted) > maxPreviouslyPostedFindings {
		posted = posted[len(posted)-maxPreviouslyPostedFindings:]
	}

	limit := inference.PreviouslyPostedBudget(budget)
	for len(posted) > 0 && r.tokens.Count(inference.RenderPreviouslyPostedFindings(posted)) > limit {
		drop := max(1, len(posted)/4)
		posted = posted[drop:]
	}
	// The model is told not to re-raise what is listed; a finding trimmed
	// here can come back as a duplicate comment.
	if len(posted) < len(existing) {
		r.log.InfoContext(ctx, fmt.Sprintf(
			"%s: previously posted findings trimmed: %d of %d shown to the model, closed then oldest dropped (caps: %d findings, %d tokens)",
			prTag, len(posted), len(existing), maxPreviouslyPostedFindings, limit))
	}
	return posted
}

// logPromptComposition attributes the prompt's input tokens to its parts.
//
// The run's cost follows the prompt, and the prompt hardly follows the diff:
// a one-line push carries the same template, AGENTS.md and whole file. The
// rest is whatever the named parts do not cover: template, system message,
// schema and the cross-file notes.
func (r *Reviewer) logPromptComposition(ctx context.Context, prTag string, a inference.AssembledPrompt, incremental bool) {
	mode := "full"
	if incremental {
		mode = "incremental"
	}
	b := a.Breakdown
	rest := a.PromptTokens - b.RepoInstructions - b.Files - b.CumulativeDiff - b.PreviouslyPosted - b.PreviousOverview - b.Ticket - b.PRDescription
	r.log.InfoContext(ctx, fmt.Sprintf(
		// A sum, not a list: the parts add up to the total and share its
		// unit. Listed after a dash they read as bytes or KB.
		"%s: prompt %d tokens (%s review) = files %d + cumulative PR diff %d + AGENTS.md %d + previously posted %d + previous overview %d + ticket %d + PR description %d + template and rest %d",
		prTag, a.PromptTokens, mode, b.Files, b.CumulativeDiff, b.RepoInstructions, b.PreviouslyPosted, b.PreviousOverview, b.Ticket, b.PRDescription, rest))
}

// previousOverview is the overview the PR's last completed review wrote, for
// an incremental one to build on (user, 2026-10-28), or "" if none. An
// unreadable one reviews without it: the overview then starts over from
// what this prompt shows, as before it was carried.
func (r *Reviewer) previousOverview(ctx context.Context, key store.PRKey, prTag string) string {
	s := safeDB(ctx, r.log, "LastSummary", func() (*store.PRSummary, error) {
		return r.store.LastSummary(ctx, key)
	})
	if s == nil || strings.TrimSpace(s.Overview) == "" {
		return ""
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: previous PR overview carried into the prompt (%d chars, cap %d)",
		prTag, len([]rune(strings.TrimSpace(s.Overview))), inference.PreviousOverviewCap))
	return s.Overview
}

// validateFindings applies inference.ValidateFindings and logs every drop and
// every moved line, one line each, plus a count line when anything changed.
// A drop is Info, not Warn: it is the check working, not a fault.
func (r *Reviewer) validateFindings(ctx context.Context, prTag string, findings []inference.ReviewFinding, anchors diff.AnchorIndex) inference.Validation {
	v := inference.ValidateFindings(findings, anchors)
	for _, d := range v.Dropped {
		headline := ""
		if d.Finding.Headline != nil {
			headline = *d.Finding.Headline
		}
		r.log.InfoContext(ctx, fmt.Sprintf("%s: dropped finding on %s:%d (%s): %s",
			prTag, d.Finding.File, d.Finding.Line, d.Reason, headline))
	}
	for _, m := range v.Reanchored {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: moved finding on %s from line %d to %d, where its evidence is",
			prTag, m.File, m.From, m.To))
	}
	if len(v.Dropped) > 0 || len(v.Reanchored) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: findings validated: %d kept (%d moved), %d dropped",
			prTag, len(v.Kept), len(v.Reanchored), len(v.Dropped)))
	}
	return v
}

// dedupe drops findings an earlier run already posted, keyed on file, line
// and severity. repeats are the dropped ones, for the log. A stored path is
// resolved like a new one: older rows kept the model's a/ or b/ prefix.
//
// One exception: a task noergler resolved because its rule left AGENTS.md
// holds back nothing here; its key goes to retired, and holdBackRetired
// holds back its plain repeat once capTasks has settled which findings
// are tasks (blocking.go). A reworded rule resolves the old task; the
// violation it still sees under the new wording must block again.
func dedupe(findings []inference.ReviewFinding, existing []store.Finding, idx diff.AnchorIndex) (out, repeats []inference.ReviewFinding, retired map[dedupeKey]bool) {
	seen := make(map[dedupeKey]bool, len(existing))
	retired = map[dedupeKey]bool{}
	for _, f := range existing {
		k := dedupeKey{inference.ShownPath(f.FilePath, idx), f.LineNumber, f.Severity}
		if blockingOf(f) == inference.BlockingRuleRemoved {
			retired[k] = true
			continue
		}
		seen[k] = true
	}
	out = make([]inference.ReviewFinding, 0, len(findings))
	for _, f := range findings {
		if seen[keyOf(f)] {
			repeats = append(repeats, f)
			continue
		}
		out = append(out, f)
	}
	return out, repeats, retired
}

// holdBackRetired is dedupe's second half, after capTasks: a finding on a
// retired key is a repeat unless it is a task itself.
func holdBackRetired(findings []inference.ReviewFinding, retired map[dedupeKey]bool) (out, repeats []inference.ReviewFinding) {
	out = make([]inference.ReviewFinding, 0, len(findings))
	for _, f := range findings {
		if retired[keyOf(f)] && !f.IsTask() {
			repeats = append(repeats, f)
			continue
		}
		out = append(out, f)
	}
	return out, repeats
}

// dedupeKey is what makes a finding a repeat: file, line and severity.
type dedupeKey struct {
	file     string
	line     int
	severity string
}

func keyOf(f inference.ReviewFinding) dedupeKey { return dedupeKey{f.File, f.Line, f.Severity} }

// sortAndLimit orders findings by severity and caps them.
//
// SortStableFunc, not SortFunc: findings of equal severity must keep the
// model's order or the numbered Issues list shuffles between runs of the
// same review (TestSortAndLimit).
func sortAndLimit(findings []inference.ReviewFinding, maxComments int) ([]inference.ReviewFinding, bool) {
	sorted := slices.Clone(findings)
	// A blocking task first within its severity: the cap must not cut it
	// for a plain finding (blocking.go).
	slices.SortStableFunc(sorted, inference.CompareFindings)
	truncated := len(sorted) > maxComments
	if truncated {
		sorted = sorted[:maxComments]
	}
	return sorted, truncated
}

// postedTasks counts the findings that reached the PR as a Bitbucket task.
// A post is a finding or a task, never both (user, 2026-11-17): the run's
// findings_posted leaves these out. One Bitbucket refused went up as a
// comment (BlockingTaskFailed) and counts as a finding.
func postedTasks(findings []inference.ReviewFinding, ids []int) int {
	n := 0
	for i, f := range findings {
		if ids[i] != 0 && f.IsTask() {
			n++
		}
	}
	return n
}

// postInlineComments posts one comment per finding.
//
// ids is parallel to findings and holds the Bitbucket comment id, or 0 where
// the post failed. posted and failed count the two outcomes: ids is always
// len(findings) long, so its length is not the success count.
//
// A failure is counted and logged but never stored: a finding row exists
// only for a comment that is actually on the PR
// (TestFailedInlineCommentIsNotStored).
//
// A blocking task goes up as a task; one Bitbucket refuses (a 4xx) is
// posted as a plain comment instead (BlockingTaskFailed, set in findings),
// so the finding still reaches the PR. Any other failure fails it like a
// comment: the task may exist with its answer lost, and a comment beside
// it would say there is none.
func (r *Reviewer) postInlineComments(ctx context.Context, l lang.Lang, labels []string, project, repo string, prID int, findings []inference.ReviewFinding) (ids []int, posted, failed int) {
	ids = make([]int, len(findings))
	for i, f := range findings {
		var id int
		var err error
		if f.IsTask() {
			id, err = r.bitbucket.PostInlineTask(ctx, project, repo, prID, f.File, f.Line, render.InlineComment(f, l, labels...))
			if status := bitbucket.Status(err); status >= 400 && status < 500 {
				r.log.WarnContext(ctx, fmt.Sprintf("Failed to post blocking task on %s:%d, posting it as a comment: %v", f.File, f.Line, err))
				findings[i].Blocking = inference.BlockingTaskFailed
				f = findings[i]
			}
		}
		if !f.IsTask() {
			id, err = r.bitbucket.PostInlineComment(ctx, project, repo, prID, f.File, f.Line, render.InlineComment(f, l, labels...))
		}
		if err != nil {
			failed++
			r.log.ErrorContext(ctx, fmt.Sprintf("Failed to post inline comment on %s:%d: %v", f.File, f.Line, err))
			continue
		}
		ids[i] = id
		posted++
	}
	return ids, posted, failed
}

// recordRun writes the run row and returns its id.
func (r *Reviewer) recordRun(ctx context.Context, prReviewID int64, result inference.ReviewResult,
	sourceCommit string, plan *reviewPlan, elapsed time.Duration, findingsPosted, tasksPosted int) int64 {
	if prReviewID == 0 {
		return 0
	}
	kind := runKind(plan.mention)
	incrementalFrom := plan.incrementalFrom
	var from *string
	if incrementalFrom != "" {
		from = &incrementalFrom
	}
	// The next review's overview builds on this one; an empty or sentinel
	// one keeps the stored overview rather than erasing it.
	var summary *store.PRSummary
	if o := strings.TrimSpace(result.Review.Summary.Overview); o != "" && !render.IsNoneSentinel(o) {
		summary = &store.PRSummary{Overview: o}
	}
	return safeDB(ctx, r.log, "InsertRun", func() (int64, error) {
		return r.store.InsertRun(ctx, store.Run{
			PullRequestID:    prReviewID,
			Kind:             kind,
			Incremental:      incrementalFrom != "",
			FromCommit:       from,
			ToCommit:         sourceCommit,
			ModelLabel:       r.llm.Label(),
			PromptTokens:     result.Cost.PromptTokens,
			CachedTokens:     result.Cost.CachedTokens,
			CompletionTokens: result.Cost.CompletionTokens,
			CostNanoUSD:      result.Cost.NanoUSD,
			ElapsedMS:        elapsed.Milliseconds(),
			FindingsPosted:   findingsPosted,
			TasksPosted:      tasksPosted,
			LinesAdded:       plan.diffAdded,
			LinesRemoved:     plan.diffRemoved,
			FilesChanged:     plan.totalFiles,
			ReviewedAdded:    plan.sentAdded,
			ReviewedRemoved:  plan.sentRemoved,
			Summary:          summary,
		})
	})
}

// recordFindings stores one row per posted finding.
func (r *Reviewer) recordFindings(ctx context.Context, prReviewID, runID int64, findings []inference.ReviewFinding, ids []int) {
	if prReviewID == 0 || runID == 0 {
		return
	}
	for i, f := range findings {
		if ids[i] == 0 {
			continue // the comment never made it onto the PR
		}
		commentID := ids[i]
		row := store.Finding{
			PullRequestID:      prReviewID,
			RunID:              runID,
			FilePath:           f.File,
			LineNumber:         f.Line,
			Severity:           f.Severity,
			Confidence:         f.Confidence,
			Headline:           f.Headline,
			CommentText:        f.Comment,
			Suggestion:         f.Suggestion,
			BitbucketCommentID: &commentID,
			Category:           f.Category,
		}
		if f.Blocking != "" {
			outcome := string(f.Blocking)
			row.BlockingOutcome = &outcome
		}
		if f.BlockingText != "" {
			text := f.BlockingText
			row.BlockingRule = &text
		}
		safeDBErr(ctx, r.log, "InsertFinding", func() error { return r.store.InsertFinding(ctx, row) })
	}
}

// resolveCost returns this run's cost and the PR total, both nil when the
// gateway did not price the run.
//
// The PR total is only read when this run is priced. Reading the aggregate
// unconditionally would show a total on an unpriced run and could trip the
// cost banner (TestUnpricedRunShowsNoCostLine).
func (r *Reviewer) resolveCost(ctx context.Context, key store.PRKey, result inference.ReviewResult) (run, cumulative *float64) {
	if !result.Cost.Priced() {
		return nil, nil
	}
	runUSD := float64(*result.Cost.NanoUSD) / nanoPerUSD
	run = &runUSD

	if total := safeDB(ctx, r.log, "PRCost", func() (*int64, error) {
		return r.store.PRCost(ctx, key)
	}); total != nil {
		totalUSD := float64(*total) / nanoPerUSD
		cumulative = &totalUSD
	}
	return run, cumulative
}

// upsert writes the PR row with the given reviewed commit and returns its id.
//
// An empty commit is written as NULL, not "": the skip paths pass the prior
// pointer back on purpose and a PR with no prior review has none.
func (r *Reviewer) upsert(ctx context.Context, u store.PRUpsert, lastReviewedCommit string) int64 {
	if lastReviewedCommit != "" {
		u.LastReviewedCommit = &lastReviewedCommit
	}
	return safeDB(ctx, r.log, "UpsertPullRequest", func() (int64, error) {
		return r.store.UpsertPullRequest(ctx, u)
	})
}

// upsertKeepingPrior writes the PR row for a push that was not reviewed, so
// the prior commit stands, and returns the row id and that commit. A failed
// read keeps the stored pointer: passing "" back would NULL it and make the
// next push a full, paid review.
func (r *Reviewer) upsertKeepingPrior(ctx context.Context, u store.PRUpsert, key store.PRKey) (int64, string) {
	prior, readOK := r.priorCommit(ctx, key)
	u.KeepPointer = !readOK
	return r.upsert(ctx, u, prior), prior
}

// priorCommit is the last successfully reviewed commit, or "". readOK is
// false only when the read failed.
func (r *Reviewer) priorCommit(ctx context.Context, key store.PRKey) (commit string, readOK bool) {
	readOK = true
	commit = safeDB(ctx, r.log, "GetLastReviewedCommit", func() (string, error) {
		c, ok, err := r.store.GetLastReviewedCommit(ctx, key)
		readOK = err == nil
		if !ok {
			return "", err
		}
		return c, err
	})
	return commit, readOK
}

// notFullContext is the "reviewed without full file context" list: a
// content-skipped file that the prompt carries as an entry of its own (the
// set compression kept, after it), without its body. A file the prompt does
// not carry was not reviewed at all, and a rename-only one has no body to
// miss. Read from what was sent, so both review paths agree.
func notFullContext(contentSkipped []string, sent []diff.FileReviewData) []string {
	entry := map[string]bool{}
	for _, f := range sent {
		if !f.ContentFetched && !diff.IsRenameOnly(f.Diff) {
			entry[f.Path] = true
		}
	}
	var out []string
	for _, p := range contentSkipped {
		if entry[p] {
			out = append(out, p)
		}
	}
	return out
}

func epochMSToTime(ms int64) *time.Time {
	if ms == 0 {
		return nil
	}
	t := time.UnixMilli(ms).UTC()
	return &t
}

// logHTTPTotals reports what one review spent upstream. The wording is
// `Review HTTP totals - bitbucket=N jira=N inference=N (per-method detail)`
// and is pinned by TestLogHTTPTotals.
//
// Nothing is logged when no request was made: the author and actor gates
// return before any HTTP, and an empty totals line for every skipped PR is
// noise. inference counts too: the client llmwire is built with wraps its
// transport in httpstats.Transport (see inference.countingClient), so a
// review that called the model reports it rather than a constant 0.
func (r *Reviewer) logHTTPTotals(ctx context.Context, prTag string, c *httpstats.Counter) {
	totals := c.Summarize()
	if len(totals) == 0 {
		return
	}
	// Any label beyond the three named ones still appears, in the detail.
	methods := c.Methods()
	detail := make([]string, 0, len(methods))
	for _, k := range slices.Sorted(maps.Keys(methods)) {
		detail = append(detail, fmt.Sprintf("%s=%d", k, methods[k]))
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: Review HTTP totals - bitbucket=%d jira=%d inference=%d (%s)",
		prTag, totals["bitbucket"], totals["jira"], totals["inference"],
		strings.Join(detail, " ")))
}

// logCost writes the per-call cost record. WARNING only when the gateway
// should have priced the call and did not, or priced it at zero after
// consuming tokens: an endpoint that is not a LiteLLM proxy never prices, and
// warning on every one of its calls would train the operator to ignore it.
func (r *Reviewer) logCost(ctx context.Context, prTag string, cost inference.CallCost) {
	line, warn := cost.LogLine()
	if warn {
		r.log.WarnContext(ctx, prTag+": "+line)
		return
	}
	r.log.InfoContext(ctx, prTag+": "+line)
}
