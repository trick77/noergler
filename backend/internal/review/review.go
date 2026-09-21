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
	"github.com/trick77/noergler/internal/logging"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// contextExpansionRatio decides when a PR counts as small: its files plus
// room to expand them still fit the budget.
const contextExpansionRatio = 1.5

// nanoPerUSD converts the DB's BIGINT nano-USD to the USD used at the edges.
const nanoPerUSD = 1_000_000_000.0

// ReviewPullRequest runs the review pipeline for one PR.
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
// ReviewPullRequest runs the whole review inline: prepare, infer, post.
// HandleMention and the tests use it; the queue uses the staged entry below.
func (r *Reviewer) ReviewPullRequest(ctx context.Context, payload *webhook.Payload, skipAuthorCheck bool) {
	plan, ctx, ok := r.prepare(ctx, payload, skipAuthorCheck)
	if !ok {
		return
	}
	r.post(ctx, plan, r.infer(ctx, plan))
}

// prepare is everything up to and including prompt assembly: the guards, the
// Bitbucket fetches, Jira, and the token budget. It returns the plan the
// inference and posting stages need, and the ctx carrying pr_tag and the
// httpstats scope.
//
// ok is false when the review is over: the caller must not continue. Every
// such exit has already logged its own HTTP totals.
func (r *Reviewer) prepare(ctx context.Context, payload *webhook.Payload, skipAuthorCheck bool) (*reviewPlan, context.Context, bool) {
	pr := payload.PullRequest
	project, repo := payload.ProjectRepo()
	if project == "" || repo == "" {
		r.log.ErrorContext(ctx, "Could not extract project/repo from webhook payload")
		return nil, ctx, false
	}

	prTag := fmt.Sprintf("%s/%s#%d", project, repo, pr.ID)
	kind := runKind(skipAuthorCheck)
	ctx = logging.With(ctx, "pr_tag", prTag, "repo", project+"/"+repo, "pr_id", pr.ID)

	// Per-review HTTP accounting. The bitbucket and jira transports already
	// record into the scope; without one opened here every count was dropped
	// and the totals line never existed. Deferred, so a review that skips or
	// fails still reports what it spent.
	// NOT deferred: with the inference call staged off the worker, a defer
	// here fires before the gateway call and before posting, so the totals
	// line would report inference=0 and none of the post stage's Bitbucket
	// calls. Each exit below logs its own; the full path logs in post.
	ctx, httpCounter := httpstats.WithScope(ctx)

	key := prKey(project, repo, pr.ID)
	author := pr.Author.User.Name

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
				reason, why = "ignored author", SkipIgnoredAuthor
			}
			r.log.InfoContext(ctx, fmt.Sprintf("Skipping %s by %s (%s)", prTag, author, reason))
			return nil, ctx, r.abort(ctx, key, kind, why, prTag, httpCounter)
		}
	}
	// 3. A push by an ignored account (CI amending someone's PR) is not a
	// reason to re-review; the author's next push is.
	if !skipAuthorCheck && payload.Actor != nil && r.isIgnoredAuthor(payload.Actor.Name) {
		r.log.InfoContext(ctx, fmt.Sprintf("Skipping %s: pushed by ignored author %s", prTag, payload.Actor.Name))
		return nil, ctx, r.abort(ctx, key, kind, SkipIgnoredAuthor, prTag, httpCounter)
	}

	// 4. Skip state. If the user removed our summary comment they want
	// noergler to leave this PR alone.
	if !r.checkSkipState(ctx, key, prTag) {
		return nil, ctx, r.abort(ctx, key, kind, SkipIgnoredPR, prTag, httpCounter)
	}

	upsert := store.PRUpsert{
		Key: key, TeamSlug: r.TeamSlug,
		Author: &author, Title: &pr.Title, OpenedAt: epochMSToTime(pr.CreatedDate),
	}

	// 5. Opt-out keyword in the branch name.
	if keyword := r.cfg.OptOutBranchKeyword; keyword != "" &&
		strings.Contains(strings.ToLower(pr.FromRef.DisplayID), strings.ToLower(keyword)) {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: branch %q contains opt-out keyword %q, skipping review",
			prTag, pr.FromRef.DisplayID, keyword))
		// The pointer advances here: an opt-out branch is a stable state of
		// the PR, not a transient failure.
		prReviewID := r.upsert(ctx, upsert, pr.FromRef.LatestCommit)
		r.postOrUpdateSummary(ctx, project, repo, pr.ID, prReviewID,
			render.OptOutBranchSummary(keyword, pr.FromRef.DisplayID))
		return nil, ctx, r.abort(ctx, key, kind, SkipBranchOptOut, prTag, httpCounter)
	}

	repoInstructions := r.fetchRepoInstructions(ctx, project, repo, pr.FromRef.LatestCommit, pr.ToRef.DisplayID)

	// 6. AGENTS.md required and absent.
	if repoInstructions == "" && r.cfg.RequireAgentsMD {
		r.log.InfoContext(ctx, fmt.Sprintf(
			"%s: no AGENTS.md found on PR or target branch, skipping review (set REVIEW_REQUIRE_AGENTS_MD=false to override)", prTag))
		prReviewID := r.upsert(ctx, upsert, pr.FromRef.LatestCommit)
		r.postOrUpdateSummary(ctx, project, repo, pr.ID, prReviewID, render.AgentsMDMissingSummary())
		return nil, ctx, r.abort(ctx, key, kind, SkipNoAgentsMD, prTag, httpCounter)
	}

	// 7. AGENTS.md over the hard token limit.
	if repoInstructions != "" && r.cfg.AgentsMDMaxTokens > 0 {
		if n := r.tokens.Count(repoInstructions); n > r.cfg.AgentsMDMaxTokens {
			r.log.InfoContext(ctx, fmt.Sprintf(
				"%s: AGENTS.md too large (%d > %d tokens), skipping review (set REVIEW_AGENTS_MD_MAX_TOKENS=0 to disable the hard limit)",
				prTag, n, r.cfg.AgentsMDMaxTokens))
			prReviewID := r.upsert(ctx, upsert, pr.FromRef.LatestCommit)
			r.postOrUpdateSummary(ctx, project, repo, pr.ID, prReviewID,
				render.AgentsMDTooLargeSummary(n, r.cfg.AgentsMDMaxTokens, r.cfg.AgentsMDCustomLink))
			return nil, ctx, r.abort(ctx, key, kind, SkipAgentsMDTooLarge, prTag, httpCounter)
		}
	}

	// 8. Per-PR cost cap, on the auto path only. A mention is the explicit
	// override and always runs. Fail open: an unpriced PR never blocks.
	if !skipAuthorCheck {
		if cost := safeDB(ctx, r.log, "PRCost", func() (*int64, error) {
			return r.store.PRCost(ctx, key)
		}); cost != nil {
			cumulative := float64(*cost) / nanoPerUSD
			if cumulative >= r.cfg.MaxPRCostUSD {
				// 2 decimals, not the 3 the cost log lines use: these are the
				// same two figures costLimitNotice renders into the banner
				// below, and a log that disagrees with the posted comment
				// about the number that paused reviews is worse than a log
				// that is less precise than its neighbours.
				r.log.InfoContext(ctx, fmt.Sprintf(
					"%s: PR cost $%.2f >= limit $%.2f - skipping auto-review (@%s to review manually, or raise REVIEW_MAX_PR_COST_USD)",
					prTag, cumulative, r.cfg.MaxPRCostUSD, r.bitbucket.BotUsername()))
				// This push was not reviewed, so the prior commit stands.
				prior := r.priorCommit(ctx, key)
				prReviewID := r.upsert(ctx, upsert, prior)
				r.costLimitNotice(ctx, project, repo, pr.ID, prReviewID, cumulative, r.cfg.MaxPRCostUSD)
				return nil, ctx, r.abort(ctx, key, kind, SkipCostCap, prTag, httpCounter)
			}
		}
	}

	r.log.InfoContext(ctx, fmt.Sprintf("Starting review of %s by %s (branch: %s)", prTag, author, pr.FromRef.DisplayID))
	started := time.Now()

	sourceCommit := pr.FromRef.LatestCommit
	lastReviewed := r.priorCommit(ctx, key)

	// 9. Incremental review when the event is a push and we have a pointer.
	rawDiff, cumulativePRDiff, incrementalFrom, ok := r.resolveDiff(ctx, payload, key, prTag, sourceCommit, lastReviewed, upsert)
	if !ok {
		return nil, ctx, r.abort(ctx, key, kind, SkipNone, prTag, httpCounter)
	}

	files, contentSkipped := r.prepareFiles(ctx, project, repo, rawDiff, sourceCommit, prTag)
	// 13. Nothing reviewable after the content fetch.
	if len(files) == 0 {
		r.log.InfoContext(ctx, prTag+" has no reviewable files after content fetch, skipping")
		return nil, ctx, r.abort(ctx, key, kind, SkipNoReviewable, prTag, httpCounter)
	}

	// Counted before compression: this is the whole PR's scope.
	diffAdded, diffRemoved := countDiffLines(rawDiff)
	totalFiles := len(files)

	// 14. Small PRs get context expansion; large ones are compressed first.
	var otherModified, deletedPaths, renamedPaths []string
	budget := r.llm.InputTokenBudget()
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
	}
	files = diff.ExpandAllFiles(files, r.cfg.DiffExtraLinesBefore, r.cfg.DiffExtraLinesAfter,
		r.cfg.DiffMaxExtraLinesDynamicContext, r.cfg.DiffAllowDynamicContext)

	// 15. Compression can drop everything.
	if len(files) == 0 {
		r.log.InfoContext(ctx, prTag+" has no reviewable files after compression, skipping")
		return nil, ctx, r.abort(ctx, key, kind, SkipNoReviewable, prTag, httpCounter)
	}

	// 16. Cross-file context, Jira, previously-posted findings.
	relationships := diff.BuildRelationships(files)
	crossFile := diff.RenderRelationships(relationships)
	if crossFile != "" {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: cross-file context: %d relationship(s)", prTag, len(relationships)))
	}

	ticket, parentTicket := r.fetchTicket(ctx, pr.FromRef.DisplayID, pr.Title, prTag)
	ticketContext := renderTicketContext(ticket, parentTicket)

	existing := safeDB(ctx, r.log, "ExistingFindings", func() ([]store.Finding, error) {
		return r.store.ExistingFindings(ctx, key)
	})
	posted := r.trimPreviouslyPosted(existing, budget)

	assembled := inference.AssembleReviewPrompt(inference.AssembleRequest{
		Template:              r.template,
		Files:                 files,
		RepoInstructions:      repoInstructions,
		OtherModifiedPaths:    otherModified,
		DeletedFilePaths:      deletedPaths,
		RenamedFilePaths:      renamedPaths,
		TicketContext:         ticketContext,
		TicketComplianceCheck: r.cfg.TicketComplianceCheck,
		CrossFileContext:      crossFile,
		CumulativePRDiff:      cumulativePRDiff,
		PreviouslyPosted:      posted,
	}, r.tokens.Count)

	return &reviewPlan{
		key:             key,
		project:         project,
		repo:            repo,
		prID:            pr.ID,
		prTag:           prTag,
		upsert:          upsert,
		sourceCommit:    sourceCommit,
		incrementalFrom: incrementalFrom,
		mention:         skipAuthorCheck,
		started:         started,
		counter:         httpCounter,
		prompt:          assembled.Prompt,
		promptTokens:    assembled.PromptTokens,
		existing:        existing,
		contentSkipped:  contentSkipped,
		ticket:          ticket,
		parentTicket:    parentTicket,
		breakdown:       breakdown(assembled),
		crossFileSyms:   symbolsOf(relationships),
		agentsMDFound:   repoInstructions != "",
		budget:          budget,
		filesReviewed:   len(files),
		totalFiles:      totalFiles + len(deletedPaths) + len(renamedPaths),
		diffAdded:       diffAdded,
		diffRemoved:     diffRemoved,
	}, ctx, true
}

// infer is the gateway call, and nothing else. It is the only stage that may
// run off the review worker.
func (r *Reviewer) infer(ctx context.Context, plan *reviewPlan) inference.ReviewResult {
	return r.llm.Review(ctx, inference.ReviewRequest{
		Prompt:         plan.prompt,
		PromptTokens:   plan.promptTokens,
		ResponseSchema: inference.ReviewResponseFormat(),
	})
}

// post is everything after the gateway answers: the outcome branches, the
// comments, the rows and the summary. It runs on the review worker, so every
// Bitbucket call here is still one at a time.
func (r *Reviewer) post(ctx context.Context, plan *reviewPlan, result inference.ReviewResult) {
	defer r.logHTTPTotals(ctx, plan.prTag, plan.counter)

	key, prTag, upsert := plan.key, plan.prTag, plan.upsert
	project, repo, sourceCommit := plan.project, plan.repo, plan.sourceCommit
	started, existing := plan.started, plan.existing

	// One cost record per call, and only when a call actually happened: the
	// too-large branch decides locally before any request, and a
	// transport error returns with no cost, so both would otherwise report an
	// absent cost for a call the gateway never answered.
	if result.Outcome == inference.OutcomeOK || result.Outcome == inference.OutcomeUnparseable {
		r.logCost(ctx, prTag, result.Cost)
	}

	// 17-19. Terminal branches. Each preserves the prior commit, posts a
	// notice and writes no run row.
	//
	// 19b. OutcomeError is NOT one of them: a non-overflow API error is only
	// logged. No upsert, no notice, no run row.
	if result.Outcome != inference.OutcomeOK {
		r.handleNonOK(ctx, result, key, runKind(plan.mention), prTag, sourceCommit, upsert)
		return
	}

	// 20. Dedup against what earlier runs already posted.
	findings := dedupe(result.Review.Findings, existing)
	// 21. Sort by severity and cap.
	findings, truncated := sortAndLimit(findings, r.cfg.MaxComments)

	// 22. The pointer finally advances.
	prReviewID := r.upsert(ctx, upsert, sourceCommit)

	// 23. Post the inline comments. postedIDs is parallel to findings, with
	// a zero where the post failed, so the count of successes is separate
	// from the slice length.
	postedIDs, postedCount, failed := r.postInlineComments(ctx, project, repo, plan.prID, findings)

	elapsed := time.Since(started)

	// 24. Run row, then cost, then the summary.
	//
	// The run row must exist before any finding row: store.Finding.RunID is
	// required and comes from InsertRun, so posting and inserting findings
	// cannot be interleaved ahead of it. The comments are already on the PR
	// by this point, so a crash before InsertRun leaves them there with no
	// finding row.
	runID := r.recordRun(ctx, prReviewID, result, sourceCommit, plan.incrementalFrom, plan.mention,
		elapsed, postedCount, plan.diffAdded, plan.diffRemoved, plan.totalFiles)
	r.recordFindings(ctx, prReviewID, runID, findings, postedIDs)

	// The attempt row carries the run's id, so the feed can show this run's
	// findings and cost without the dashboard having to guess which run a
	// successful attempt produced. Written after InsertRun for that reason;
	// a zero runID (the insert failed) still records that the attempt
	// succeeded, with the figures missing rather than wrong.
	attempt := store.Attempt{
		Key: key, TeamSlug: r.TeamSlug, Kind: runKind(plan.mention),
		Outcome: inference.OutcomeOK.String(),
	}
	ms := elapsed.Milliseconds()
	attempt.ElapsedMS = &ms
	if runID != 0 {
		attempt.RunID = &runID
	}
	r.recordAttempt(ctx, attempt)

	runCost, cumulativeCost := r.resolveCost(ctx, key, result)

	summary := render.Summary(render.SummaryInput{
		Findings:                   findings,
		Truncated:                  truncated,
		Summary:                    result.Review.Summary,
		AgentsMDFound:              plan.agentsMDFound,
		ContentSkippedFiles:        plan.contentSkipped,
		TokenUsage:                 tokenUsage(result),
		PromptBreakdown:            plan.breakdown,
		Ticket:                     plan.ticket,
		ParentTicket:               plan.parentTicket,
		ComplianceRequirements:     result.Review.ComplianceRequirements,
		ComplianceExtractionFailed: result.Review.ComplianceRequirements == nil,
		TicketComplianceCheck:      r.cfg.TicketComplianceCheck,
		JiraEnabled:                r.jiraEnabled(),
		ElapsedSeconds:             elapsed.Seconds(),
		ElapsedPresent:             true,
		ReviewedCommit:             sourceCommit,
		IncrementalFrom:            plan.incrementalFrom,
		FilesReviewed:              plan.filesReviewed,
		TotalFiles:                 plan.totalFiles,
		FilesCountsSet:             true,
		DiffAdded:                  plan.diffAdded,
		DiffRemoved:                plan.diffRemoved,
		CrossFileSymbols:           plan.crossFileSyms,
		InputBudget:                plan.budget,
		ContextWindow:              r.llm.ContextWindow(),
		ModelLabel:                 r.llm.Label(),
		RunCostUSD:                 runCost,
		CumulativeCostUSD:          cumulativeCost,
		KeySpendUSD:                keySpendUSD(result),
		MaxPRCostUSD:               r.cfg.MaxPRCostUSD,
		AgentsMDWarnTokens:         r.cfg.AgentsMDWarnTokens,
	})

	// A completed run at or over the cap keeps the banner on top, so the
	// over-budget state stays loud. The summary is freshly built each run, so
	// no strip is needed.
	if cumulativeCost != nil && *cumulativeCost >= r.cfg.MaxPRCostUSD {
		summary = render.CostLimitBanner(*cumulativeCost, r.cfg.MaxPRCostUSD, r.bitbucket.BotUsername(), false) +
			"\n\n" + summary
	}

	r.postOrUpdateSummary(ctx, project, repo, plan.prID, prReviewID, summary)

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

// checkSkipState reports whether the review may proceed.
//
// An ignored PR stops here. When a summary comment is tracked, a 404 on
// refetch means the user deleted it, which is the backstop for repos not
// subscribed to pr:comment:deleted. Any other status, or a network failure,
// falls THROUGH to the review: never silence a live PR on a transient fault.
func (r *Reviewer) checkSkipState(ctx context.Context, key store.PRKey, prTag string) bool {
	state := safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
		return r.store.GetSkipState(ctx, key)
	})
	if state == nil {
		return true
	}
	if state.IgnoredAt != nil {
		r.log.InfoContext(ctx, prTag+": PR ignored (summary comment removed) - skipping")
		return false
	}
	if state.Summary == nil {
		return true
	}

	_, err := r.bitbucket.FetchPRComment(ctx, key.Project, key.Repo, key.PRID, state.Summary.ID)
	if err == nil {
		return true
	}
	if status := bitbucket.Status(err); status == http.StatusNotFound {
		safeDBErr(ctx, r.log, "MarkIgnored", func() error { return r.store.MarkIgnored(ctx, key) })
		r.log.InfoContext(ctx, fmt.Sprintf("%s: summary comment %d was deleted - ignoring PR from now on",
			prTag, state.Summary.ID))
		return false
	} else if status != 0 {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: could not verify summary comment %d (HTTP %d) - proceeding",
			prTag, state.Summary.ID, status))
		return true
	}
	r.log.WarnContext(ctx, prTag+": summary-comment check failed - proceeding")
	return true
}

// resolveDiff decides between an incremental and a full review and fetches
// the diff. ok is false when the review must stop.
func (r *Reviewer) resolveDiff(ctx context.Context, payload *webhook.Payload, key store.PRKey, prTag, sourceCommit, lastReviewed string, upsert store.PRUpsert) (rawDiff, cumulative, incrementalFrom string, ok bool) {
	project, repo, prID := key.Project, key.Repo, key.PRID
	isIncremental := false

	if payload.EventKey == webhook.EventFromRefUpdated && lastReviewed != "" && sourceCommit != "" {
		// 9. Same SHA: nothing changed (a retrigger or a no-op force push).
		if sourceCommit == lastReviewed {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: HEAD unchanged since last review (%s), skipping",
				prTag, shortSHA(sourceCommit, 10)))
			return "", "", "", false
		}
		incDiff, err := r.bitbucket.FetchCommitDiff(ctx, project, repo, lastReviewed, sourceCommit)
		switch {
		case errors.Is(err, bitbucket.ErrIncrementalDiffUnavailable):
			// Expected control flow: the branch was rebased or squashed.
			// INFO, no traceback, and fall through to the full review. We
			// must NOT skip just because the optimization did not apply.
			r.log.InfoContext(ctx, fmt.Sprintf("%s: incremental diff unavailable (%v) - running full review", prTag, err))
		case err != nil:
			r.log.WarnContext(ctx, fmt.Sprintf("%s: incremental diff failed unexpectedly, falling back to full review: %v", prTag, err))
		case strings.TrimSpace(incDiff) != "":
			rawDiff = incDiff
			isIncremental = true
			incrementalFrom = lastReviewed
			r.log.InfoContext(ctx, fmt.Sprintf("%s: incremental review %s..%s",
				prTag, shortSHA(lastReviewed, 10), shortSHA(sourceCommit, 10)))
		default:
			// HEAD moved but the compare is empty: a rebase to an identical
			// tree, or a Bitbucket edge case. Never silently skip when the
			// SHA actually changed.
			r.log.WarnContext(ctx, fmt.Sprintf(
				"%s: HEAD moved %s -> %s but compare/diff is empty - falling back to full review to avoid missing changes",
				prTag, shortSHA(lastReviewed, 10), shortSHA(sourceCommit, 10)))
		}
	}

	if !isIncremental {
		full, err := r.bitbucket.FetchPRDiff(ctx, project, repo, prID, 0)
		var tooLarge *bitbucket.ContentTooLarge
		if errors.As(err, &tooLarge) {
			// 10. Nothing in this push was reviewed, so the next push must
			// not go incremental from it.
			r.logDiffTooLarge(ctx, prTag, project, repo, prID, err, tooLarge)
			prior := r.priorCommit(ctx, key)
			prReviewID := r.upsert(ctx, upsert, prior)
			r.postOrUpdateSummary(ctx, project, repo, prID, prReviewID, render.DiffTooLargeSummary(tooLarge.Limit))
			return "", "", "", false
		}
		if err != nil {
			r.log.ErrorContext(ctx, fmt.Sprintf("%s: failed to fetch PR diff: %v", prTag, err))
			return "", "", "", false
		}
		// 11. An empty diff is nothing to review.
		if strings.TrimSpace(full) == "" {
			r.log.InfoContext(ctx, prTag+" has empty diff, skipping")
			return "", "", "", false
		}
		return full, "", "", true
	}

	// 12. The cumulative PR diff is cross-file context for an incremental
	// review, so the model can check invariants split across commits. Best
	// effort: a failure here must not block the review.
	cumulative = r.fetchCumulativeDiff(ctx, key, prTag)
	return rawDiff, cumulative, incrementalFrom, true
}

// fetchCumulativeDiff returns the whole-PR diff for cross-file context, or
// "" when it is unavailable or over budget.
func (r *Reviewer) fetchCumulativeDiff(ctx context.Context, key store.PRKey, prTag string) string {
	full, err := r.bitbucket.FetchPRDiff(ctx, key.Project, key.Repo, key.PRID, 0)
	var tooLarge *bitbucket.ContentTooLarge
	switch {
	case errors.As(err, &tooLarge):
		r.log.InfoContext(ctx, fmt.Sprintf("%s: cumulative PR diff dropped (%v)", prTag, err))
		return ""
	case err != nil:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: failed to fetch cumulative PR diff for context: %v", prTag, err))
		return ""
	case full == "":
		return ""
	}

	budget := inference.CumulativeDiffBudget(r.llm.InputTokenBudget())
	// Tokenizing expands the text in RAM, so a diff hopelessly over budget
	// by byte count alone is dropped without ever being tokenized.
	size, unit := len(full), "bytes"
	if size <= budget*bytesPerTokenCeiling {
		size, unit = r.tokens.Count(full), "tokens"
	}
	if size > budget {
		r.log.WarnContext(ctx, fmt.Sprintf(
			"%s: cumulative PR diff %d %s exceeds budget %d tokens (model context %d), dropping",
			prTag, size, unit, budget, r.llm.ContextWindow()))
		return ""
	}
	return full
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
func (r *Reviewer) handleNonOK(ctx context.Context, result inference.ReviewResult, key store.PRKey, kind store.RunKind, prTag, sourceCommit string, upsert store.PRUpsert) {
	short := shortOrUnknown(sourceCommit)
	project, repo, prID := key.Project, key.Repo, key.PRID

	if result.Outcome == inference.OutcomeError {
		r.log.ErrorContext(ctx, fmt.Sprintf("Review of %s failed: %v", prTag, result.Err))
		r.recordAttempt(ctx, store.Attempt{
			Key: key, TeamSlug: r.TeamSlug, Kind: kind,
			Outcome: result.Outcome.String(),
		})
		return
	}

	// Read the prior commit BEFORE the upsert, which would otherwise
	// overwrite it with this failed commit.
	prior := r.priorCommit(ctx, key)
	prReviewID := r.upsert(ctx, upsert, prior)

	switch result.Outcome {
	case inference.OutcomeTimedOut:
		r.log.ErrorContext(ctx, fmt.Sprintf("Review of %s aborted - no response within %.0fs (commit %s)",
			prTag, inference.CallTimeout.Seconds(), short))
		r.timeoutNotice(ctx, project, repo, prID, prReviewID, sourceCommit, prior)
	case inference.OutcomeUnparseable:
		r.log.ErrorContext(ctx, fmt.Sprintf(
			"Review of %s aborted - model returned an unparseable/refused response (commit %s)", prTag, short))
		r.unparseableNotice(ctx, project, repo, prID, prReviewID, sourceCommit, prior)
	case inference.OutcomeTooLarge:
		r.log.ErrorContext(ctx, fmt.Sprintf(
			"Review of %s skipped - PR too large for the model's context window (commit %s)", prTag, short))
		r.tooLargeNotice(ctx, project, repo, prID, prReviewID, sourceCommit, prior)
	}

	r.recordAttempt(ctx, store.Attempt{
		Key: key, TeamSlug: r.TeamSlug, Kind: kind,
		Outcome: result.Outcome.String(),
	})
}

// fetchTicket resolves the Jira ticket for a PR, or nil when there is none.
func (r *Reviewer) fetchTicket(ctx context.Context, branch, title, prTag string) (ticket, parent *jira.Ticket) {
	if !r.jiraEnabled() {
		return nil, nil
	}
	id := extractTicketID(branch, title)
	if id == "" {
		return nil, nil
	}
	ticket, parent, err := r.jira.FetchTicketWithParent(ctx, id)
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: Jira lookup for %s failed: %v", prTag, id, err))
		return nil, nil
	}
	if ticket != nil {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: linked Jira ticket %s", prTag, id))
	}
	return ticket, parent
}

// trimPreviouslyPosted caps the previously-posted block by count and then by
// rendered token size.
//
// The tail is the most recent (the store returns oldest first). Dropping the
// oldest in 25% chunks avoids re-rendering once per item.
func (r *Reviewer) trimPreviouslyPosted(existing []store.Finding, budget int) []inference.PostedFinding {
	posted := make([]inference.PostedFinding, 0, len(existing))
	for _, f := range existing {
		line := f.LineNumber
		posted = append(posted, inference.PostedFinding{
			FilePath: f.FilePath, LineNumber: &line, Severity: f.Severity, CommentText: f.CommentText,
		})
	}
	if len(posted) > maxPreviouslyPostedFindings {
		posted = posted[len(posted)-maxPreviouslyPostedFindings:]
	}

	limit := inference.PreviouslyPostedBudget(budget)
	for len(posted) > 0 && r.tokens.Count(inference.RenderPreviouslyPostedFindings(posted)) > limit {
		drop := max(1, len(posted)/4)
		posted = posted[drop:]
	}
	return posted
}

// dedupe drops findings an earlier run already posted, keyed on file, line
// and severity.
func dedupe(findings []inference.ReviewFinding, existing []store.Finding) []inference.ReviewFinding {
	type dedupeKey struct {
		file     string
		line     int
		severity string
	}
	seen := make(map[dedupeKey]bool, len(existing))
	for _, f := range existing {
		seen[dedupeKey{f.FilePath, f.LineNumber, f.Severity}] = true
	}
	out := make([]inference.ReviewFinding, 0, len(findings))
	for _, f := range findings {
		if !seen[dedupeKey{f.File, f.Line, f.Severity}] {
			out = append(out, f)
		}
	}
	return out
}

// severityOrder ranks a severity for sorting; an unknown value sorts last.
func severityOrder(s string) int {
	switch s {
	case "issue":
		return 0
	case "suggestion":
		return 1
	default:
		return 99
	}
}

// sortAndLimit orders findings by severity and caps them.
//
// SortStableFunc, not SortFunc: findings of equal severity must keep the
// model's order or the numbered Issues list shuffles between runs of the
// same review (TestSortAndLimit).
func sortAndLimit(findings []inference.ReviewFinding, maxComments int) ([]inference.ReviewFinding, bool) {
	sorted := slices.Clone(findings)
	slices.SortStableFunc(sorted, func(a, b inference.ReviewFinding) int {
		return severityOrder(a.Severity) - severityOrder(b.Severity)
	})
	truncated := len(sorted) > maxComments
	if truncated {
		sorted = sorted[:maxComments]
	}
	return sorted, truncated
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
func (r *Reviewer) postInlineComments(ctx context.Context, project, repo string, prID int, findings []inference.ReviewFinding) (ids []int, posted, failed int) {
	ids = make([]int, len(findings))
	for i, f := range findings {
		id, err := r.bitbucket.PostInlineComment(ctx, project, repo, prID, f.File, f.Line, render.InlineComment(f))
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
	sourceCommit, incrementalFrom string, mention bool, elapsed time.Duration,
	postedCount, added, removed, filesChanged int) int64 {
	if prReviewID == 0 {
		return 0
	}
	kind := store.RunAuto
	if mention {
		kind = store.RunMention
	}
	var from *string
	if incrementalFrom != "" {
		from = &incrementalFrom
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
			FindingsPosted:   postedCount,
			LinesAdded:       added,
			LinesRemoved:     removed,
			FilesChanged:     filesChanged,
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

// priorCommit is the last successfully reviewed commit, or "".
func (r *Reviewer) priorCommit(ctx context.Context, key store.PRKey) string {
	commit := safeDB(ctx, r.log, "GetLastReviewedCommit", func() (string, error) {
		c, ok, err := r.store.GetLastReviewedCommit(ctx, key)
		if !ok {
			return "", err
		}
		return c, err
	})
	return commit
}

func tokenUsage(result inference.ReviewResult) render.TokenUsage {
	return render.TokenUsage{
		Prompt:     int(result.Cost.PromptTokens),
		Completion: int(result.Cost.CompletionTokens),
		Present:    true,
	}
}

func breakdown(a inference.AssembledPrompt) render.PromptBreakdown {
	return render.PromptBreakdown{
		Template:         a.Breakdown.Template,
		RepoInstructions: a.Breakdown.RepoInstructions,
		Files:            a.Breakdown.Files,
		Present:          true,
	}
}

func keySpendUSD(result inference.ReviewResult) float64 {
	if result.Cost.KeySpendNanoUSD == nil {
		return 0
	}
	return float64(*result.Cost.KeySpendNanoUSD) / nanoPerUSD
}

func symbolsOf(rels []diff.CrossFileRelationship) []string {
	if len(rels) == 0 {
		return nil
	}
	out := make([]string, len(rels))
	for i, rel := range rels {
		out[i] = rel.Symbol
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
