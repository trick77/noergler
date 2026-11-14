package review

import (
	"context"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"

	"github.com/trick77/noergler/internal/gitrepo"
	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/merge"
	"github.com/trick77/noergler/internal/prompts"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// The PR author can have noergler resolve a simple merge conflict ("behebe
// den Merge-Konflikt"; user, 2026-11-13): the target branch merged into the
// PR's branch as a merge commit, pushed over git. Nothing builds or tests
// it, so only simple conflicts, all or nothing:
//   - code first: a conflict where both branches only added lines, none in
//     common, is target's lines then source's, no model, no cost;
//   - the rest goes to the model, which picks existing lines by id and
//     writes none and brings back no line a branch deleted or changed
//     (merge.ValidateResolution checks both);
//   - every line dropped is quoted in the answer.
// The bounds are merge's constants. Only the PR author may ask. The
// fetch and the push run on the worker, like every Bitbucket call.

const (
	// conflictStageTimeout bounds each worker stage of a request: the
	// fetch and merge, and the commit and push.
	conflictStageTimeout = 60 * time.Second
	// conflictMaxBytes bounds what a fetch reads off the wire, and
	// conflictMaxInflated what it holds: the objects stay in memory,
	// inflated, beside GOMEMLIMIT (1500 MiB) and every team's reviews.
	conflictMaxBytes    int64 = 300 << 20
	conflictMaxInflated int64 = 600 << 20
	// conflictMaxFileBytes bounds one both-changed file, as the file
	// fetches of a review are (BITBUCKET_MAX_FILE_BYTES' default).
	conflictMaxFileBytes int64 = 1 << 20
)

// conflictDepths are the fetch depths tried for the merge base: most PRs
// branch off within the first, a long-lived branch needs the second.
var conflictDepths = []int{50, 500}

// The command's words, decided by code like a stop (stop.go): a verb near
// "conflict", in a short comment, neither negated nor about the commenter
// doing it ("I'll resolve the conflict myself", "Ich löse den Konflikt
// später"): a push the author did not ask for is worse than a missed
// command. Wording it misses reaches the Q&A, which answers in words.
const (
	conflictVerbs = `behebe|beheben|behebst|löse|lösen|löst|loese|loesen|auflösen|aufzulösen|` +
		`fix|fixe|fixen|flick|flicke|flicken|resolve|solve`
	conflictNouns = `konflikt|konflikte|konflikts|mergekonflikt|mergekonflikte|conflict|conflicts`
)

var (
	conflictNear = near(conflictVerbs, conflictNouns, 3)
	conflictNot  = words(`nicht|kein|keine|keinen|nie|niemals|not|no|never|don't|dont|do not|doesn't|won't|` +
		`can't|cant|cannot|shouldn't`)
	// conflictSelf is the commenter speaking of their own doing, or of
	// later: no order to the bot.
	conflictSelf = words(`i|i'll|i'm|i've|myself|we|we'll|ich|wir|selbst|selber|später|later|tomorrow|morgen`)
)

// conflictRequest reports whether text, the comment with the mention
// stripped, asks noergler to resolve the PR's merge conflict.
func conflictRequest(text string) bool {
	t := normalize(text)
	return short(t) && !conflictNot.MatchString(t) && !conflictSelf.MatchString(t) && conflictNear.MatchString(t)
}

// conflictJob is one request across its stages. Across the model's call it
// holds the merged files and the tree's entries, never the fetch: the repo
// is detached.
type conflictJob struct {
	key                    store.PRKey
	prTag, team            string
	commentID              int
	lang                   lang.Lang
	source, target         string
	sourceHead, targetHead plumbing.Hash
	remote                 gitrepo.Remote
	repo                   *gitrepo.Repo
	sig                    gitrepo.Signature
	// files is the merge's tree; content the new bytes of the paths both
	// branches changed and merged cleanly.
	files   map[string]gitrepo.Entry
	content map[string][]byte
	merged  []*conflictFile
	clean   []string
	// pending are the conflicts the model answers, in the prompt's order.
	pending []conflictRef
	res     inference.ConflictResult
	// counter is the mention's HTTP scope: a staged request logs its
	// totals when its post ends, like a Q&A.
	counter *httpstats.Counter
	// offer is a review's check for an offer (conflictOffer), not a
	// request: it merges and pushes nothing, and logs as an offer.
	offer bool
}

// conflictFile is a file with conflicts: its chunks, and per conflict the
// resolution (nil until the model's is in) and who chose it.
type conflictFile struct {
	path        string
	chunks      []merge.Chunk
	conflicts   []*merge.Conflict
	resolutions [][]merge.Ref
	byModel     []bool
}

type conflictRef struct {
	file *conflictFile
	k    int
}

// handleConflict answers a merge-conflict request. handedOff as for a
// staged Q&A: the model's call went to the pool, and the post logs the
// HTTP totals of counter.
func (r *Reviewer) handleConflict(ctx context.Context, payload *webhook.Payload, key store.PRKey, prTag, team string,
	sched Scheduler, counter *httpstats.Counter) (handedOff bool) {
	comment, pr := payload.Comment, payload.PullRequest
	l := r.languageFor(ctx, comment.Author.Name)
	if pr.Author.User.Name == "" || !strings.EqualFold(comment.Author.Name, pr.Author.User.Name) {
		r.log.InfoContext(ctx, prTag+": merge conflict request not from the PR's author: not resolved")
		r.reply(ctx, key.Project, key.Repo, pr.ID, comment.ID, l.Pick(
			"Merge-Konflikte löse ich nur auf Anfrage des PR-Autors auf.", //nolint:misspell // German "Autors"
			"I resolve merge conflicts only when the PR's author asks."))
		return false
	}
	rowID, cumulative, over := r.overCap(ctx, key)
	if rowID == 0 {
		// The request's row is the PR's: a PR never reviewed gets one.
		r.ensurePRRow(ctx, key, pr.Author.User.Name, pr.Title, pr.CreatedDate)
	}
	r.reactivateIfIgnored(ctx, key, prTag, false)
	j := &conflictJob{key: key, prTag: prTag, team: team, commentID: comment.ID, lang: l,
		source: pr.FromRef.Branch(), target: pr.ToRef.Branch(), counter: counter}

	prepCtx, cancel := context.WithTimeout(ctx, conflictStageTimeout)
	outcome, why := r.prepareConflict(prepCtx, j, pr, over)
	cancel()
	if outcome != "" {
		r.finishConflict(ctx, j, outcome, "", why)
		return false
	}
	if len(j.pending) == 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict resolved by code alone: %d conflicts", prTag, j.count()))
		r.commitConflict(ctx, j)
		return false
	}
	// The model costs money: at the cap the request is answered, never
	// left silent, as a fix request is.
	if over {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: PR cost $%.2f >= limit $%.2f - merge conflict needs the model, not resolved",
			prTag, cumulative, r.cfg.MaxPRCostUSD))
		r.finishConflict(ctx, j, store.FixNotFeasible, "", l.Pick(
			"Diesen Merge-Konflikt kann ich nicht auflösen: das Kostenlimit dieses PRs ist erreicht.",
			"I can't resolve this merge conflict: this PR has reached its cost limit."))
		return false
	}
	req := r.conflictRequest(j)
	infer := func(inferCtx context.Context) { j.res = r.llm.ResolveConflict(inferCtx, req) }
	post := func(postCtx context.Context) { r.postConflict(postCtx, j) }
	sched.Stage(ctx, key, team, infer, post)
	return true
}

func (j *conflictJob) count() int {
	n := 0
	for _, f := range j.merged {
		n += len(f.conflicts)
	}
	return n
}

// prepareConflict fetches, merges and resolves what code can. outcome is
// "" when the merge goes on; otherwise it and the answer end the request.
// At the cost cap (overCap) only code may resolve, so the worker's time is
// spent on the shallow fetch alone, never the deeper retry.
func (r *Reviewer) prepareConflict(ctx context.Context, j *conflictJob, pr webhook.PullRequest, overCap bool) (outcome, answer string) {
	pick := j.lang.Pick
	cannot := func(logWhy, why string) (string, string) { return r.conflictRefused(ctx, j, logWhy, why) }
	failed := func(what string, err error) (string, string) { return r.conflictError(ctx, j, what, err) }
	if conflictFromFork(pr, j.key) {
		return cannot("fork", pick("der PR kommt aus einem Fork.", "the PR comes from a fork."))
	}
	if j.source == "" || j.target == "" {
		return cannot("no_branch", pick("die Branches des PRs sind unbekannt.", "the PR's branches are unknown."))
	}
	status, err := r.bitbucket.MergeStatus(ctx, j.key.Project, j.key.Repo, j.key.PRID)
	if err != nil {
		return failed("merge status", err)
	}
	if !status.Conflicted {
		r.log.InfoContext(ctx, j.prTag+": merge conflict requested, Bitbucket reports none")
		return store.FixNotFeasible, pick("Bitbucket meldet für diesen PR keinen Merge-Konflikt.",
			"Bitbucket reports no merge conflict on this PR.")
	}
	if outcome, answer := r.mergeConflict(ctx, j, overCap); outcome != "" {
		return outcome, answer
	}
	// The commit's author is the bot account as Bitbucket knows it: a
	// commit-checker hook compares it with the pusher, and a commit
	// without an email is no commit to leave on a branch.
	name, email, err := r.bitbucket.BotIdentity(ctx)
	if err != nil {
		return failed("bot identity", err)
	}
	if email == "" {
		return cannot("no_bot_email", pick("mein Bitbucket-Konto hat keine E-Mail-Adresse für den Commit.",
			"my Bitbucket account has no email address for the commit."))
	}
	j.sig = gitrepo.Signature{Name: name, Email: email}
	// The fetch's objects are not held across the model's call.
	j.repo = j.repo.Detach()
	return "", ""
}

// conflictFromFork reports a PR whose branch lives in another repository,
// whose history the target's need not reach (v1: not resolved).
func conflictFromFork(pr webhook.PullRequest, key store.PRKey) bool {
	src := pr.FromRef.Repository
	return src != nil && src.Slug != "" &&
		(!strings.EqualFold(src.Project.Key, key.Project) || !strings.EqualFold(src.Slug, key.Repo))
}

// conflictRefused logs why a conflict is no simple merge and is the
// request's answer; an offer (j.offer) only logs it.
func (r *Reviewer) conflictRefused(ctx context.Context, j *conflictJob, logWhy, why string) (string, string) {
	if j.offer {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict offer: none (%s)", j.prTag, logWhy))
	} else {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict not resolved: %s", j.prTag, logWhy))
	}
	return store.FixNotFeasible, conflictCannot(j.lang, why)
}

func (r *Reviewer) conflictError(ctx context.Context, j *conflictJob, what string, err error) (string, string) {
	if j.offer {
		// A review's own check: tried again at the next one, no request
		// failed.
		r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict offer: %s failed: %v", j.prTag, what, err))
	} else {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict: %s failed: %v", j.prTag, what, err))
	}
	return store.FixError, conflictFailed(j.lang)
}

// mergeConflict fetches both branches, merges them and resolves what code
// can, within merge's bounds: what a request does before the model, and
// all an offer does. outcome is "" when the merge stands, j.pending then
// holds what the model must answer. At the cost cap (overCap) only the
// shallow fetch is tried, never the deeper retry.
func (r *Reviewer) mergeConflict(ctx context.Context, j *conflictJob, overCap bool) (outcome, answer string) {
	pick := j.lang.Pick
	cannot := func(logWhy, why string) (string, string) { return r.conflictRefused(ctx, j, logWhy, why) }
	failed := func(what string, err error) (string, string) { return r.conflictError(ctx, j, what, err) }
	var err error
	j.remote, err = r.bitbucket.GitRemote(ctx, j.key.Project, j.key.Repo, conflictMaxBytes)
	if err != nil {
		return failed("clone URL", err)
	}
	j.remote.MaxInflated = conflictMaxInflated
	var heads []plumbing.Hash
	var base plumbing.Hash
	depths := conflictDepths
	if overCap {
		depths = depths[:1]
	}
	for i, depth := range depths {
		// The shallower fetch is let go before the deeper one: never both
		// in memory.
		j.repo = nil
		j.repo, heads, err = gitrepo.Fetch(ctx, j.remote, depth, j.source, j.target)
		if errors.Is(err, gitrepo.ErrTooLarge) {
			return cannot("too_large", pick("das Repository ist zu gross, um es zu holen.", "the repository is too large to fetch."))
		}
		if err != nil {
			return failed("fetch", err)
		}
		base, err = j.repo.MergeBase(heads[0], heads[1])
		if !errors.Is(err, gitrepo.ErrIncomplete) || i == len(depths)-1 {
			break
		}
	}
	repo := j.repo
	switch {
	case errors.Is(err, gitrepo.ErrIncomplete):
		return cannot("history_too_deep", pick("die gemeinsame Basis der Branches liegt zu weit zurück.",
			"the branches' common base lies too far back."))
	case errors.Is(err, gitrepo.ErrNoBase):
		return cannot("no_single_base", pick("die Branches haben keine eindeutige gemeinsame Basis.",
			"the branches have no single common base."))
	case err != nil:
		return failed("merge base", err)
	}
	j.sourceHead, j.targetHead = heads[0], heads[1]
	r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict: fetched %d bytes, base %s", j.prTag, repo.Read, render.ShortSHA(base.String(), 8)))

	files := make([]map[string]gitrepo.Entry, 3)
	for i, h := range []plumbing.Hash{base, j.sourceHead, j.targetHead} {
		if files[i], err = repo.Files(h); err != nil {
			return failed("tree", err)
		}
	}
	baseF, srcF, tgtF := files[0], files[1], files[2]
	paths := map[string]bool{}
	for _, m := range files {
		for p := range m {
			paths[p] = true
		}
	}
	sorted := make([]string, 0, len(paths))
	for p := range paths {
		sorted = append(sorted, p)
	}
	sort.Strings(sorted)

	j.files, j.content = map[string]gitrepo.Entry{}, map[string][]byte{}
	both := 0
	for _, p := range sorted {
		b, bok := baseF[p]
		s, sok := srcF[p]
		t, tok := tgtF[p]
		switch {
		case sok == tok && s == t:
			if sok {
				j.files[p] = s
			}
			continue
		case bok == sok && b == s:
			if tok {
				j.files[p] = t
			}
			continue
		case bok == tok && b == t:
			if sok {
				j.files[p] = s
			}
			continue
		}
		both++
		if both > merge.MaxBothChanged {
			return cannot("too_many_files", fmt.Sprintf(pick("mehr als %d Dateien wurden auf beiden Branches geändert.",
				"more than %d files changed on both branches."), merge.MaxBothChanged))
		}
		where := "`" + p + "`"
		switch {
		case !bok:
			return cannot("add_add "+p, where+pick(" wurde auf beiden Branches neu angelegt.", " was added on both branches."))
		case !sok || !tok:
			return cannot("delete_modify "+p, where+pick(" wurde auf einem Branch gelöscht und auf dem anderen geändert.",
				" was deleted on one branch and changed on the other."))
		case b.Mode != s.Mode || s.Mode != t.Mode || (s.Mode != filemode.Regular && s.Mode != filemode.Executable):
			return cannot("mode "+p, where+pick(" ist keine gewöhnliche Datei oder hat seinen Modus geändert.",
				" is no regular file or changed its mode."))
		}
		var blobs [3][]byte
		for i, e := range []gitrepo.Entry{b, s, t} {
			if blobs[i], err = repo.Blob(e.Hash, conflictMaxFileBytes); err != nil {
				if errors.Is(err, gitrepo.ErrTooLarge) {
					return cannot("file_too_large "+p, where+pick(" ist zu gross.", " is too large."))
				}
				return failed("blob", err)
			}
		}
		if why := merge.Excluded(p, blobs[0], blobs[1], blobs[2]); why != "" {
			return cannot(why+" "+p, where+pick(" ist eine Lock-, generierte oder Binärdatei.", " is a lock, generated or binary file."))
		}
		chunks, err := merge.Merge3(merge.Lines(blobs[0]), merge.Lines(blobs[2]), merge.Lines(blobs[1]))
		switch {
		case errors.Is(err, merge.ErrAmbiguous):
			return cannot("ambiguous "+p, where+pick(": wie die Zeilen zusammengehören, ist nicht eindeutig.",
				": how its lines line up is ambiguous."))
		case err != nil:
			return cannot("too_different "+p, where+pick(" wurde zu stark geändert.", " changed too much."))
		}
		j.files[p] = s
		cs := merge.Conflicts(chunks)
		if len(cs) == 0 {
			out, err := merge.Assemble(chunks, nil)
			if err != nil {
				return cannot("eol "+p, where+pick(": eine Zeile ohne Zeilenende stünde mitten in der Datei.",
					": a line without a newline would end up inside the file."))
			}
			j.content[p] = out
			j.clean = append(j.clean, p)
			continue
		}
		j.merged = append(j.merged, &conflictFile{path: p, chunks: chunks, conflicts: cs,
			resolutions: make([][]merge.Ref, len(cs)), byModel: make([]bool, len(cs))})
	}
	// A file one side added where the other added a directory: git's
	// file/directory conflict, which a tree cannot hold.
	for p := range j.files {
		for d := path.Dir(p); d != "." && d != "/"; d = path.Dir(d) {
			if _, ok := j.files[d]; ok {
				return cannot("file_dir "+d, "`"+d+"`"+pick(" ist auf einem Branch eine Datei, auf dem anderen ein Ordner.",
					" is a file on one branch and a directory on the other."))
			}
		}
	}
	n := j.count()
	switch {
	case n == 0:
		// Bitbucket sees a conflict this merge does not: never push a
		// merge the two disagree on.
		return cannot("no_conflict_found", pick("meine Sicht des Merges weicht von der von Bitbucket ab.",
			"my view of the merge differs from Bitbucket's."))
	case n > merge.MaxConflicts || len(j.merged) > merge.MaxConflictFiles:
		return cannot(fmt.Sprintf("too_many_conflicts %d in %d files", n, len(j.merged)), fmt.Sprintf(pick(
			"%d Konflikte in %d Dateien, ich löse höchstens %d in %d Dateien auf.",
			"%d conflicts in %d files, I resolve at most %d in %d files."), n, len(j.merged), merge.MaxConflicts, merge.MaxConflictFiles))
	}
	for _, f := range j.merged {
		for k, c := range f.conflicts {
			if c.Oversized() {
				return cannot("conflict_too_large "+f.path, "`"+f.path+"`"+fmt.Sprintf(pick(
					": ein Konflikt ist grösser als %d Zeilen.", ": a conflict is larger than %d lines."), merge.MaxRegionLines))
			}
			if c.Trivial() {
				f.resolutions[k] = merge.TrivialResolution(c)
				continue
			}
			f.byModel[k] = true
			j.pending = append(j.pending, conflictRef{file: f, k: k})
		}
	}
	return "", ""
}

// conflictRequest is the model's request for the pending conflicts.
func (r *Reviewer) conflictRequest(j *conflictJob) inference.ConflictRequest {
	hunks := make([]inference.ConflictHunk, 0, len(j.pending))
	for _, pc := range j.pending {
		before, after := merge.Context(pc.file.chunks, pc.k)
		hunks = append(hunks, inference.HunkOf(pc.file.path, pc.file.conflicts[pc.k], before, after))
	}
	prompt := inference.RenderConflictPrompt(inference.ConflictPromptRequest{
		Template: prompts.MergeConflict, TargetBranch: j.target, SourceBranch: j.source, Hunks: hunks, Language: j.lang,
	})
	return inference.ConflictRequest{Prompt: prompt,
		PromptTokens: r.tokens.Count(inference.ConflictSystemMessage) + r.tokens.Count(prompt)}
}

// The answers more than one stage gives.
func conflictFailed(l lang.Lang) string {
	return l.Pick("Das Auflösen des Merge-Konflikts ist fehlgeschlagen, bitte nochmals versuchen.",
		"Resolving the merge conflict failed, please ask again.")
}

func conflictUnsafe(l lang.Lang) string {
	return l.Pick("Diesen Merge-Konflikt kann ich nicht sicher selbst auflösen. Bitte löse ihn selbst auf.",
		"I can't resolve this merge conflict safely myself. Please resolve it yourself.")
}

func conflictCannot(l lang.Lang, why string) string {
	return l.Pick("Diesen Merge-Konflikt kann ich nicht selbst auflösen: ", "I can't resolve this merge conflict myself: ") + why
}

// postConflict is the model's answer checked and committed, on the worker.
func (r *Reviewer) postConflict(ctx context.Context, j *conflictJob) {
	defer r.logHTTPTotals(ctx, j.prTag, j.counter)
	pick := j.lang.Pick
	res := j.res
	if res.Cost.Answered {
		r.logCost(ctx, j.prTag, res.Cost)
	}
	r.recordCommentCall(ctx, j.key, store.CallFix, j.commentID, res.Cost)
	if r.owns != nil && !r.owns(j.key.Project, j.key.Repo) {
		r.log.InfoContext(ctx, j.prTag+": repo no longer claimed by the team, merge conflict not resolved")
		return
	}
	switch {
	case res.Outcome == inference.OutcomeTooLarge:
		r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict call too large: %v", j.prTag, res.Err))
		r.finishConflict(ctx, j, store.FixNotFeasible, "", conflictCannot(j.lang, pick("er ist zu gross.", "it is too large.")))
		return
	case res.Outcome != inference.OutcomeOK:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict call failed: %s %v", j.prTag, res.Outcome, res.Err))
		r.finishConflict(ctx, j, store.FixError, "", conflictFailed(j.lang))
		return
	case !res.Resolution.Feasible:
		why := sharpS(j.lang, strings.TrimSpace(res.Resolution.Reason))
		if why == "" {
			why = pick("er braucht mehr als die Wahl bestehender Zeilen.", "it needs more than choosing existing lines.")
		}
		r.log.InfoContext(ctx, j.prTag+": merge conflict not resolved: the model says not feasible")
		r.finishConflict(ctx, j, store.FixNotFeasible, "", conflictCannot(j.lang, lang.TruncateRunes(why, replyPostRunes/2)))
		return
	}
	rejected := func(why string) {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict resolution dropped: %s", j.prTag, why))
		r.finishConflict(ctx, j, store.FixRejected, "", conflictUnsafe(j.lang))
	}
	asked := make([]*merge.Conflict, len(j.pending))
	for i, pc := range j.pending {
		asked[i] = pc.file.conflicts[pc.k]
	}
	refs, why := merge.ReadAnswer(asked, res.Resolution.Lines)
	if why != "" {
		rejected(why)
		return
	}
	for i, pc := range j.pending {
		pc.file.resolutions[pc.k] = refs[i]
	}
	r.commitConflict(ctx, j)
}

// commitConflict assembles the files, commits the merge and pushes it.
func (r *Reviewer) commitConflict(ctx context.Context, j *conflictJob) {
	pick := j.lang.Pick
	ctx2, cancel := context.WithTimeout(ctx, conflictStageTimeout)
	defer cancel()
	content := make(map[string][]byte, len(j.content)+len(j.merged))
	for p, b := range j.content {
		content[p] = b
	}
	for _, f := range j.merged {
		out, err := merge.Assemble(f.chunks, f.resolutions)
		if err != nil {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict resolution dropped: %s: %v", j.prTag, f.path, err))
			r.finishConflict(ctx, j, store.FixRejected, "", conflictUnsafe(j.lang))
			return
		}
		content[f.path] = out
	}
	// The answer quotes every dropped line in full: one that would not fit
	// a comment is no merge to push, never a cut quote.
	placeholder := strings.Repeat("0", 40)
	if n := len([]rune(j.answer(placeholder, r.bitbucket.CommitURL(j.key.Project, j.key.Repo, placeholder)))); n > conflictAnswerRunes {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict not resolved: answer of %d runes, over %d", j.prTag, n, conflictAnswerRunes))
		r.finishConflict(ctx, j, store.FixNotFeasible, "", conflictCannot(j.lang, pick(
			"die verworfenen Zeilen sind zu lang, um sie hier vollständig zu zeigen.",
			"the dropped lines are too long to show here in full.")))
		return
	}
	commit, objects, err := j.repo.Commit(j.files, content, []plumbing.Hash{j.sourceHead, j.targetHead}, j.sig,
		j.commitMessage(r.bitbucket.CommentURL(j.key.Project, j.key.Repo, j.key.PRID, j.commentID)), time.Now())
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict: commit failed: %v", j.prTag, err))
		r.finishConflict(ctx, j, store.FixError, "", conflictFailed(j.lang))
		return
	}
	err = gitrepo.Push(ctx2, j.remote, j.repo, j.source, j.sourceHead, commit, objects)
	var rejected *gitrepo.Rejected
	switch {
	case err == nil:
	case errors.Is(err, gitrepo.ErrMoved):
		r.log.InfoContext(ctx, j.prTag+": merge conflict push refused: the branch moved")
		r.finishConflict(ctx, j, store.FixConflict, "", pick("Der Branch hat sich inzwischen bewegt. Bitte nochmals fragen.",
			"The branch has moved in the meantime. Please ask again."))
		return
	case errors.Is(err, gitrepo.ErrNoWrite):
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict push refused: %v", j.prTag, err))
		r.finishConflict(ctx, j, store.FixNoWrite, "", pick("Ich kann den Merge nicht pushen: mir fehlen die Schreibrechte auf diesen Branch.",
			"I can't push the merge: I have no write permission on this branch."))
		return
	case errors.As(err, &rejected):
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict push refused: %v", j.prTag, err))
		r.finishConflict(ctx, j, store.FixRefused, "", pick("Bitbucket hat den Push des Merges abgelehnt.", "Bitbucket refused the merge's push."))
		return
	default:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict push failed: %v", j.prTag, err))
		r.finishConflict(ctx, j, store.FixError, "", pick("Der Push des Merges ist fehlgeschlagen, bitte nochmals versuchen.",
			"Pushing the merge failed, please ask again."))
		return
	}
	sha := commit.String()
	r.finishConflict(ctx, j, store.FixCommitted, sha, j.answer(sha, r.bitbucket.CommitURL(j.key.Project, j.key.Repo, sha)))
	r.conflictResolvedOffer(ctx, j.key, j.prTag)
}

// commitMessage is git's own merge subject, then what was resolved how.
// English, like git's.
func (j *conflictJob) commitMessage(commentURL string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Merge branch '%s' into %s\n\nConflicts resolved by noergler on request:\n", j.target, j.source)
	for _, f := range j.merged {
		for k := range f.conflicts {
			how := "rule"
			if f.byModel[k] {
				how = "model"
			}
			fmt.Fprintf(&b, "- %s:%d (%s)\n", f.path, merge.LineOf(f.chunks, f.resolutions, k), how)
		}
	}
	b.WriteString("\n" + commentURL + "\n")
	return b.String()
}

// answer is the thread's confirmation: the commit, each spot and who chose
// it, every line dropped verbatim, the files merged without a conflict.
func (j *conflictJob) answer(sha, link string) string {
	pick := j.lang.Pick
	var b strings.Builder
	fmt.Fprintf(&b, "%s[`%s`](%s): %s\n\n", pick("Merge-Konflikt aufgelöst in ", "Merge conflict resolved in "),
		render.ShortSHA(sha, 8), link, fmt.Sprintf(pick("`%s` ist in `%s` gemergt.", "`%s` is merged into `%s`."), j.target, j.source))
	b.WriteString(pick("Aufgelöst:", "Resolved:") + "\n\n")
	type drop struct {
		where, side string
		lines       []string
	}
	var drops []drop
	for _, f := range j.merged {
		for k, c := range f.conflicts {
			where := fmt.Sprintf("%s:%d", f.path, merge.LineOf(f.chunks, f.resolutions, k))
			how := pick("Regel: beide Seiten übernommen", "rule: both sides kept")
			if f.byModel[k] {
				how = pick("Modell", "model")
			}
			fmt.Fprintf(&b, "- `%s` (%s)\n", where, how)
			tgt, src := merge.Dropped(c, f.resolutions[k])
			if len(tgt) > 0 {
				drops = append(drops, drop{where, j.target, tgt})
			}
			if len(src) > 0 {
				drops = append(drops, drop{where, j.source, src})
			}
		}
	}
	for _, d := range drops {
		body := strings.Join(d.lines, "\n")
		fence := lang.Fence(body)
		fmt.Fprintf(&b, "\n%s:\n\n%s\n%s\n%s\n", fmt.Sprintf(pick("Bei `%s` verworfen aus `%s`", "Dropped at `%s` from `%s`"), d.where, d.side),
			fence, body, fence)
	}
	if len(j.clean) > 0 {
		quoted := make([]string, len(j.clean))
		for i, p := range j.clean {
			quoted[i] = "`" + p + "`"
		}
		fmt.Fprintf(&b, "\n%s %s\n", pick("Ohne Konflikt zusammengeführt:", "Merged without a conflict:"), strings.Join(quoted, ", "))
	}
	b.WriteString("\n" + pick("Bitte bauen und testen, bevor du mergst.", "Please build and test before you merge."))
	return b.String()
}

// finishConflict records the request and answers it. Recorded before the
// answer: a pushed merge stands, whatever the post does.
func (r *Reviewer) finishConflict(ctx context.Context, j *conflictJob, outcome, sha, body string) {
	safeDBErr(ctx, r.log, "RecordFindingFix", func() error {
		_, err := r.store.RecordFindingFix(ctx, j.key, store.FindingFix{ConflictCommentID: j.commentID, TeamSlug: j.team,
			Outcome: outcome, CommitSHA: sha})
		return err
	})
	code, model := 0, 0
	for _, f := range j.merged {
		for _, m := range f.byModel {
			if m {
				model++
			} else {
				code++
			}
		}
	}
	tail := ""
	if sha != "" {
		tail = " " + sha
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict request: %s%s (%d code, %d model, %d clean)", j.prTag, outcome, tail, code, model, len(j.clean)))
	r.reply(ctx, j.key.Project, j.key.Repo, j.key.PRID, j.commentID, lang.TruncateRunes(body, conflictAnswerRunes))
}

// conflictAnswerRunes bounds the confirmation: dropped lines are quoted in
// full up to it (at most MaxConflicts regions of MaxRegionLines lines).
const conflictAnswerRunes = 12_000
