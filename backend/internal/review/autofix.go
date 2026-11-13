package review

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"path"
	"strings"
	"unicode/utf8"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
	"github.com/trick77/ship/kit/logging"
)

// The PR author can ask noergler to apply a finding itself ("flick das",
// "fix it"; user, 2026-11-13). The reply call classifies the request; the
// fix call writes one replacement in one file; ValidateFix checks it; the
// commit goes to the PR's source branch through Bitbucket's edit-file API.
// Always on, no setting: without write permission noergler says so, every
// time, and tries again on the next request (user, 2026-11-13).

// fixExcerptLines is how far around the finding's line the fix call sees
// the file: a fix changes a spot, so this bounds where it may land too.
const (
	fixExcerptLines      = 60
	fixExcerptLinesSmall = 20
)

// fixPlan is what a fix request needs on the worker, read before the call.
type fixPlan struct {
	// unavailable says why no fix can be asked for (log vocabulary), empty
	// when it can.
	unavailable string
	// The source side of the PR: its repo (a fork's for a fork PR), branch
	// and head, which the file was read at and the commit is based on.
	project, repo, branch, head string
	path                        string
	content                     []byte
	// code is the excerpt sent, lines from..to; anchor the finding's line
	// at the head.
	code             string
	from, to, anchor int
}

// fetchedFile is the finding's file as the reply read it: at the commit it
// was posted at, or the PR head when that commit is gone. ok is false when
// neither answered.
type fetchedFile struct {
	at, path, content string
	ok                bool
}

// prepareFix reads the finding's file at the PR head and finds the
// finding's line in it. The line number belongs to the commit the finding
// was posted at: at the head it is the line only when that commit is the
// head; otherwise the line is found by its text in the file read at that
// commit, the one nearest the old number (a later push moves lines). No
// such file, no text: no fix, never a guess.
func (r *Reviewer) prepareFix(ctx context.Context, pr webhook.PullRequest, project, repo, prTag string, f *store.Finding, run fetchedFile) *fixPlan {
	p := &fixPlan{project: project, repo: repo, branch: pr.FromRef.Branch(), head: pr.FromRef.LatestCommit,
		path: f.FilePath}
	if src := pr.FromRef.Repository; src != nil && src.Project.Key != "" && src.Slug != "" {
		p.project, p.repo = src.Project.Key, src.Slug
	}
	if p.head == "" || p.branch == "" {
		p.unavailable = "no_head"
		return p
	}
	var content []byte
	var err error
	if run.ok && run.at == p.head && p.project == project && p.repo == repo && !strings.ContainsRune(run.content, utf8.RuneError) {
		// The reply read this very file a moment ago: no second fetch on
		// the worker. Not when it holds U+FFFD: the text read replaces
		// invalid bytes with it, and a commit of that text would rewrite
		// every one of them; the bytes are fetched as served instead.
		content, p.path = []byte(run.content), run.path
	} else {
		content, err = r.bitbucket.FetchFileBytes(ctx, p.project, p.repo, p.head, f.FilePath)
		if stripped := inference.StripSidePrefix(f.FilePath); err != nil && stripped != f.FilePath && bitbucket.Status(err) == http.StatusNotFound {
			content, err = r.bitbucket.FetchFileBytes(ctx, p.project, p.repo, p.head, stripped)
			p.path = stripped
		}
	}
	var tooLarge *bitbucket.ContentTooLarge
	switch {
	case errors.As(err, &tooLarge):
		p.unavailable = "too_large"
		return p
	case err != nil:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: file for a fix on %s unavailable: %v", prTag, f.FilePath, err))
		p.unavailable = "unreadable"
		return p
	case bytes.IndexByte(content, 0) >= 0 || !utf8.Valid(content):
		// ValidateFix refuses the same: no call is paid for a fix that
		// could only be rejected (a Latin-1 file).
		p.unavailable = "binary"
		return p
	}
	p.content = content
	lines := fileLines(strings.ReplaceAll(string(content), "\r\n", "\n"))
	switch {
	case f.RunCommit != "" && f.RunCommit == p.head:
		if f.LineNumber >= 1 && f.LineNumber <= len(lines) {
			p.anchor = f.LineNumber
		}
	case run.ok && run.at == f.RunCommit:
		p.anchor = findLine(lines, fileLines(strings.ReplaceAll(run.content, "\r\n", "\n")), f.LineNumber)
	}
	if p.anchor == 0 {
		p.unavailable = "line_gone"
		return p
	}
	for _, n := range []int{fixExcerptLines, fixExcerptLinesSmall} {
		p.from, p.to = max(1, p.anchor-n), min(len(lines), p.anchor+n)
		p.code = strings.Join(lines[p.from-1:p.to], "\n")
		if len([]rune(p.code)) <= replyExcerptRunes {
			return p
		}
	}
	p.unavailable = "too_large"
	return p
}

// findLine is line n of old found again in lines (both 1-based), by text.
// A blank line has no text to find: the nearest non-blank line above it
// (else below) is found instead, and the offset carried over. The line's
// non-blank neighbours must match too, so a common line ("}", "return
// err") does not land on another occurrence when its own spot changed. 0
// when there is nothing to find.
func findLine(lines, old []string, n int) int {
	if n < 1 || n > len(old) {
		return 0
	}
	k := 0
	for d := 0; d < len(old) && k == 0; d++ {
		for _, i := range []int{n - d, n + d} {
			if i >= 1 && i <= len(old) && strings.TrimSpace(old[i-1]) != "" {
				k = i
				break
			}
		}
	}
	if k == 0 {
		return 0
	}
	at := nearestLine(lines, contextAt(old, k), k)
	if at == 0 {
		return 0
	}
	if got := at + (n - k); got >= 1 && got <= len(lines) {
		return got
	}
	return 0
}

// lineContext is a line's trimmed text and that of its nearest non-blank
// neighbours, "" past either end.
type lineContext struct{ above, line, below string }

func contextAt(lines []string, k int) lineContext {
	c := lineContext{line: strings.TrimSpace(lines[k-1])}
	for i := k - 1; i >= 1; i-- {
		if t := strings.TrimSpace(lines[i-1]); t != "" {
			c.above = t
			break
		}
	}
	for i := k + 1; i <= len(lines); i++ {
		if t := strings.TrimSpace(lines[i-1]); t != "" {
			c.below = t
			break
		}
	}
	return c
}

// nearestLine is the 1-based number of the line whose context is want,
// nearest to near; 0 when want's line is blank or no line has it.
func nearestLine(lines []string, want lineContext, near int) int {
	if want.line == "" {
		return 0
	}
	best := 0
	for i, l := range lines {
		// One neighbour may have changed (a line inserted above or below);
		// both changed is another place.
		if strings.TrimSpace(l) != want.line {
			continue
		}
		if c := contextAt(lines, i+1); c.above != want.above && c.below != want.below {
			continue
		}
		if n := i + 1; best == 0 || abs(n-near) < abs(best-near) {
			best = n
		}
	}
	return best
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// fixTarget is what a fix request's posting needs, from a finding's thread
// or from noergler's Q&A answer: one way through for both.
type fixTarget struct {
	key           store.PRKey
	team          string
	project, repo string
	prID          int
	prTag         string
	// where names the source in the log: "finding a.go:12", "answer 950".
	where string
	// replyTo is the author's request, the answer goes under it; threadID is
	// the thread's comment for the commit message's link.
	replyTo  int
	threadID *int
	lang     lang.Lang
	plan     *fixPlan
	res      *inference.FindingFixResult
	// One of findingID and answerID is set: the finding_fixes row's source.
	findingID int64
	answerID  int
	// anchor is an answer's code line, kept with the reply about it.
	anchor *store.BotReplyAnchor
	// done is the commit made for this source already, nil when none.
	done *string
	// prefix goes before the answer: what the same turn confirmed.
	prefix string
}

// postFix is a fix request's posting stage, on the worker: validate, commit,
// confirm. Every request gets one answer in the thread, never silence
// (user, 2026-11-13), and one finding_fixes row, the metric's source.
// posted reports whether the answer reached the thread.
func (r *Reviewer) postFix(ctx context.Context, t *fixTarget) (posted bool) {
	pick := t.lang.Pick
	if t.done != nil {
		return r.postFixAnswer(ctx, t, "", pick("Schon umgesetzt in `", "Already applied in `")+render.ShortSHA(*t.done, 8)+"`.")
	}
	outcome, sha, body := r.applyFix(ctx, t)
	// Recorded before the answer: the commit is made, whatever the post does.
	inserted, err := r.store.RecordFindingFix(ctx, t.key, store.FindingFix{FindingID: t.findingID, AnswerCommentID: t.answerID,
		TeamSlug: t.team, Outcome: outcome, CommitSHA: sha})
	switch {
	case err != nil:
		r.log.WarnContext(ctx, "DB operation failed, using fallback: RecordFindingFix", logging.Err(err))
	case !inserted && sha != "":
		r.log.WarnContext(ctx, fmt.Sprintf("%s: fix on %s committed as %s but not recorded: it has a committed fix already, or its PR no row", t.prTag, t.where, sha))
	case !inserted:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: fix request on %s (%s) not recorded: its PR has no row", t.prTag, t.where, outcome))
	}
	posted = r.postFixAnswer(ctx, t, sha, body)
	r.log.InfoContext(ctx, fmt.Sprintf("%s: fix requested on %s: %s%s", t.prTag, t.where, outcome, func() string {
		if sha == "" {
			return ""
		}
		return " " + sha
	}()))
	return posted
}

// applyFix decides the outcome and writes the commit. body is the answer
// posted in the thread, in the author's language.
func (r *Reviewer) applyFix(ctx context.Context, t *fixTarget) (outcome, sha, body string) {
	pick := t.lang.Pick
	cannot := func(why string) string {
		return pick("Kann ich nicht selbst umsetzen: ", "I can't apply this myself: ") + why
	}
	p := t.plan
	if p == nil || p.unavailable != "" {
		why := "unknown"
		if p != nil {
			why = p.unavailable
		}
		r.log.InfoContext(ctx, fmt.Sprintf("%s: fix on %s not asked for: %s", t.prTag, t.where, why))
		return store.FixNotFeasible, "", cannot(unavailableText(t.lang, why))
	}
	res := t.res
	if res != nil && res.Outcome == inference.OutcomeTooLarge {
		// Deterministic: asking again would fail the same way, and pay.
		r.log.InfoContext(ctx, fmt.Sprintf("%s: fix call on %s too large: %v", t.prTag, t.where, res.Err))
		return store.FixNotFeasible, "", cannot(unavailableText(t.lang, "too_large"))
	}
	if res == nil || res.Outcome != inference.OutcomeOK {
		if res != nil {
			r.log.WarnContext(ctx, fmt.Sprintf("%s: fix call on %s failed: %s %v", t.prTag, t.where, res.Outcome, res.Err))
		}
		return store.FixError, "", pick("Der Fix ist fehlgeschlagen, bitte nochmals versuchen.", "Writing the fix failed, please ask again.")
	}
	if !res.Fix.Feasible {
		why := sharpS(t.lang, strings.TrimSpace(res.Fix.Reason))
		if why == "" {
			why = pick("die Änderung geht über eine Stelle in einer Datei hinaus.", "the change goes beyond one spot in one file.")
		}
		return store.FixNotFeasible, "", cannot(lang.TruncateRunes(why, replyPostRunes/2))
	}
	content, reason := inference.ValidateFix(p.content, p.from, p.to, res.Fix)
	if reason != "" {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: fix on %s dropped: %s", t.prTag, t.where, reason))
		return store.FixRejected, "", pick("Kann ich nicht sicher selbst ändern. Bitte passe die Stelle selbst an.",
			"I can't change this safely myself. Please make the change yourself.")
	}
	subject, ok := inference.CheckCommitSubject(sharpS(t.lang, res.Fix.CommitSubject))
	if !ok {
		subject = pick("setze Review-Vorschlag in ", "apply review suggestion in ") + path.Base(p.path) + pick(" um", "")
	}
	message := subject
	if t.threadID != nil {
		message += "\n\n" + r.bitbucket.CommentURL(t.project, t.repo, t.prID, *t.threadID)
	}
	sha, err := r.bitbucket.EditFile(ctx, p.project, p.repo, p.branch, p.path, p.head, message, content)
	if err != nil {
		status := bitbucket.Status(err)
		r.log.WarnContext(ctx, fmt.Sprintf("%s: fix commit on %s refused: %v", t.prTag, t.where, err))
		switch {
		case status == http.StatusUnauthorized || status == http.StatusForbidden:
			return store.FixNoWrite, "", pick("Kann ich nicht selbst committen: mir fehlen die Schreibrechte auf diesen Branch.",
				"I can't commit this myself: I have no write permission on this branch.")
		case status == http.StatusConflict:
			return store.FixConflict, "", pick("Der Branch hat sich inzwischen bewegt. Bitte nochmals fragen.",
				"The branch has moved in the meantime. Please ask again.")
		case status >= 400 && status < 500:
			return store.FixRefused, "", pick("Bitbucket hat den Commit abgelehnt.", "Bitbucket refused the commit.")
		}
		return store.FixError, "", pick("Der Commit ist fehlgeschlagen, bitte nochmals versuchen.", "The commit failed, please ask again.")
	}
	link := r.bitbucket.CommitURL(p.project, p.repo, sha)
	body = pick("Umgesetzt in ", "Applied in ") + "[`" + render.ShortSHA(sha, 8) + "`](" + link + "):\n\n> " + subject + "\n\n" +
		pick("Bitte prüfe die Anpassung genau, bevor du mergst.", "Please review the change carefully before you merge.")
	return store.FixCommitted, sha, body
}

// unavailableText is why a fix could not even be asked for, for the author.
func unavailableText(l lang.Lang, why string) string {
	switch why {
	case "binary":
		return l.Pick("die Datei ist keine Textdatei.", "the file is not a text file.")
	case "too_large":
		return l.Pick("die Datei oder die Projektrichtlinien sind zu gross.", "the file or the project guidelines are too large.")
	case "line_gone":
		return l.Pick("die Stelle gibt es im aktuellen Stand nicht mehr.", "that line is no longer in the current code.")
	case "no_anchor":
		return l.Pick("die Frage hängt an keiner Code-Zeile.", "the question is not on a line of code.")
	case "state_unread":
		return l.Pick("ob das schon umgesetzt ist, liess sich gerade nicht lesen. Bitte nochmals versuchen.",
			"whether this was applied already could not be read just now. Please ask again.")
	}
	return l.Pick("die Datei liess sich nicht lesen.", "the file could not be read.")
}

// postFixAnswer posts the answer under the author's reply. On a finding's
// thread its id is kept with the finding, so a reply under it is a reply on
// the finding again; after a Q&A it is kept as a reply about the answer,
// so "apply it" under it applies the answer (a second commit of it is
// "already applied"). A failed post after a commit is an error: the commit
// stands unannounced.
func (r *Reviewer) postFixAnswer(ctx context.Context, t *fixTarget, sha, body string) bool {
	body = t.prefix + lang.TruncateRunes(body, replyPostRunes)
	posted, err := r.bitbucket.ReplyToComment(ctx, t.project, t.repo, t.prID, t.replyTo, body)
	if err != nil {
		if sha != "" {
			r.log.ErrorContext(ctx, fmt.Sprintf("%s: fix on %s committed as %s, but its confirmation failed to post: %v", t.prTag, t.where, sha, err))
			return false
		}
		r.log.ErrorContext(ctx, fmt.Sprintf("%s: answer to a fix request on %s failed to post: %v", t.prTag, t.where, err))
		return false
	}
	if t.findingID != 0 {
		r.recordFindingBotReply(ctx, t.key, posted, t.findingID)
	} else {
		// About the answer: "apply it" under this reply (asked again after
		// a moved branch, say) applies the answer, never this reply.
		safeDBErr(ctx, r.log, "InsertAnswerBotReply", func() error {
			return r.store.InsertAnswerBotReply(ctx, t.key, posted, t.anchor, t.answerID)
		})
	}
	return true
}
