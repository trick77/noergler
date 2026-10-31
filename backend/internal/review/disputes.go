package review

import (
	"context"
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/store"
)

// Dispute signals: a PR author's dispute of a finding, with a project-wide
// reason, is one occurrence of a signal of the repo. Consulted on the
// finding-reply path only (HandleFindingReply); a review never reads them,
// and a signal never drops or softens a finding. Its only way into a review
// is a human pasting the proposed bullet into AGENTS.md.

const (
	// disputeSignalsListed caps the open signals the reply prompt lists.
	disputeSignalsListed = 20
	// disputePRsLinked caps the earlier PRs the statement links.
	disputePRsLinked = 10
)

// disputeSignalsOn is false only for a Reviewer built without the knobs;
// config always sets them inside their ranges.
func (r *Reviewer) disputeSignalsOn() bool {
	return r.cfg.DisputeSignalWindowDays > 0 && r.cfg.DisputeSignalThreshold > 1
}

// signalOccurrence is this dispute's place in its signal: the signal (0 for
// a new one), the bullet it records, the distinct PRs in the window with
// this one, the earlier ones and the distinct authors.
type signalOccurrence struct {
	id      int64
	rule    string
	prs     int
	earlier []store.DisputePR
	authors int
}

// countDisputeSignal maps the model's match onto the list the prompt showed
// and counts the signal's PRs in the window, this PR included. A match off
// the list is a new signal; a failed read counts this PR alone, so the
// statement is never posted on a guess.
func (r *Reviewer) countDisputeSignal(ctx context.Context, j *replyJob, match int, rule string) *signalOccurrence {
	o := &signalOccurrence{rule: rule, prs: 1, authors: 1}
	switch {
	case match == 0:
		return o
	case match > len(j.signalIDs):
		r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on finding %s names dispute signal [%d] of %d listed: counted as a new signal",
			j.prTag, j.where, match, len(j.signalIDs)))
		return o
	}
	o.id = j.signalIDs[match-1]
	o.earlier = safeDB(ctx, r.log, "DisputeSignalPRs", func() ([]store.DisputePR, error) {
		return r.store.DisputeSignalPRs(ctx, o.id, r.cfg.DisputeSignalWindowDays, j.key)
	})
	o.prs += len(o.earlier)
	seen := map[string]bool{strings.ToLower(j.author): true}
	for _, p := range o.earlier {
		if a := strings.ToLower(p.Author); a != "" && !seen[a] {
			seen[a] = true
			o.authors++
		}
	}
	return o
}

// logState is the AGENTS.md part of the reply's log line.
func (o *signalOccurrence) logState(threshold, windowDays int) string {
	sig := "new signal"
	if o.id != 0 {
		sig = fmt.Sprintf("signal %d", o.id)
	}
	state := "held"
	if o.prs >= threshold {
		state = "proposed"
	}
	return fmt.Sprintf("%s (%s, %d/%d PRs in %dd, %d authors)", state, sig, o.prs, threshold, windowDays, o.authors)
}

// disputeStatement is what a dispute at the threshold adds to the answer:
// how often the reason came back, where, and the bullet that would stop it.
func (r *Reviewer) disputeStatement(j *replyJob, o *signalOccurrence, line string) string {
	links := make([]string, 0, disputePRsLinked)
	for i, p := range o.earlier {
		if i == disputePRsLinked {
			break
		}
		ref := fmt.Sprintf("#%d", p.PRID)
		if p.CommentID != nil {
			ref = "[" + ref + "](" + r.bitbucket.CommentURL(j.project, j.repo, p.PRID, *p.CommentID) + ")"
		}
		links = append(links, ref)
	}
	if more := len(o.earlier) - len(links); more > 0 {
		links = append(links, j.lang.Pick(fmt.Sprintf("+%d weitere", more), fmt.Sprintf("+%d more", more)))
	}
	authors := j.lang.Pick(fmt.Sprintf("%d Verfasser", o.authors), plural(o.authors, "author", "authors"))
	w := r.cfg.DisputeSignalWindowDays
	list := strings.Join(links, ", ")
	if list != "" {
		list += "; "
	}
	head := j.lang.Pick(
		fmt.Sprintf("Das ist der %d. PR in den letzten %d Tagen mit diesem Einwand (%s%s). Vorschlag für `AGENTS.md`, damit das künftig nicht mehr bemängelt wird:",
			o.prs, w, list, authors),
		fmt.Sprintf("This is the %s PR in the last %d days disputing this for the same reason (%s%s). Proposed `AGENTS.md` change, so it is not raised again:",
			ordinalEN(o.prs), w, list, authors))
	fence := codeFence(line)
	return "\n\n" + head + "\n\n" + fence + "markdown\n" + line + "\n" + fence
}

func plural(n int, one, many string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, one)
	}
	return fmt.Sprintf("%d %s", n, many)
}

// ordinalEN is 1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st.
func ordinalEN(n int) string {
	suffix := "th"
	if n%100 < 11 || n%100 > 13 {
		switch n % 10 {
		case 1:
			suffix = "st"
		case 2:
			suffix = "nd"
		case 3:
			suffix = "rd"
		}
	}
	return fmt.Sprintf("%d%s", n, suffix)
}
