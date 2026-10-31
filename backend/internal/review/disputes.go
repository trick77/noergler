package review

import (
	"context"
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
)

// Dispute signals: a PR author's dispute of a finding, with a project-wide
// reason, is one occurrence of a signal of the repo. Consulted on the
// finding-reply path only (HandleFindingReply); a review never reads them,
// and a signal never drops or softens a finding. Its only way into a review
// is a human pasting the proposed bullet into AGENTS.md.
//
// Known limit: two disputes of the same new rule whose calls overlap both
// see no signal and start one each; later prompts list both and the count
// splits. Rare (the window is one gateway call), and a semantic merge would
// take a model call. Likewise only the disputeSignalsListed most recently
// disputed signals are listed: a repo with more open ones can start a
// duplicate of an older one.

const (
	// disputeSignalsListed caps the open signals the reply prompt lists.
	disputeSignalsListed = 20
	// disputePRsLinked caps the earlier PRs the statement links.
	disputePRsLinked = 10
)

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

// weighDispute decides what the model's verdict means for the repo's dispute
// signals: the AGENTS.md part of the log line, and the occurrence to count
// and record (nil for none). Only a dispute counts. A match is honoured only
// if it names the list the prompt showed. A matched rule AGENTS.md already
// states counts nothing and is logged: the review raised a finding AGENTS.md
// rules out. A merged bullet that fails the format check falls back to the
// listed wording, so a valid match still counts.
func (r *Reviewer) weighDispute(ctx context.Context, j *replyJob, rep inference.FindingReply) (string, *signalOccurrence) {
	line := rep.AgentsMDLine
	if rep.Intent != "disputes" {
		if line != "" {
			return "dropped (not a dispute)", nil
		}
		return "none", nil
	}
	match := rep.MatchesSignal
	if match > len(j.signalIDs) {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on finding %s names dispute signal [%d] of %d listed: none matched",
			j.prTag, j.where, match, len(j.signalIDs)))
		match = 0
	}
	ruledOut := func() {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on finding %s matches dispute signal %d, which AGENTS.md already states: the review raised a finding AGENTS.md rules out",
			j.prTag, j.where, j.signalIDs[match-1]))
	}
	if line == "" {
		if match > 0 {
			ruledOut()
		}
		return "none", nil
	}
	reason := r.checkBullet(j, line)
	switch {
	case reason == "already_present":
		if match > 0 {
			ruledOut()
		}
		return "dropped (already_present)", nil
	case reason != "" && match == 0:
		return "dropped (" + reason + ")", nil
	case reason != "":
		listed := j.signalRules[match-1]
		switch lr := r.checkBullet(j, listed); lr {
		case "":
			line = listed
		case "already_present":
			ruledOut()
			return "dropped (already_present)", nil
		default:
			return "dropped (" + reason + ")", nil
		}
	}
	// Posted and stored Swiss-spelled for a German answer, like the prose.
	line = sharpS(j.lang, line)
	o := r.countDisputeSignal(ctx, j, match, line)
	state := o.logState(r.cfg.DisputeSignalThreshold, r.cfg.DisputeSignalWindowDays)
	if reason != "" {
		state += ", listed wording (" + reason + ")"
	}
	return state, o
}

// checkBullet is CheckAgentsMDLine on the line as the model wrote it and,
// for a German answer, as it would be posted: AGENTS.md may be written with
// ß or with ss, and either spelling of the rule is already there.
func (r *Reviewer) checkBullet(j *replyJob, line string) string {
	reason := inference.CheckAgentsMDLine(line, j.agentsMD)
	if ss := sharpS(j.lang, line); reason == "" && ss != line {
		reason = inference.CheckAgentsMDLine(ss, j.agentsMD)
	}
	return reason
}

// countDisputeSignal counts the signal's PRs in the window, this PR
// included; match is 0 (a new signal) or a valid index into the list. A
// failed read counts this PR alone, so the statement is never posted on a
// guess.
func (r *Reviewer) countDisputeSignal(ctx context.Context, j *replyJob, match int, rule string) *signalOccurrence {
	o := &signalOccurrence{rule: rule, prs: 1, authors: 1}
	if match == 0 {
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
func (r *Reviewer) disputeStatement(j *replyJob, o *signalOccurrence) string {
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
	list := strings.Join(links, ", ")
	if list != "" {
		list += "; "
	}
	window := render.PluralIn(j.lang, r.cfg.DisputeSignalWindowDays, "Tag", "Tagen", "day", "days")
	head := j.lang.Pick(
		fmt.Sprintf("Das ist der %d. PR innert %s mit diesem Einwand (%s%d Verfasser). Vorschlag für `AGENTS.md`, damit das künftig nicht mehr bemängelt wird:",
			o.prs, window, list, o.authors),
		fmt.Sprintf("This is the %s PR within %s disputing this for the same reason (%s%s). Proposed `AGENTS.md` change, so it is not raised again:",
			ordinalEN(o.prs), window, list, render.Plural(o.authors, "author")))
	fence := codeFence(o.rule)
	return "\n\n" + head + "\n\n" + fence + "markdown\n" + o.rule + "\n" + fence
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
