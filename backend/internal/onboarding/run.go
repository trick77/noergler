package onboarding

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
)

// httpDetailCap bounds how much of a failing response body reaches a row's
// detail. Counted in RUNES, not bytes, so a multi-byte body is not cut
// mid-character.
const httpDetailCap = 200

// httpDetail renders a Bitbucket failure as "HTTP <status>: <capped body>".
// It is only ever called with an error that carries a status.
func httpDetail(err error) string {
	var se *bitbucket.StatusError
	if !errors.As(err, &se) {
		return err.Error()
	}
	return fmt.Sprintf("HTTP %d: %s", se.Status, truncateRunes(se.Body, httpDetailCap))
}

func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

// Status reports one target: owned, readable by the bot, and the state of its
// webhook. It only returns an error for a transport failure in the
// project-hook guard; Run turns that into a row.
func (o *Onboarder) Status(ctx context.Context, target Target) (StatusRow, error) {
	claim := o.claim(ctx, target)
	stray := []string{}
	foreign := []string{}
	if !claim.Owned {
		return StatusRow{target, false, false, notOwnedReason(target, claim), stray, foreign}, nil
	}
	refused, err := o.refuseRepoTarget(ctx, target, claim)
	if err != nil {
		if s := bitbucket.Status(err); s != 0 {
			return StatusRow{target, true, claim.BotCanRead,
				fmt.Sprintf("project hook check HTTP %d", s), stray, foreign}, nil
		}
		// A status error is a verdict, above. A transport failure is not: it
		// escapes to Run, which files it as an error row.
		return StatusRow{}, err
	}
	if refused != "" {
		return StatusRow{target, true, claim.BotCanRead, "blocked: " + refused, stray, foreign}, nil
	}

	var webhook string
	hooks, err := o.admin.ListWebhooks(ctx, target.Project, target.Repo)
	switch {
	case err != nil && bitbucket.Status(err) != 0:
		webhook = fmt.Sprintf("HTTP %d", bitbucket.Status(err))
	case err != nil:
		webhook = fmt.Sprintf("error: %v", err)
	default:
		existing := hookNamed(hooks, o.opts.WebhookName)
		switch {
		case existing == nil:
			webhook = "missing"
		case !o.isOurs(existing):
			webhook = "foreign: " + hookURL(existing)
		default:
			diff := o.diffWebhook(existing)
			if len(diff) == 0 {
				webhook = "ok"
			} else {
				webhook = "stale: " + strings.Join(diff, "; ")
			}
		}
	}

	if target.IsProject() && !strings.HasPrefix(webhook, "HTTP ") && !strings.HasPrefix(webhook, "error: ") {
		// The hook verdict above stands on its own; a failure here only means
		// the repo-level check is unknown, which is not healthy either.
		found, fgn, err := o.StrayRepoHooks(ctx, target.Project)
		switch {
		case err != nil && bitbucket.Status(err) != 0:
			webhook += fmt.Sprintf(" (repo hooks unchecked: HTTP %d)", bitbucket.Status(err))
		case err != nil:
			webhook += fmt.Sprintf(" (repo hooks unchecked: %v)", err)
		default:
			foreign = fgn
			if foreign == nil {
				foreign = []string{}
			}
			stray = stray[:0]
			for _, s := range found {
				stray = append(stray, s.Target.Key())
			}
		}
	}
	return StatusRow{target, claim.Owned, claim.BotCanRead, webhook, stray, foreign}, nil
}

// Onboard puts the webhook on one target, granting the bot read access first
// when asked to. It only returns an error for a transport failure in the
// project-hook guard or in the grant.
func (o *Onboarder) Onboard(ctx context.Context, target Target) (TargetResult, error) {
	claim := o.claim(ctx, target)
	if !claim.Owned {
		return TargetResult{target, "skipped", notOwnedReason(target, claim), []string{}}, nil
	}
	refused, err := o.refuseRepoTarget(ctx, target, claim)
	if err != nil {
		if bitbucket.Status(err) != 0 {
			return TargetResult{target, "failed", "project hook check " + httpDetail(err), []string{}}, nil
		}
		return TargetResult{}, err
	}
	if refused != "" {
		return TargetResult{target, "skipped", refused, []string{}}, nil
	}

	var notes []string
	bot := o.bot.BotUsername()
	if !claim.BotCanRead {
		if !o.opts.GrantBot {
			return TargetResult{target, "skipped", fmt.Sprintf(
				"%s cannot read it; grant %s %s in Bitbucket or run grant-bot",
				bot, bot, target.BotPermission()), []string{}}, nil
		}
		if !o.opts.DryRun {
			if err := o.admin.GrantUserPermission(ctx, target.Project, target.Repo, bot, target.BotPermission()); err != nil {
				if bitbucket.Status(err) != 0 {
					return TargetResult{target, "failed", fmt.Sprintf(
						"grant %s to %s: %s", target.BotPermission(), bot, httpDetail(err)), []string{}}, nil
				}
				return TargetResult{}, err
			}
			o.log.InfoContext(ctx, fmt.Sprintf("[%s] granted %s %s", target.Key(), bot, target.BotPermission()))
		}
		notes = append(notes, fmt.Sprintf("%s granted %s", bot, target.BotPermission()))
	}

	webhookID, diff, err := o.UpsertWebhook(ctx, target)
	if err != nil {
		var fh *ForeignHook
		switch {
		case errors.As(err, &fh):
			return TargetResult{target, "failed", fh.Msg, []string{}}, nil
		case bitbucket.Status(err) != 0:
			return TargetResult{target, "failed", "upsert webhook " + httpDetail(err), []string{}}, nil
		default:
			return TargetResult{target, "failed", fmt.Sprintf("upsert webhook: %v", err), []string{}}, nil
		}
	}

	switch {
	case len(diff) == 1 && diff[0] == "create":
		notes = append([]string{fmt.Sprintf("webhook created (id=%d)", webhookID)}, notes...)
	case len(diff) > 0:
		notes = append([]string{fmt.Sprintf("webhook updated (id=%d)", webhookID)}, notes...)
	default:
		notes = append([]string{"webhook already up to date"}, notes...)
	}

	if target.IsProject() && !o.opts.NoPrune {
		pruned, err := o.PruneRepoHooks(ctx, target.Project)
		if err != nil {
			notes = append(notes, fmt.Sprintf("prune of repo hooks failed: %v", err))
		} else if len(pruned) > 0 {
			notes = append(notes, fmt.Sprintf(
				"pruned %d repo hook(s): %s", len(pruned), strings.Join(pruned, ", ")))
		}
	}

	prefix := ""
	if o.opts.DryRun {
		prefix = "dry-run: "
	}
	return TargetResult{target, "ok", prefix + strings.Join(notes, ", "), diff}, nil
}

// Remove deletes this instance's webhook from the target. A no-op if absent,
// and a foreign hook is left alone. Nothing escapes as an error here.
func (o *Onboarder) Remove(ctx context.Context, target Target) (TargetResult, error) {
	hooks, err := o.admin.ListWebhooks(ctx, target.Project, target.Repo)
	if err != nil {
		if bitbucket.Status(err) != 0 {
			return TargetResult{target, "failed", "list webhooks " + httpDetail(err), []string{}}, nil
		}
		return TargetResult{target, "failed", fmt.Sprintf("list webhooks: %v", err), []string{}}, nil
	}
	existing := hookNamed(hooks, o.opts.WebhookName)
	if existing == nil {
		return TargetResult{target, "skipped",
			fmt.Sprintf("no %s webhook found", quoted(o.opts.WebhookName)), []string{}}, nil
	}
	if !o.isOurs(existing) {
		return TargetResult{target, "skipped", fmt.Sprintf(
			"%s webhook points at another noergler (%s), left alone",
			quoted(o.opts.WebhookName), hookURL(existing)), []string{}}, nil
	}
	webhookID, err := hookID(existing)
	if err != nil {
		return TargetResult{}, err
	}
	if o.opts.DryRun {
		return TargetResult{target, "ok",
			fmt.Sprintf("dry-run: would remove webhook id=%d", webhookID), []string{}}, nil
	}
	if err := o.admin.DeleteWebhook(ctx, target.Project, target.Repo, webhookID); err != nil {
		if bitbucket.Status(err) != 0 {
			return TargetResult{target, "failed", "delete webhook " + httpDetail(err), []string{}}, nil
		}
		return TargetResult{target, "failed", fmt.Sprintf("delete webhook: %v", err), []string{}}, nil
	}
	o.log.InfoContext(ctx, fmt.Sprintf("[%s] removed webhook id=%d", target.Key(), webhookID))
	return TargetResult{target, "ok", fmt.Sprintf("webhook removed: id=%d", webhookID), []string{}}, nil
}

// Run applies action to every target in order, one failure never aborting the
// rest. It returns the rows (StatusRow for "status", TargetResult otherwise),
// the rendered table and whether the result is healthy.
//
// Run is sequential: a whole project's repos can be onboarded in one request
// and Bitbucket sees one write at a time.
func Run(ctx context.Context, o *Onboarder, action Action, targets []Target) (rows []StatusRow, results []TargetResult, text string, healthy bool) {
	if action == ActionStatus {
		rows = make([]StatusRow, 0, len(targets))
		for _, target := range targets {
			row, err := o.Status(ctx, target)
			if err != nil {
				o.log.ErrorContext(ctx, fmt.Sprintf("[%s] unexpected error: %v", target.Key(), err))
				// The error row loses the claim: owned and bot both come back
				// false even when the target was owned.
				row = StatusRow{target, false, false, fmt.Sprintf("error: %v", err), []string{}, []string{}}
			}
			rows = append(rows, row)
		}
		return rows, nil, RenderStatus(rows), StatusHealthy(rows)
	}

	step := o.Onboard
	if action == ActionRemove {
		step = o.Remove
	}
	results = make([]TargetResult, 0, len(targets))
	for _, target := range targets {
		res, err := step(ctx, target)
		if err != nil {
			o.log.ErrorContext(ctx, fmt.Sprintf("[%s] unexpected error: %v", target.Key(), err))
			res = TargetResult{target, "failed", fmt.Sprintf("unexpected error: %v", err), []string{}}
		}
		results = append(results, res)
	}
	return nil, results, RenderResults(results), ResultsHealthy(results)
}
