package review

import (
	"context"
	"fmt"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/store"
)

// confirmCompliance is the ticket check the summary shows: only the
// requirements a quoted line proves implemented (inference.ConfirmCompliance).
// Every one left out gets a log line and its reason on the run page. nil in,
// nil out: an unparseable answer stays "extraction failed", not "none
// proven".
func (r *Reviewer) confirmCompliance(ctx context.Context, plan *reviewPlan, reqs []inference.ComplianceRequirement) []inference.ComplianceRequirement {
	if reqs == nil {
		return nil
	}
	// No ticket or the check off: the summary shows no list, so there is
	// nothing to confirm, log or record.
	if plan.ticket == nil || !r.cfg.TicketComplianceCheck {
		return reqs
	}
	kept, checks := inference.ConfirmCompliance(reqs, plan.anchors, plan.history)
	for _, c := range checks {
		req := c.Requirement
		// Never a secret in the logs or on the run page; the decision
		// still shows.
		secret := inference.AnySecret(req.Requirement, req.File, req.Quote)
		if c.Drop != "" {
			what := fmt.Sprintf("%q", lang.TruncateRunes(req.Requirement, 120))
			if secret {
				what = "(withheld: secret-like text)"
			}
			r.log.InfoContext(ctx, fmt.Sprintf("%s: ticket requirement %s left out of the summary: %s",
				plan.prTag, what, c.Drop))
		}
		rec := store.ComplianceCheck{Requirement: req.Requirement, File: req.File, Quote: req.Quote, Drop: string(c.Drop)}
		if secret {
			rec.Requirement, rec.File, rec.Quote = "", "", ""
		}
		plan.detail.Compliance = append(plan.detail.Compliance, rec)
	}
	return kept
}
