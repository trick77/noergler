package onboarding

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"sync"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/config"
)

// repoHookConcurrency bounds the per-repo webhook listings in flight. One
// listing per repo and a project has dozens, so a few at a time keeps the
// request short without hammering Bitbucket.
const repoHookConcurrency = 4

// Options are the knobs the caller sets per request. The zero value matches
// Python's defaults: name "noergler", no dry run, no bot grant, prune on
// (hence NoPrune, not Prune).
type Options struct {
	// WebhookName defaults to DefaultWebhookName when empty.
	WebhookName string
	DryRun      bool
	// GrantBot: create the bot's read access when it is missing, instead of
	// skipping the target.
	GrantBot bool
	// NoPrune leaves stray repo-level hooks under a project target alone.
	NoPrune bool
}

// Onboarder holds one request's clients and settings.
type Onboarder struct {
	admin      AdminClient
	bot        BotClient
	team       *config.Team
	webhookURL string
	opts       Options
	log        *slog.Logger

	// instanceURL is webhookURL up to the last "/webhook/". Hooks are matched
	// by name AND instance: a hook named `noergler` that points at another
	// instance (intg next to prod) is reported, never pruned or rewritten.
	instanceURL string
}

// New builds an Onboarder. log may be nil.
func New(admin AdminClient, bot BotClient, team *config.Team, webhookURL string, opts Options, log *slog.Logger) *Onboarder {
	if opts.WebhookName == "" {
		opts.WebhookName = DefaultWebhookName
	}
	if log == nil {
		log = slog.New(slog.DiscardHandler)
	}
	return &Onboarder{
		admin: admin, bot: bot, team: team, webhookURL: webhookURL,
		opts: opts, log: log, instanceURL: instanceURL(webhookURL),
	}
}

// instanceURL cuts a webhook URL at its last "/webhook/", mirroring Python's
// rsplit("/webhook/", 1)[0]: without the separator the URL is unchanged.
func instanceURL(webhookURL string) string {
	if i := strings.LastIndex(webhookURL, "/webhook/"); i >= 0 {
		return webhookURL[:i]
	}
	return webhookURL
}

// WebhookName is the name this Onboarder matches and writes.
func (o *Onboarder) WebhookName() string { return o.opts.WebhookName }

// -- claim and access -- //

// claim decides whether the team holds the target and, when it does, whether
// the bot can read it.
//
// The bot check swallows every error — 401, 403, 404, 500, a timeout, DNS,
// TLS: all of them mean "cannot read", and none of them is the caller's fault.
// This is the exact opposite of HasAdmin, which only forgives 401/403.
func (o *Onboarder) claim(ctx context.Context, target Target) Claim {
	kind := claimKind(o.team, target.Project)
	owned := kind == "whole"
	if target.Repo != "" {
		owned = o.team.Owns(target.Project, target.Repo)
	}
	botCanRead := false
	if owned {
		var err error
		if target.Repo == "" {
			_, err = o.bot.GetProject(ctx, target.Project)
		} else {
			_, err = o.bot.GetRepo(ctx, target.Project, target.Repo)
		}
		if err != nil {
			o.log.InfoContext(ctx, fmt.Sprintf("bot cannot read %s: %s", target.Key(), err))
		} else {
			botCanRead = true
		}
	}
	return Claim{Owned: owned, Kind: kind, BotCanRead: botCanRead}
}

// isOurs reports whether a hook belongs to this noergler instance. The
// trailing slash is load-bearing: without it
// https://n.example.com.evil/webhook/x matches https://n.example.com.
func (o *Onboarder) isOurs(hook map[string]any) bool {
	url, ok := hook["url"].(string)
	return ok && strings.HasPrefix(url, o.instanceURL+"/")
}

// hookNamed finds the first hook with that name.
func hookNamed(hooks []map[string]any, name string) map[string]any {
	for _, h := range hooks {
		if n, ok := h["name"].(string); ok && n == name {
			return h
		}
	}
	return nil
}

// hookID reads a JSON-decoded id, which arrives as a float64.
func hookID(hook map[string]any) (int, error) {
	switch v := hook["id"].(type) {
	case float64:
		return int(v), nil
	case int:
		return v, nil
	}
	return 0, fmt.Errorf("webhook has no numeric id: %v", hook["id"])
}

// hookURL is a hook's url for a message, rendered the way Python's str() does
// (a missing url prints as None).
func hookURL(hook map[string]any) string {
	if u, ok := hook["url"].(string); ok {
		return u
	}
	return "None"
}

func (o *Onboarder) buildWebhookBody() bitbucket.Webhook {
	w := bitbucket.Webhook{
		Name:                    o.opts.WebhookName,
		URL:                     o.webhookURL,
		Active:                  true,
		Events:                  append([]string(nil), config.RequiredWebhookEvents...),
		SSLVerificationRequired: true,
	}
	w.Configuration.Secret = o.team.WebhookSecret
	return w
}

// pyRepr renders a string the way Python's repr() does for the messages that
// carry one: single quotes, with a backslash-escaped quote inside.
func pyRepr(s string) string {
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		return `"` + s + `"`
	}
	return "'" + strings.ReplaceAll(s, "'", `\'`) + "'"
}

// pyList renders a sorted string slice as a Python list literal.
func pyList(items []string) string {
	parts := make([]string, len(items))
	for i, s := range items {
		parts[i] = pyRepr(s)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// pyTruthy is Python's truth test over a JSON-decoded value, for the two
// `not existing.get(k, True)` checks and the `configuration` emptiness one.
func pyTruthy(v any) bool {
	switch t := v.(type) {
	case nil:
		return false
	case bool:
		return t
	case string:
		return t != ""
	case float64:
		return t != 0
	case []any:
		return len(t) > 0
	case map[string]any:
		return len(t) > 0
	}
	return true
}

// diffWebhook lists what an existing hook would have to change to match ours.
// The order is fixed and part of the user-facing text. An empty diff means the
// hook is up to date. ["create"] is a reserved sentinel, never produced here.
func (o *Onboarder) diffWebhook(existing map[string]any) []string {
	diffs := []string{}
	if u, ok := existing["url"].(string); !ok || u != o.webhookURL {
		var was string
		if ok {
			was = pyRepr(u)
		} else if existing["url"] == nil {
			was = "None"
		} else {
			was = fmt.Sprintf("%v", existing["url"])
		}
		diffs = append(diffs, fmt.Sprintf("url: %s -> %s", was, pyRepr(o.webhookURL)))
	}

	existingEvents := map[string]bool{}
	if evs, ok := existing["events"].([]any); ok {
		for _, e := range evs {
			if s, ok := e.(string); ok {
				existingEvents[s] = true
			}
		}
	}
	required := map[string]bool{}
	for _, e := range config.RequiredWebhookEvents {
		required[e] = true
	}
	if !sameSet(existingEvents, required) {
		var missing, extra []string
		for e := range required {
			if !existingEvents[e] {
				missing = append(missing, e)
			}
		}
		for e := range existingEvents {
			if !required[e] {
				extra = append(extra, e)
			}
		}
		sort.Strings(missing)
		sort.Strings(extra)
		diffs = append(diffs, fmt.Sprintf("events: missing=%s extra=%s", pyList(missing), pyList(extra)))
	}

	// Absent means True: Bitbucket omits neither, but a hand-made body might.
	if v, present := existing["active"]; present && !pyTruthy(v) {
		diffs = append(diffs, "active: False -> True")
	}
	if v, present := existing["sslVerificationRequired"]; present && !pyTruthy(v) {
		diffs = append(diffs, "sslVerificationRequired: False -> True")
	}
	// Bitbucket never returns the stored secret, so only its absence is
	// visible; a secret-only change needs remove + onboard.
	if !pyTruthy(existing["configuration"]) {
		diffs = append(diffs, "configuration.secret: (unset) -> (set)")
	}
	return diffs
}

func sameSet(a, b map[string]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

// -- building blocks -- //

// UpsertWebhook creates or updates the webhook on the target and returns its
// id with the diff that was applied. A fresh hook comes back as ["create"];
// on a dry run its id is -1.
func (o *Onboarder) UpsertWebhook(ctx context.Context, target Target) (int, []string, error) {
	hooks, err := o.admin.ListWebhooks(ctx, target.Project, target.Repo)
	if err != nil {
		return 0, nil, err
	}
	existing := hookNamed(hooks, o.opts.WebhookName)
	if existing != nil && !o.isOurs(existing) {
		return 0, nil, &ForeignHook{Msg: fmt.Sprintf(
			"%s hook points at another noergler (%s); "+
				"gone? remove it with that instance first, else use another name",
			pyRepr(o.opts.WebhookName), hookURL(existing))}
	}
	body := o.buildWebhookBody()
	if existing == nil {
		o.log.InfoContext(ctx, fmt.Sprintf("[%s] creating webhook %s", target.Key(), pyRepr(o.opts.WebhookName)))
		if o.opts.DryRun {
			return -1, []string{"create"}, nil
		}
		created, err := o.admin.CreateWebhook(ctx, target.Project, target.Repo, body)
		if err != nil {
			return 0, nil, err
		}
		return created.ID, []string{"create"}, nil
	}

	id, err := hookID(existing)
	if err != nil {
		return 0, nil, err
	}
	diff := o.diffWebhook(existing)
	if len(diff) == 0 {
		o.log.InfoContext(ctx, fmt.Sprintf("[%s] webhook already up to date", target.Key()))
		return id, diff, nil
	}
	o.log.InfoContext(ctx, fmt.Sprintf("[%s] updating webhook id=%d changes=%s", target.Key(), id, pyList(diff)))
	if !o.opts.DryRun {
		if _, err := o.admin.UpdateWebhook(ctx, target.Project, target.Repo, id, body); err != nil {
			return 0, nil, err
		}
	}
	return id, diff, nil
}

// StrayHook is one of this instance's repo-level hooks under a project target.
type StrayHook struct {
	Target Target
	ID     int
}

// StrayRepoHooks lists this instance's repo-level hooks under a project, and
// the same-named foreign ones. With a project webhook in place the former
// deliver every event a second time; the latter belong to another noergler and
// are left alone.
//
// Both lists come back in the order ListRepos returned the repos: a map would
// randomise the status table and the pruned list. A repo whose "slug" is
// missing or not a string is silently skipped.
func (o *Onboarder) StrayRepoHooks(ctx context.Context, project string) ([]StrayHook, []string, error) {
	repos, err := o.admin.ListRepos(ctx, project)
	if err != nil {
		return nil, nil, err
	}
	var slugs []string
	for _, r := range repos {
		if s, ok := r["slug"].(string); ok {
			slugs = append(slugs, s)
		}
	}

	found := make([][]map[string]any, len(slugs))
	errs := make([]error, len(slugs))
	sem := make(chan struct{}, repoHookConcurrency)
	var wg sync.WaitGroup
	for i, slug := range slugs {
		wg.Add(1)
		go func(i int, slug string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			found[i], errs[i] = o.admin.ListWebhooks(ctx, project, slug)
		}(i, slug)
	}
	wg.Wait()

	// First error in listing order wins, as asyncio.gather's does.
	for _, err := range errs {
		if err != nil {
			return nil, nil, err
		}
	}

	var stray []StrayHook
	var foreign []string
	for i, slug := range slugs {
		hook := hookNamed(found[i], o.opts.WebhookName)
		if hook == nil {
			continue
		}
		repoTarget := Target{Project: project, Repo: slug}
		if o.isOurs(hook) {
			id, err := hookID(hook)
			if err != nil {
				return nil, nil, err
			}
			stray = append(stray, StrayHook{Target: repoTarget, ID: id})
		} else {
			foreign = append(foreign, repoTarget.Key()+" -> "+hookURL(hook))
		}
	}
	return stray, foreign, nil
}

// PruneRepoHooks deletes this instance's stray repo-level hooks under a
// project, one at a time and not transactionally: a failure halfway leaves the
// earlier deletions done. Foreign hooks are only warned about; the list is
// dropped, since only Status surfaces them.
func (o *Onboarder) PruneRepoHooks(ctx context.Context, project string) ([]string, error) {
	stray, foreign, err := o.StrayRepoHooks(ctx, project)
	if err != nil {
		return nil, err
	}
	for _, entry := range foreign {
		o.log.WarnContext(ctx, fmt.Sprintf(
			"[%s] hook named %s points at another noergler, left alone", entry, pyRepr(o.opts.WebhookName)))
	}
	pruned := []string{}
	for _, s := range stray {
		if !o.opts.DryRun {
			if err := o.admin.DeleteWebhook(ctx, project, s.Target.Repo, s.ID); err != nil {
				return nil, err
			}
			o.log.InfoContext(ctx, fmt.Sprintf("[%s] deleted stray repo webhook id=%d", s.Target.Key(), s.ID))
		}
		pruned = append(pruned, s.Target.Key())
	}
	return pruned, nil
}

// refuseRepoTarget returns why a repo target is refused, or "" when it is not.
// A repo hook is refused when it would sit next to a project webhook of this
// instance: every event would then be delivered twice (see PruneRepoHooks).
//
// No project admin on a shared project (401/403) is not a refusal: there is
// nothing more to check. Any other failure is returned, because a Bitbucket
// hiccup is not a licence to double-hook.
func (o *Onboarder) refuseRepoTarget(ctx context.Context, target Target, claim Claim) (string, error) {
	if target.IsProject() {
		return "", nil
	}
	if claim.Kind == "whole" {
		return wholeClaimReason(target), nil
	}
	hooks, err := o.admin.ListWebhooks(ctx, target.Project, "")
	if err != nil {
		if s := bitbucket.Status(err); s == http.StatusUnauthorized || s == http.StatusForbidden {
			return "", nil
		}
		return "", err
	}
	projectHook := hookNamed(hooks, o.opts.WebhookName)
	if projectHook != nil && o.isOurs(projectHook) {
		id := "None"
		if n, err := hookID(projectHook); err == nil {
			id = fmt.Sprintf("%d", n)
		}
		return fmt.Sprintf(
			"%s still has this instance's project webhook (id=%s), "+
				"which delivers every event as well; remove it first", target.Project, id), nil
	}
	return "", nil
}
