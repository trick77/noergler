package teams

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/secretbox"
	"github.com/trick77/noergler/internal/store"
)

// A team may push its own Bitbucket and Jira accounts (PUT
// /teams/{team}/credentials). Without one it uses the shared account from the
// instance env. Its own Bitbucket token is its own bot: comments post as that
// account and the team mentions @<its username>, not the shared bot.
//
// Fault split: the instance key is shared layer, a key that does not match
// the stored key check aborts boot. Everything about one team's own
// credentials (a row that does not decrypt, a revoked token, a username
// Bitbucket disagrees with) is that team's alone and never disables it: the
// team stays up, reports the state, and can push a working token without an
// admin. A team that could not reach its own Bitbucket account has its
// webhooks acknowledged and not processed (Runtime.BitbucketBlocked).

// Kind is which account a credential is for.
type Kind string

// The two kinds, as they appear in the route path and the kind column.
const (
	KindBitbucket Kind = "bitbucket"
	KindJira      Kind = "jira"
)

// ParseKind reads a kind off a request path.
func ParseKind(s string) (Kind, bool) {
	switch Kind(s) {
	case KindBitbucket, KindJira:
		return Kind(s), true
	}
	return "", false
}

// Status is the health of a team's own account. A fixed set: it is what the
// unauthenticated dashboard shows, so it never carries a reason's text (which
// can name an env var or echo an upstream body); that stays in the logs.
type Status string

// The whole set; the API, the dashboard and the UI know no other.
const (
	StatusOK               Status = "ok"
	StatusRejected         Status = "rejected"
	StatusUnreachable      Status = "unreachable"
	StatusUndecryptable    Status = "undecryptable"
	StatusUsernameMismatch Status = "username_mismatch"
)

// Account is one kind of a team's account as the API and the dashboard show
// it. Never the token.
type Account struct {
	// Own is true for a pushed account, false for the shared one.
	Own bool
	// Username is the team's own bot (its @mention handle); Bitbucket only,
	// empty for the shared account.
	Username string
	// Status and Since are set for an own account only.
	Status Status
	Since  time.Time
}

// blocking reports whether the account cannot be worked with at all. Not
// unreachable: that is Bitbucket being down, which the next call may fix.
func (a Account) blocking() bool {
	return a.Own && (a.Status == StatusRejected || a.Status == StatusUndecryptable || a.Status == StatusUsernameMismatch)
}

// CredentialStore is the credential half of the store.
type CredentialStore interface {
	ListCredentials(ctx context.Context) ([]store.SealedCredential, error)
	PutCredentials(ctx context.Context, creds []store.SealedCredential, updatedBy string) error
	DeleteCredential(ctx context.Context, team, kind, updatedBy string) (bool, error)
	KeyCheck(ctx context.Context) (nonce, ciphertext []byte, found bool, err error)
	PutKeyCheck(ctx context.Context, nonce, ciphertext []byte) error
}

// Opened is a stored credential after decryption. Undecryptable means the row
// did not open under a key that passed the key check: a damaged row, or one
// copied from another team.
type Opened struct {
	Username      string
	Token         string
	Undecryptable bool
}

const keyCheckPlain = "noergler secrets key check"

var keyCheckAAD = []byte("secrets_key_check")

func credentialAAD(slug string, kind Kind) []byte { return []byte(slug + "|" + string(kind)) }

// OpenCredentials checks the instance key and opens the stored credentials of
// the known teams (enabled or disabled in teams.yaml). A row of a team no
// longer configured is skipped and decides nothing.
//
// An error is a shared-layer fault and aborts boot: the store unreadable, a
// key that does not open the key check while known teams have credentials,
// or no key check at all and a key that opens none of them. A key check that
// does not open with nothing of a known team stored is reset, since nothing
// is lost. A row that does not open under a key that passed is that team's
// fault: it comes back Undecryptable.
func OpenCredentials(ctx context.Context, st CredentialStore, box *secretbox.Box, known map[string]bool, log *slog.Logger) (map[string]map[Kind]Opened, error) {
	all, err := st.ListCredentials(ctx)
	if err != nil {
		return nil, fmt.Errorf("team credentials unreadable: %w", err)
	}
	var rows []store.SealedCredential
	for _, row := range all {
		if _, ok := ParseKind(row.Kind); ok && known[row.Team] {
			rows = append(rows, row)
			continue
		}
		log.WarnContext(ctx, fmt.Sprintf("team credentials for an unknown team ignored team=%s kind=%s", row.Team, row.Kind))
	}

	out := map[string]map[Kind]Opened{}
	opened := 0
	for _, row := range rows {
		kind, _ := ParseKind(row.Kind)
		o := Opened{Username: row.Username}
		if token, err := box.Open(row.Nonce, row.Ciphertext, credentialAAD(row.Team, kind)); err != nil {
			o.Undecryptable = true
		} else {
			o.Token = string(token)
			opened++
		}
		if out[row.Team] == nil {
			out[row.Team] = map[Kind]Opened{}
		}
		out[row.Team][kind] = o
	}

	nonce, ct, found, err := st.KeyCheck(ctx)
	if err != nil {
		return nil, fmt.Errorf("secrets key check unreadable: %w", err)
	}
	switch {
	case found:
		if plain, err := box.Open(nonce, ct, keyCheckAAD); err == nil && string(plain) == keyCheckPlain {
			return out, nil
		}
		if len(rows) > 0 {
			return nil, fmt.Errorf("NOERGLER_SECRETS_KEY does not match the key the %d stored team credential(s) were sealed with", len(rows))
		}
		log.WarnContext(ctx, "NOERGLER_SECRETS_KEY changed; no team credentials are stored, the key check is reset")
	case len(rows) > 0 && opened == 0:
		// Credentials but no key check (a partial restore): the rows are the
		// only witness, and not one of them opens.
		return nil, fmt.Errorf("NOERGLER_SECRETS_KEY opens none of the %d stored team credential(s), and no key check is stored", len(rows))
	}
	nonce, ct, err = box.Seal([]byte(keyCheckPlain), keyCheckAAD)
	if err != nil {
		return nil, err
	}
	if err := st.PutKeyCheck(ctx, nonce, ct); err != nil {
		return nil, fmt.Errorf("secrets key check not writable: %w", err)
	}
	return out, nil
}

// botSwitch is the team's Bitbucket client as the review pipeline and
// onboarding see it: the shared bot, or the team's own once it pushed one,
// swapped live.
//
// Each call reads the current client, so a push lands on the next call, not
// the next job. A review that straddles a push may post with both accounts;
// the summary edit falls back to a fresh comment when the old bot's is not
// the new one's to edit (review.shouldRepost).
type botSwitch struct {
	cur atomic.Pointer[bitbucket.Client]
}

func (b *botSwitch) c() *bitbucket.Client { return b.cur.Load() }

func (b *botSwitch) BotUsername() string { return b.c().BotUsername() }
func (b *botSwitch) CommentURL(project, repo string, prID, commentID int) string {
	return b.c().CommentURL(project, repo, prID, commentID)
}
func (b *botSwitch) FetchPRDiff(ctx context.Context, project, repo string, prID, contextLines int) (string, error) {
	return b.c().FetchPRDiff(ctx, project, repo, prID, contextLines)
}
func (b *botSwitch) FetchPRChanges(ctx context.Context, project, repo string, prID int) ([]string, error) {
	return b.c().FetchPRChanges(ctx, project, repo, prID)
}
func (b *botSwitch) FetchCommitDiff(ctx context.Context, project, repo string, prID int, fromCommit, toCommit string) (string, error) {
	return b.c().FetchCommitDiff(ctx, project, repo, prID, fromCommit, toCommit)
}
func (b *botSwitch) FetchFileContent(ctx context.Context, project, repo, commit, path string) (string, error) {
	return b.c().FetchFileContent(ctx, project, repo, commit, path)
}
func (b *botSwitch) PostInlineComment(ctx context.Context, project, repo string, prID int, file string, line int, body string) (int, error) {
	return b.c().PostInlineComment(ctx, project, repo, prID, file, line, body)
}
func (b *botSwitch) PostPRComment(ctx context.Context, project, repo string, prID int, text string) (int, int, error) {
	return b.c().PostPRComment(ctx, project, repo, prID, text)
}
func (b *botSwitch) ReplyToComment(ctx context.Context, project, repo string, prID, parentCommentID int, text string) (int, error) {
	return b.c().ReplyToComment(ctx, project, repo, prID, parentCommentID, text)
}
func (b *botSwitch) FetchPRComment(ctx context.Context, project, repo string, prID, commentID int) (*bitbucket.Comment, error) {
	return b.c().FetchPRComment(ctx, project, repo, prID, commentID)
}
func (b *botSwitch) UpdatePRComment(ctx context.Context, project, repo string, prID, commentID, version int, text string) (int, error) {
	return b.c().UpdatePRComment(ctx, project, repo, prID, commentID, version, text)
}
func (b *botSwitch) ResolveCommentThread(ctx context.Context, project, repo string, prID, commentID int) error {
	return b.c().ResolveCommentThread(ctx, project, repo, prID, commentID)
}
func (b *botSwitch) GetProject(ctx context.Context, project string) error {
	return b.c().GetProject(ctx, project)
}
func (b *botSwitch) GetRepo(ctx context.Context, project, repo string) error {
	return b.c().GetRepo(ctx, project, repo)
}

// jiraSwitch is the team's Jira client, swapped live like botSwitch.
type jiraSwitch struct{ cur atomic.Pointer[jira.Client] }

var errNoJira = errors.New("no usable Jira account")

func (j *jiraSwitch) FetchTicketWithParent(ctx context.Context, key string) (*jira.Ticket, *jira.Ticket, error) {
	c := j.cur.Load()
	if c == nil {
		return nil, nil, errNoJira
	}
	return c.FetchTicketWithParent(ctx, key)
}

// recheckEvery bounds identity re-probes per account: a burst of 401s from
// one review costs one probe, not one per request. A var for the tests.
var recheckEvery = 30 * time.Second

// probeTimeout bounds one identity probe that is not a push: the re-probe
// inside the worker's call that saw a 401 (every team's queue waits on it),
// the one in the webhook route (Bitbucket times its delivery out), and the
// boot probe of each team in turn. Short on purpose: a probe that does not
// answer in time leaves the status as it was.
const probeTimeout = 5 * time.Second

// accounts owns a team's two accounts: the live clients, their state, and
// the push and reset that change them.
type accounts struct {
	slug   string
	log    *slog.Logger
	shared *bitbucket.Client
	// jiraShared is the instance's Jira user; a team's own token is a clone
	// of it, so the transport is shared. Nil when the team has no Jira.
	jiraShared *jira.Client
	store      CredentialStore
	box        *secretbox.Box
	bot        *botSwitch
	jira       *jiraSwitch

	// mu guards state, gen, tokens and checked. gen counts the clients
	// installed per kind; an observer carries the one it was built for, so
	// a late outcome from a replaced client never marks the new one.
	mu      sync.Mutex
	state   map[Kind]Account
	gen     map[Kind]uint64
	tokens  map[Kind]string
	checked map[Kind]time.Time
	// pushMu makes prove, persist and swap one step per team.
	pushMu sync.Mutex
}

// AccountDeps are what a team's accounts are built from.
type AccountDeps struct {
	Store CredentialStore
	Box   *secretbox.Box
	// Shared is the instance bot; a team's own account is a clone of it
	// with the team's token, so the transport is shared.
	Shared *bitbucket.Client
	// Jira is the team's resolved Jira config: the instance URL and token.
	Jira config.Jira
	Log  *slog.Logger
}

func newAccounts(slug string, d AccountDeps) (*accounts, error) {
	a := &accounts{
		slug: slug, log: d.Log, shared: d.Shared, store: d.Store, box: d.Box,
		bot: &botSwitch{}, jira: &jiraSwitch{},
		state: map[Kind]Account{}, gen: map[Kind]uint64{},
		tokens: map[Kind]string{}, checked: map[Kind]time.Time{},
	}
	if d.Jira.URL != "" {
		c, err := jira.New(d.Jira, d.Log)
		if err != nil {
			return nil, err
		}
		a.jiraShared = c
	}
	return a, nil
}

// install records the account and the token behind it, and returns the
// generation its client must observe under. An own account counts as just
// checked: it was proven (or found broken) a moment ago.
func (a *accounts) install(kind Kind, acct Account, token string) uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.gen[kind]++
	a.state[kind] = acct
	a.tokens[kind] = token
	a.checked[kind] = time.Now()
	return a.gen[kind]
}

func (a *accounts) get(kind Kind) Account {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.state[kind]
}

// setStatus moves an own account to next, unless its client was replaced
// meanwhile (gen) or nothing changes. Undecryptable and username_mismatch are
// final for the client they were found with: no call outcome or recheck
// clears them, only a push installs a new one. One log line per change.
func (a *accounts) setStatus(kind Kind, gen uint64, next Status, detail string) {
	a.mu.Lock()
	cur := a.state[kind]
	final := cur.Status == StatusUndecryptable || cur.Status == StatusUsernameMismatch
	if a.gen[kind] != gen || !cur.Own || cur.Status == next || final {
		a.mu.Unlock()
		return
	}
	cur.Status, cur.Since = next, time.Now()
	a.state[kind] = cur
	a.mu.Unlock()
	line := fmt.Sprintf("team credential status changed kind=%s status=%s", kind, next)
	if detail != "" {
		line += ": " + detail
	}
	if next == StatusOK {
		a.log.Info(line)
	} else {
		a.log.Warn(line)
	}
}

// observer turns an own account's request outcomes into its status: a
// success is ok, a transport failure unreachable. A 401 alone proves nothing
// about the token: Bitbucket answers 401 too when a working token lacks
// rights on one project or repo. It triggers a recheck of the token itself,
// which decides. Anything else (403, 404, 5xx) says nothing about the token.
func (a *accounts) observer(kind Kind, gen uint64) func(int, error) {
	return func(status int, err error) {
		switch {
		case err != nil && errors.Is(err, context.Canceled):
		case err != nil:
			a.setStatus(kind, gen, StatusUnreachable, "")
		case status == http.StatusUnauthorized:
			ctx, cancel := context.WithTimeout(context.Background(), probeTimeout)
			defer cancel()
			a.recheck(ctx, kind)
		case status >= 200 && status < 300:
			a.setStatus(kind, gen, StatusOK, "")
		}
	}
}

// recheck proves an own account's token again, at most once per
// recheckEvery, and sets its status from the answer. It runs where it is
// called from: inside the worker's call that saw a 401, or in the webhook
// route of a blocked team, never on a goroutine of its own. Bounded by
// probeTimeout; a probe that gets no answer changes nothing, so a slow
// Bitbucket neither blocks nor unblocks a team. Undecryptable and
// username_mismatch are not rechecked (setStatus).
func (a *accounts) recheck(ctx context.Context, kind Kind) {
	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()
	a.mu.Lock()
	cur, gen, token := a.state[kind], a.gen[kind], a.tokens[kind]
	if !cur.Own || cur.Status == StatusUndecryptable || cur.Status == StatusUsernameMismatch ||
		time.Since(a.checked[kind]) < recheckEvery {
		a.mu.Unlock()
		return
	}
	a.checked[kind] = time.Now()
	a.mu.Unlock()
	var status Status
	var detail string
	switch kind {
	case KindBitbucket:
		status, detail = probeBitbucket(ctx, a.shared.WithAccount(token, cur.Username, a.log, nil), cur.Username)
	case KindJira:
		status, detail = probeJira(ctx, a.jiraShared.WithToken(token))
	}
	if status == StatusUnreachable {
		return
	}
	a.setStatus(kind, gen, status, detail)
}

// probeBitbucket proves a token and the username it is pushed with. The
// detail is for the log and the pushing team, never the dashboard.
func probeBitbucket(ctx context.Context, c *bitbucket.Client, username string) (Status, string) {
	who, err := c.Identity(ctx)
	if err != nil {
		if st := bitbucket.Status(err); st == http.StatusUnauthorized || st == http.StatusForbidden {
			return StatusRejected, fmt.Sprintf("Bitbucket rejected the token (HTTP %d)", st)
		}
		return StatusUnreachable, fmt.Sprintf("Bitbucket did not answer: %v", err)
	}
	// No name is no proof: the username becomes the team's @ and the
	// bot's-own-comment check, so an unchecked one could have the bot answer
	// its own notices. Bitbucket Data Center names the account in
	// X-AUSERNAME or the whoami servlet; neither answering is refused.
	if who == "" {
		return StatusUsernameMismatch, "Bitbucket does not say which account the token belongs to, so the username cannot be checked"
	}
	if !strings.EqualFold(who, username) {
		return StatusUsernameMismatch, fmt.Sprintf("the token belongs to %s, not %s", who, username)
	}
	return StatusOK, ""
}

func probeJira(ctx context.Context, c *jira.Client) (Status, string) {
	if err := c.CheckConnectivity(ctx); err != nil {
		if st := jira.Status(err); st == http.StatusUnauthorized || st == http.StatusForbidden {
			return StatusRejected, fmt.Sprintf("Jira rejected the token (HTTP %d)", st)
		}
		return StatusUnreachable, fmt.Sprintf("Jira did not answer: %v", err)
	}
	return StatusOK, ""
}

// start installs the boot-time clients: the team's own where it has one,
// whatever state it is in, else the shared ones. A bad own account is
// reported, logged once, and the team starts anyway.
func (a *accounts) start(ctx context.Context, opened map[Kind]Opened) {
	if o, ok := opened[KindBitbucket]; ok {
		a.useOwnBitbucket(ctx, o)
	} else {
		a.useSharedBitbucket()
	}
	if a.jiraShared == nil {
		return
	}
	if o, ok := opened[KindJira]; ok {
		a.useOwnJira(ctx, o)
	} else {
		a.useSharedJira()
	}
}

func (a *accounts) useSharedBitbucket() {
	a.install(KindBitbucket, Account{}, "")
	a.bot.cur.Store(a.shared)
}

func (a *accounts) useOwnBitbucket(ctx context.Context, o Opened) {
	if o.Undecryptable {
		// No token to send. The client keeps the username, so the mention
		// trigger stays the team's own bot rather than falling back to the
		// shared one; no observer, nothing it sends could tell us more.
		a.install(KindBitbucket, Account{Own: true, Username: o.Username, Status: StatusUndecryptable, Since: time.Now()}, "")
		a.bot.cur.Store(a.shared.WithAccount("", o.Username, a.log, nil))
		a.log.WarnContext(ctx, fmt.Sprintf("team credentials not usable kind=%s status=%s: the stored token does not decrypt; "+
			"webhooks are acknowledged and not processed until the team pushes it again", KindBitbucket, StatusUndecryptable))
		return
	}
	// Bounded: teams start one after another before the server listens, so
	// a degraded Bitbucket must not hold boot. No answer reads unreachable,
	// which does not block; the first call that works sets ok.
	pctx, cancel := context.WithTimeout(ctx, probeTimeout)
	status, detail := probeBitbucket(pctx, a.shared.WithAccount(o.Token, o.Username, a.log, nil), o.Username)
	cancel()
	gen := a.install(KindBitbucket, Account{Own: true, Username: o.Username, Status: status, Since: time.Now()}, o.Token)
	a.bot.cur.Store(a.shared.WithAccount(o.Token, o.Username, a.log, a.observer(KindBitbucket, gen)))
	if status != StatusOK {
		a.log.WarnContext(ctx, fmt.Sprintf("team credentials not usable kind=%s status=%s: %s", KindBitbucket, status, detail))
		return
	}
	a.log.InfoContext(ctx, fmt.Sprintf("Bitbucket: own bot %s", o.Username))
}

func (a *accounts) useSharedJira() {
	a.install(KindJira, Account{}, "")
	a.jira.cur.Store(a.jiraShared)
}

func (a *accounts) useOwnJira(ctx context.Context, o Opened) {
	if o.Undecryptable {
		a.install(KindJira, Account{Own: true, Status: StatusUndecryptable, Since: time.Now()}, "")
		a.jira.cur.Store(nil)
		a.log.WarnContext(ctx, fmt.Sprintf("team credentials not usable kind=%s status=%s: the stored token does not decrypt; "+
			"reviews run without the ticket check until the team pushes it again", KindJira, StatusUndecryptable))
		return
	}
	c := a.jiraShared.WithToken(o.Token)
	pctx, cancel := context.WithTimeout(ctx, probeTimeout)
	status, detail := probeJira(pctx, c)
	cancel()
	gen := a.install(KindJira, Account{Own: true, Status: status, Since: time.Now()}, o.Token)
	a.jira.cur.Store(c.Observed(a.observer(KindJira, gen)))
	if status != StatusOK {
		a.log.WarnContext(ctx, fmt.Sprintf("team credentials not usable kind=%s status=%s: %s", KindJira, status, detail))
	}
}

// PushError is a pushed credential that failed its proof, or cannot be taken
// at all. Nothing was stored and nothing changed. Detail is for the pushing
// team (it holds the token).
type PushError struct {
	Kind   Kind
	Status Status
	Detail string
}

func (e *PushError) Error() string { return fmt.Sprintf("%s: %s", e.Kind, e.Detail) }

// BitbucketPush is a team's own bot as pushed: its username (the @mention
// handle) and its token.
type BitbucketPush struct {
	Username string
	Token    string
}

// push proves every part, persists all of them at once, then swaps each in;
// a part left nil is kept. A *PushError is a proof failing, and then nothing
// changed; so does a failed persist.
func (a *accounts) push(ctx context.Context, bb *BitbucketPush, jiraToken *string, by string) error {
	a.pushMu.Lock()
	defer a.pushMu.Unlock()
	if bb != nil {
		if status, detail := probeBitbucket(ctx, a.shared.WithAccount(bb.Token, bb.Username, a.log, nil), bb.Username); status != StatusOK {
			return &PushError{Kind: KindBitbucket, Status: status, Detail: detail}
		}
	}
	var jc *jira.Client
	if jiraToken != nil {
		if a.jiraShared == nil {
			return &PushError{Kind: KindJira, Detail: "this team has no Jira to use a token with"}
		}
		jc = a.jiraShared.WithToken(*jiraToken)
		if status, detail := probeJira(ctx, jc); status != StatusOK {
			return &PushError{Kind: KindJira, Status: status, Detail: detail}
		}
	}

	var rows []store.SealedCredential
	if bb != nil {
		row, err := a.seal(KindBitbucket, bb.Username, bb.Token)
		if err != nil {
			return err
		}
		rows = append(rows, row)
	}
	if jc != nil {
		row, err := a.seal(KindJira, "", *jiraToken)
		if err != nil {
			return err
		}
		rows = append(rows, row)
	}
	if err := a.store.PutCredentials(ctx, rows, by); err != nil {
		return err
	}

	if bb != nil {
		gen := a.install(KindBitbucket, Account{Own: true, Username: bb.Username, Status: StatusOK, Since: time.Now()}, bb.Token)
		a.bot.cur.Store(a.shared.WithAccount(bb.Token, bb.Username, a.log, a.observer(KindBitbucket, gen)))
	}
	if jc != nil {
		gen := a.install(KindJira, Account{Own: true, Status: StatusOK, Since: time.Now()}, *jiraToken)
		a.jira.cur.Store(jc.Observed(a.observer(KindJira, gen)))
	}
	return nil
}

func (a *accounts) seal(kind Kind, username, token string) (store.SealedCredential, error) {
	nonce, ct, err := a.box.Seal([]byte(token), credentialAAD(a.slug, kind))
	if err != nil {
		return store.SealedCredential{}, err
	}
	return store.SealedCredential{Team: a.slug, Kind: string(kind), Username: username, Nonce: nonce, Ciphertext: ct}, nil
}

// reset drops the team's own account and goes back to the shared one; false
// when the team had none.
func (a *accounts) reset(ctx context.Context, kind Kind, by string) (bool, error) {
	a.pushMu.Lock()
	defer a.pushMu.Unlock()
	removed, err := a.store.DeleteCredential(ctx, a.slug, string(kind), by)
	if err != nil || !removed {
		return removed, err
	}
	if kind == KindBitbucket {
		a.useSharedBitbucket()
	} else {
		a.useSharedJira()
	}
	return true, nil
}
