package teams

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/secretbox"
	"github.com/trick77/noergler/internal/store"
)

// fakeCredStore is the credential store in memory.
type fakeCredStore struct {
	mu       sync.Mutex
	rows     map[string]store.SealedCredential // "slug|kind"
	check    [2][]byte
	hasCheck bool
	putErr   error
}

func newCredStore() *fakeCredStore {
	return &fakeCredStore{rows: map[string]store.SealedCredential{}}
}

func (f *fakeCredStore) ListCredentials(context.Context) ([]store.SealedCredential, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []store.SealedCredential
	for _, r := range f.rows {
		out = append(out, r)
	}
	return out, nil
}

// PutCredentials is all or none, like the store's transaction.
func (f *fakeCredStore) PutCredentials(_ context.Context, creds []store.SealedCredential, _ string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.putErr != nil {
		return f.putErr
	}
	for _, c := range creds {
		f.rows[c.Team+"|"+c.Kind] = c
	}
	return nil
}

func (f *fakeCredStore) DeleteCredential(_ context.Context, team, kind, _ string) (bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	_, ok := f.rows[team+"|"+kind]
	delete(f.rows, team+"|"+kind)
	return ok, nil
}

func (f *fakeCredStore) KeyCheck(context.Context) ([]byte, []byte, bool, error) {
	return f.check[0], f.check[1], f.hasCheck, nil
}

func (f *fakeCredStore) PutKeyCheck(_ context.Context, nonce, ct []byte) error {
	f.check, f.hasCheck = [2][]byte{nonce, ct}, true
	return nil
}

func testBox(t *testing.T, fill byte) *secretbox.Box {
	t.Helper()
	b, err := secretbox.New(bytes.Repeat([]byte{fill}, secretbox.KeySize))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// seal stores a credential the way a push does.
func (f *fakeCredStore) seal(t *testing.T, box *secretbox.Box, slug string, kind Kind, username, token string) {
	t.Helper()
	nonce, ct, err := box.Seal([]byte(token), credentialAAD(slug, kind))
	if err != nil {
		t.Fatal(err)
	}
	f.rows[slug+"|"+string(kind)] = store.SealedCredential{Team: slug, Kind: string(kind), Username: username, Nonce: nonce, Ciphertext: ct}
}

// known is the set of configured teams OpenCredentials is given.
func known(slugs ...string) map[string]bool {
	out := map[string]bool{}
	for _, s := range slugs {
		out[s] = true
	}
	return out
}

// fakeUpstream answers Bitbucket and Jira at once: a token's account is
// tokens[token], unknown tokens are 401. Bitbucket names the account in
// X-AUSERNAME. A path in denied answers 404 to everyone; a path in noRights
// answers 401 to everyone, as Bitbucket does for a working token without
// rights on a project or repo.
type fakeUpstream struct {
	*httptest.Server
	mu       sync.Mutex
	tokens   map[string]string
	denied   map[string]bool
	noRights map[string]bool
	broken   map[string]bool // 503 to everyone
	calls    int
}

func newUpstream(t *testing.T, tokens map[string]string) *fakeUpstream {
	u := &fakeUpstream{tokens: tokens, denied: map[string]bool{}, noRights: map[string]bool{}, broken: map[string]bool{}}
	u.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u.mu.Lock()
		u.calls++
		who, ok := u.tokens[strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")]
		denied, noRights, broken := u.denied[r.URL.Path], u.noRights[r.URL.Path], u.broken[r.URL.Path]
		u.mu.Unlock()
		// No whoami servlet: the account is named in X-AUSERNAME or not at
		// all, so a token mapped to "" is one Bitbucket names no account for.
		if r.URL.Path == "/plugins/servlet/applinks/whoami" || broken {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		if !ok || noRights {
			http.Error(w, `{"errors":[{"message":"Authentication failed"}]}`, http.StatusUnauthorized)
			return
		}
		if denied {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("X-AUSERNAME", who)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"name":"` + who + `"}`))
	}))
	t.Cleanup(u.Close)
	return u
}

func (u *fakeUpstream) revoke(token string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	delete(u.tokens, token)
}

func (u *fakeUpstream) restore(token, who string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.tokens[token] = who
}

func (u *fakeUpstream) shared(t *testing.T) *bitbucket.Client {
	t.Helper()
	log, _ := bufLogger()
	c, err := bitbucket.New(config.Bitbucket{BaseURL: u.URL, Token: "shared-token", Username: "rop"}, log)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func accountsFor(t *testing.T, u *fakeUpstream, st *fakeCredStore, box *secretbox.Box, slug string) (*accounts, *bytes.Buffer) {
	t.Helper()
	log, buf := bufLogger()
	a, err := newAccounts(slug, AccountDeps{
		Store: st, Box: box, Shared: u.shared(t),
		Jira: config.Jira{URL: u.URL, Token: "shared-jira"},
		Log:  log.With("team", slug),
	})
	if err != nil {
		t.Fatal(err)
	}
	return a, buf
}

// noRecheckLimit lets every 401 re-probe at once, for tests that drive
// several in a row.
func noRecheckLimit(t *testing.T) {
	t.Helper()
	was := recheckEvery
	recheckEvery = 0
	t.Cleanup(func() { recheckEvery = was })
}

func TestOpenCredentials_WritesTheKeyCheckOnce(t *testing.T) {
	st, box := newCredStore(), testBox(t, 1)
	log, _ := bufLogger()
	if _, err := OpenCredentials(context.Background(), st, box, known(), log); err != nil || !st.hasCheck {
		t.Fatalf("first boot: err=%v check=%v", err, st.hasCheck)
	}
	first := st.check
	if _, err := OpenCredentials(context.Background(), st, box, known(), log); err != nil {
		t.Fatalf("second boot: %v", err)
	}
	if !bytes.Equal(first[1], st.check[1]) {
		t.Error("a matching key check was rewritten")
	}
}

// The shared fault: a key that does not open the key check while credentials
// are stored aborts boot, once, instead of every team failing on its own.
func TestOpenCredentials_WrongKeyIsSharedFault(t *testing.T) {
	st := newCredStore()
	l, _ := bufLogger()
	if _, err := OpenCredentials(context.Background(), st, testBox(t, 1), known("platform"), l); err != nil {
		t.Fatal(err)
	}
	st.seal(t, testBox(t, 1), "platform", KindBitbucket, "plat-bot", "tok")
	_, err := OpenCredentials(context.Background(), st, testBox(t, 2), known("platform"), l)
	if err == nil || !strings.Contains(err.Error(), "NOERGLER_SECRETS_KEY does not match") {
		t.Fatalf("err = %v, want the key mismatch", err)
	}
}

// Nothing stored, nothing lost: a changed key just resets the check.
func TestOpenCredentials_WrongKeyWithNothingStoredResets(t *testing.T) {
	st := newCredStore()
	l, buf := bufLogger()
	_, _ = OpenCredentials(context.Background(), st, testBox(t, 1), known(), l)
	if _, err := OpenCredentials(context.Background(), st, testBox(t, 2), known(), l); err != nil {
		t.Fatalf("err = %v, want a reset", err)
	}
	if _, err := OpenCredentials(context.Background(), st, testBox(t, 2), known(), l); err != nil {
		t.Fatalf("after reset: %v", err)
	}
	if !strings.Contains(buf.String(), "key check is reset") {
		t.Error("the reset is not logged")
	}
}

// A row of a team no longer in teams.yaml decides nothing: it neither blocks
// a key change nor reaches any team.
func TestOpenCredentials_OrphanRowsAreIgnored(t *testing.T) {
	st := newCredStore()
	l, buf := bufLogger()
	_, _ = OpenCredentials(context.Background(), st, testBox(t, 1), known("platform"), l)
	st.seal(t, testBox(t, 1), "gone", KindBitbucket, "gone-bot", "tok")
	opened, err := OpenCredentials(context.Background(), st, testBox(t, 2), known("platform"), l)
	if err != nil {
		t.Fatalf("an orphan row blocked a key change: %v", err)
	}
	if _, ok := opened["gone"]; ok {
		t.Error("an orphan row was opened")
	}
	if !strings.Contains(buf.String(), "unknown team ignored team=gone") {
		t.Error("the orphan is not logged")
	}
}

// Credentials without a key check (a partial restore): the rows decide. A
// key that opens none is the shared fault; one that opens them writes the
// check, so a later wrong key is caught by it.
func TestOpenCredentials_LostKeyCheck(t *testing.T) {
	st := newCredStore()
	l, _ := bufLogger()
	st.seal(t, testBox(t, 1), "platform", KindBitbucket, "plat-bot", "tok")
	if _, err := OpenCredentials(context.Background(), st, testBox(t, 2), known("platform"), l); err == nil {
		t.Fatal("a key that opens no row passed without a key check")
	}
	if st.hasCheck {
		t.Fatal("the wrong key was written as the key check")
	}
	if _, err := OpenCredentials(context.Background(), st, testBox(t, 1), known("platform"), l); err != nil || !st.hasCheck {
		t.Fatalf("the right key: err=%v check=%v", err, st.hasCheck)
	}
}

// A row that does not open under a key that passed is that team's fault
// only: it comes back Undecryptable and boot goes on.
func TestOpenCredentials_OneTeamsBrokenRowIsThatTeamsOnly(t *testing.T) {
	st, box := newCredStore(), testBox(t, 1)
	l, _ := bufLogger()
	_, _ = OpenCredentials(context.Background(), st, box, known(), l)
	st.seal(t, box, "payments", KindBitbucket, "pay-bot", "pay-tok")
	st.seal(t, box, "platform", KindBitbucket, "plat-bot", "plat-tok")
	// platform's row copied over payments': the associated data no longer
	// matches.
	copied := st.rows["platform|bitbucket"]
	copied.Team = "payments"
	st.rows["payments|bitbucket"] = copied

	opened, err := OpenCredentials(context.Background(), st, box, known("payments", "platform"), l)
	if err != nil {
		t.Fatalf("a team's row must not abort boot: %v", err)
	}
	if o := opened["payments"][KindBitbucket]; !o.Undecryptable || o.Token != "" {
		t.Errorf("payments = %+v, want undecryptable", o)
	}
	if o := opened["platform"][KindBitbucket]; o.Undecryptable || o.Token != "plat-tok" || o.Username != "plat-bot" {
		t.Errorf("platform = %+v", o)
	}
}

func TestBoot_WrongSecretsKeyAbortsBoot(t *testing.T) {
	st := newCredStore()
	l, _ := bufLogger()
	_, _ = OpenCredentials(context.Background(), st, testBox(t, 1), known(), l)
	st.seal(t, testBox(t, 1), "platform", KindJira, "", "tok")
	app := appWith(map[string]*config.Team{"platform": {Slug: "platform"}}, []string{"platform"}, nil)
	_, err := Boot(context.Background(), app, Deps{
		Claims: emptyStore(), Log: l, Credentials: st, Box: testBox(t, 2), Env: noEnv,
	})
	if err == nil {
		t.Fatal("a wrong secrets key must abort boot")
	}
}

// Team A's revoked token, team B's working one: each gets its own state,
// neither disables anything.
func TestAccounts_BootStateIsPerTeam(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop", "b-tok": "b-bot", "shared-jira": "rop"})
	st, box := newCredStore(), testBox(t, 1)

	a, aLog := accountsFor(t, u, st, box, "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "revoked"}})
	b, _ := accountsFor(t, u, st, box, "beta")
	b.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "b-bot", Token: "b-tok"}})
	c, _ := accountsFor(t, u, st, box, "gamma")
	c.start(context.Background(), nil)

	if got := a.get(KindBitbucket); !got.Own || got.Status != StatusRejected || !got.blocking() {
		t.Errorf("alpha = %+v, want own and rejected", got)
	}
	if !strings.Contains(aLog.String(), "status=rejected") || strings.Contains(aLog.String(), "revoked") {
		t.Errorf("alpha's log must name the status and never the token:\n%s", aLog)
	}
	if got := b.get(KindBitbucket); got.Status != StatusOK || got.blocking() || b.bot.BotUsername() != "b-bot" {
		t.Errorf("beta = %+v bot=%s", got, b.bot.BotUsername())
	}
	if got := c.get(KindBitbucket); got.Own || c.bot.BotUsername() != "rop" {
		t.Errorf("gamma = %+v bot=%s, want the shared bot", got, c.bot.BotUsername())
	}
}

func TestAccounts_UndecryptableKeepsTheTeamsHandleAndBlocks(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop", "shared-jira": "rop"})
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{
		KindBitbucket: {Username: "a-bot", Undecryptable: true},
		KindJira:      {Undecryptable: true},
	})
	if got := a.get(KindBitbucket); got.Status != StatusUndecryptable || !got.blocking() || a.bot.BotUsername() != "a-bot" {
		t.Errorf("bitbucket = %+v bot=%s", got, a.bot.BotUsername())
	}
	// Jira never blocks (only BitbucketBlocked does): reviews run without
	// the ticket check.
	if got := a.get(KindJira); got.Status != StatusUndecryptable {
		t.Errorf("jira = %+v", got)
	}
	if _, _, err := a.jira.FetchTicketWithParent(context.Background(), "ABC-1"); err == nil {
		t.Error("an undecryptable Jira token must fail the fetch, not use the shared one")
	}
}

func TestAccounts_UsernameMismatchBlocks(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop", "tok": "someone-else", "shared-jira": "rop"})
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})
	if got := a.get(KindBitbucket); got.Status != StatusUsernameMismatch || !got.blocking() {
		t.Errorf("bitbucket = %+v", got)
	}
}

// A token revoked mid-run: the 401 makes the account prove its token again,
// which is refused, so it flips to rejected; once the token works again, the
// next success flips it back. One log line per change.
func TestAccounts_RevokedTokenFlipsAfterARecheck(t *testing.T) {
	noRecheckLimit(t)
	u := newUpstream(t, map[string]string{"shared-token": "rop", "tok": "a-bot", "shared-jira": "rop"})
	a, buf := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})

	u.revoke("tok")
	_ = a.bot.GetProject(context.Background(), "PLAT")
	_ = a.bot.GetProject(context.Background(), "PLAT")
	if got := a.get(KindBitbucket); got.Status != StatusRejected {
		t.Fatalf("after a 401 and a refused recheck: %+v", got)
	}
	if n := strings.Count(buf.String(), "status=rejected"); n != 1 {
		t.Errorf("logged %d status changes for two 401s, want one", n)
	}
	u.restore("tok", "a-bot")
	_ = a.bot.GetProject(context.Background(), "PLAT")
	if got := a.get(KindBitbucket); got.Status != StatusOK {
		t.Fatalf("after a success: %+v", got)
	}
}

// Bitbucket's 401 for a working token without rights on one repo (a bot not
// granted yet) is not a rejected token: the recheck proves it and the team
// stays unblocked.
func TestAccounts_MissingRightsAreNotARejectedToken(t *testing.T) {
	noRecheckLimit(t)
	u := newUpstream(t, map[string]string{"shared-token": "rop", "tok": "a-bot", "shared-jira": "rop"})
	u.noRights["/rest/api/1.0/projects/PAY"] = true
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})

	if err := a.bot.GetProject(context.Background(), "PAY"); bitbucket.Status(err) != http.StatusUnauthorized {
		t.Fatalf("want the 401 through, got %v", err)
	}
	if got := a.get(KindBitbucket); got.Status != StatusOK || got.blocking() {
		t.Errorf("a missing grant blocked the team: %+v", got)
	}
}

// A burst of 401s costs one recheck per recheckEvery, not one per request.
func TestAccounts_RecheckIsRateLimited(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop", "tok": "a-bot", "shared-jira": "rop"})
	u.noRights["/rest/api/1.0/projects/PAY"] = true
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})

	u.mu.Lock()
	before := u.calls
	u.mu.Unlock()
	for range 5 {
		_ = a.bot.GetProject(context.Background(), "PAY")
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	if n := u.calls - before; n != 5 {
		t.Errorf("%d calls for five 401s just after boot, want five and no recheck", n)
	}
}

// A blocked team sends no Bitbucket call that could clear it, so the webhook
// route rechecks a rejected token: re-enabled in Bitbucket, it unblocks.
func TestRuntime_BlockedRejectedTokenRecovers(t *testing.T) {
	noRecheckLimit(t)
	u := newUpstream(t, map[string]string{"shared-token": "rop", "shared-jira": "rop"})
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})
	rt := NewRuntime(&config.Team{Slug: "alpha"}, nil, nil, nil)
	rt.accounts = a

	if _, blocked := rt.BitbucketBlocked(context.Background()); !blocked {
		t.Fatal("a refused token must block")
	}
	u.restore("tok", "a-bot")
	if bb, blocked := rt.BitbucketBlocked(context.Background()); blocked || bb.Status != StatusOK {
		t.Errorf("after the token works again: %+v blocked=%v", bb, blocked)
	}
}

// An observer of a replaced client never marks the new one.
func TestAccounts_StaleObserverIsIgnored(t *testing.T) {
	noRecheckLimit(t)
	u := newUpstream(t, map[string]string{"shared-token": "rop", "tok": "a-bot", "tok2": "a-bot", "shared-jira": "rop"})
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})
	old := a.bot.c()
	if err := a.push(context.Background(), &BitbucketPush{Username: "a-bot", Token: "tok2"}, nil, "team:alpha"); err != nil {
		t.Fatal(err)
	}
	u.revoke("tok")
	_ = old.GetProject(context.Background(), "PLAT")
	if got := a.get(KindBitbucket); got.Status != StatusOK {
		t.Errorf("the old client's 401 marked the new account: %+v", got)
	}
}

func TestAccounts_PushIsProvenBeforeAnythingChanges(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop", "good": "a-bot", "jira-good": "a-jira", "shared-jira": "rop"})
	st, box := newCredStore(), testBox(t, 1)
	a, buf := accountsFor(t, u, st, box, "alpha")
	a.start(context.Background(), nil)

	for name, tc := range map[string]struct {
		bb   *BitbucketPush
		jira string
		want Status
	}{
		"rejected token":    {bb: &BitbucketPush{Username: "a-bot", Token: "bad"}, want: StatusRejected},
		"username mismatch": {bb: &BitbucketPush{Username: "not-a-bot", Token: "good"}, want: StatusUsernameMismatch},
		// A good Bitbucket part does not land when the Jira part fails.
		"jira rejected": {bb: &BitbucketPush{Username: "a-bot", Token: "good"}, jira: "bad", want: StatusRejected},
	} {
		var jt *string
		if tc.jira != "" {
			jt = &tc.jira
		}
		err := a.push(context.Background(), tc.bb, jt, "team:alpha")
		var pe *PushError
		if !errors.As(err, &pe) || pe.Status != tc.want {
			t.Errorf("%s: err = %v, want a %s PushError", name, err, tc.want)
		}
		if len(st.rows) != 0 || a.get(KindBitbucket).Own || a.bot.BotUsername() != "rop" {
			t.Errorf("%s: something changed: rows=%d state=%+v", name, len(st.rows), a.get(KindBitbucket))
		}
	}

	jt := "jira-good"
	if err := a.push(context.Background(), &BitbucketPush{Username: "a-bot", Token: "good"}, &jt, "team:alpha"); err != nil {
		t.Fatal(err)
	}
	if a.bot.BotUsername() != "a-bot" || !a.get(KindBitbucket).Own || !a.get(KindJira).Own {
		t.Errorf("after push: bot=%s bb=%+v jira=%+v", a.bot.BotUsername(), a.get(KindBitbucket), a.get(KindJira))
	}
	row := st.rows["alpha|bitbucket"]
	if bytes.Contains(row.Ciphertext, []byte("good")) || row.Username != "a-bot" {
		t.Errorf("stored row = %+v", row)
	}
	if plain, err := box.Open(row.Nonce, row.Ciphertext, credentialAAD("alpha", KindBitbucket)); err != nil || string(plain) != "good" {
		t.Errorf("stored token does not open: %q %v", plain, err)
	}
	if strings.Contains(buf.String(), "good") || strings.Contains(buf.String(), "jira-good") {
		t.Errorf("a token reached the log:\n%s", buf)
	}

	if ok, err := a.reset(context.Background(), KindBitbucket, "team:alpha"); !ok || err != nil {
		t.Fatalf("reset = %v %v", ok, err)
	}
	if a.bot.BotUsername() != "rop" || a.get(KindBitbucket).Own {
		t.Errorf("after reset: bot=%s", a.bot.BotUsername())
	}
	if ok, _ := a.reset(context.Background(), KindBitbucket, "team:alpha"); ok {
		t.Error("a second reset reported a credential")
	}
}

// A failed persist swaps nothing in, neither part.
func TestAccounts_PersistFailureChangesNothing(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop", "good": "a-bot", "jira-good": "a-jira", "shared-jira": "rop"})
	st := newCredStore()
	st.putErr = errors.New("db down")
	a, _ := accountsFor(t, u, st, testBox(t, 1), "alpha")
	a.start(context.Background(), nil)
	jt := "jira-good"
	if err := a.push(context.Background(), &BitbucketPush{Username: "a-bot", Token: "good"}, &jt, "x"); err == nil {
		t.Fatal("want the store error")
	}
	if a.bot.BotUsername() != "rop" || a.get(KindBitbucket).Own || a.get(KindJira).Own {
		t.Error("a failed persist swapped a client in")
	}
}

// A Jira token for a team without a Jira is refused as such, not as a
// database fault.
func TestAccounts_JiraPushWithoutJira(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop"})
	log, _ := bufLogger()
	a, err := newAccounts("alpha", AccountDeps{Store: newCredStore(), Box: testBox(t, 1), Shared: u.shared(t), Log: log})
	if err != nil {
		t.Fatal(err)
	}
	a.start(context.Background(), nil)
	jt := "x"
	var pe *PushError
	if err := a.push(context.Background(), nil, &jt, "x"); !errors.As(err, &pe) || pe.Kind != KindJira {
		t.Fatalf("err = %v, want a jira PushError", err)
	}
}

// A token Bitbucket names no account for cannot prove the username, and the
// username is the bot's own-comment check: refused, never taken on faith.
func TestAccounts_UnverifiableUsernameIsRefused(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop", "tok": "", "shared-jira": "rop"})
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), nil)
	var pe *PushError
	if err := a.push(context.Background(), &BitbucketPush{Username: "a-bot", Token: "tok"}, nil, "x"); !errors.As(err, &pe) || pe.Status != StatusUsernameMismatch {
		t.Fatalf("err = %v, want a username_mismatch PushError", err)
	}
}

// username_mismatch is final for its client: a call that works (onboarding,
// a recheck) must not unblock a team whose bot posts under another name.
func TestAccounts_MismatchIsNotClearedByASuccess(t *testing.T) {
	noRecheckLimit(t)
	u := newUpstream(t, map[string]string{"shared-token": "rop", "tok": "someone-else", "shared-jira": "rop"})
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})
	_ = a.bot.GetProject(context.Background(), "PLAT")
	a.recheck(context.Background(), KindBitbucket)
	if got := a.get(KindBitbucket); got.Status != StatusUsernameMismatch || !got.blocking() {
		t.Errorf("a success cleared the mismatch: %+v", got)
	}
}

// A recheck that gets no answer changes nothing: a slow Bitbucket neither
// blocks nor unblocks.
func TestAccounts_UnansweredRecheckChangesNothing(t *testing.T) {
	noRecheckLimit(t)
	u := newUpstream(t, map[string]string{"shared-token": "rop", "shared-jira": "rop"})
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})
	if got := a.get(KindBitbucket); got.Status != StatusRejected {
		t.Fatalf("boot = %+v", got)
	}
	u.Close()
	a.recheck(context.Background(), KindBitbucket)
	if got := a.get(KindBitbucket); got.Status != StatusRejected {
		t.Errorf("an unanswered recheck changed the status: %+v", got)
	}
}

// Only a refusal (401, 403, 404) means grant-bot is needed; a 5xx says
// nothing about rights.
func TestRuntime_UnreachableClaimsIgnoresServerErrors(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop", "tok": "a-bot", "shared-jira": "rop"})
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})
	u.mu.Lock()
	u.broken["/rest/api/1.0/projects/PLAT"] = true
	u.mu.Unlock()
	rt := NewRuntime(&config.Team{Slug: "alpha", Projects: []config.ProjectScope{{Key: "PLAT"}}}, nil, nil, nil)
	rt.accounts = a
	if got := rt.UnreachableClaims(context.Background()); len(got) != 0 {
		t.Errorf("a 503 listed %v as needing grant-bot", got)
	}
}

func TestRuntime_UnreachableClaims(t *testing.T) {
	u := newUpstream(t, map[string]string{"shared-token": "rop", "tok": "a-bot", "shared-jira": "rop"})
	u.denied["/rest/api/1.0/projects/PAY/repos/ledger"] = true
	a, _ := accountsFor(t, u, newCredStore(), testBox(t, 1), "alpha")
	a.start(context.Background(), map[Kind]Opened{KindBitbucket: {Username: "a-bot", Token: "tok"}})
	rt := NewRuntime(&config.Team{Slug: "alpha", Projects: []config.ProjectScope{
		{Key: "PLAT"}, {Key: "PAY", Repos: []string{"billing", "ledger"}},
	}}, nil, nil, nil)
	rt.accounts = a
	if got := rt.UnreachableClaims(context.Background()); len(got) != 1 || got[0] != "PAY/ledger" {
		t.Errorf("unreachable = %v", got)
	}
}
