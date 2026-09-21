package api

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/httpapi"
	"github.com/trick77/noergler/internal/logging"
	"github.com/trick77/noergler/internal/queue"
	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/teams"
	"github.com/trick77/noergler/internal/webhook"
)

const (
	testSlug   = "platform"
	testSecret = "plat-secret"
	testBot    = "noergler"
)

// fakeReviewer records which handler the dispatch table picked.
type fakeReviewer struct{ calls []string }

func (f *fakeReviewer) ReviewPullRequest(context.Context, *webhook.Payload, bool) {
	f.calls = append(f.calls, "review")
}
func (f *fakeReviewer) HandleMention(context.Context, *webhook.Payload) {
	f.calls = append(f.calls, "mention")
}
func (f *fakeReviewer) HandleCommentDeleted(context.Context, *webhook.Payload) {
	f.calls = append(f.calls, "comment-deleted")
}
func (f *fakeReviewer) HandlePRMerged(context.Context, *webhook.Payload) {
	f.calls = append(f.calls, "merged")
}
func (f *fakeReviewer) HandlePRDeclined(context.Context, *webhook.Payload) {
	f.calls = append(f.calls, "declined")
}
func (f *fakeReviewer) HandlePRDeleted(context.Context, *webhook.Payload) {
	f.calls = append(f.calls, "deleted")
}
func (f *fakeReviewer) IsAutoReviewAuthor(string) bool { return true }
func (f *fakeReviewer) SetAuthorLists(_, _ []string)   {}
func (f *fakeReviewer) String() string                 { return strings.Join(f.calls, ",") }

// fakeQueue captures submissions without running a worker. Jobs are run on
// demand so a test can assert which reviewer method the closure calls.
type fakeQueue struct {
	keys    []store.PRKey
	tags    []string
	jobs    []queue.JobFunc
	outcome string
}

func (q *fakeQueue) Submit(key store.PRKey, _ *webhook.Payload, _ string) string {
	q.keys = append(q.keys, key)
	if q.outcome != "" {
		return q.outcome
	}
	return "queued"
}

func (q *fakeQueue) SubmitJob(key store.PRKey, _ string, fn queue.JobFunc) string {
	q.tags = append(q.tags, key.Tag())
	q.jobs = append(q.jobs, fn)
	return "queued"
}

func (q *fakeQueue) runAll() {
	for _, fn := range q.jobs {
		fn(context.Background())
	}
}

type harness struct {
	srv  *httpapi.Server
	q    *fakeQueue
	rv   *fakeReviewer
	rt   *teams.Runtime
	reg  *teams.Registry
	log  *slog.Logger
	logs *bytes.Buffer
}

// setStore re-registers the routes with a settings store. The mux refuses a
// duplicate pattern, so this builds a fresh server.
func (h *harness) setStore(t *testing.T, db SettingsStore) {
	t.Helper()
	h.srv = newServer(h)
	Register(h.srv, Deps{
		Teams: h.reg, Queue: h.q, Log: h.log, Store: db, BotUsername: testBot,
	})
}

func newServer(h *harness) *httpapi.Server { return httpapi.New(h.reg.Status, h.log) }

func newRequest(method, path, body string) *http.Request {
	return httptest.NewRequest(method, path, bytes.NewReader([]byte(body)))
}

// serve runs one request and returns the recorded response.
func (h *harness) serve(r *http.Request) *http.Response {
	w := httptest.NewRecorder()
	h.srv.Handler().ServeHTTP(w, r)
	return w.Result()
}

func (h *harness) body(w *http.Response) string {
	b, _ := io.ReadAll(w.Body)
	w.Body = io.NopCloser(bytes.NewReader(b))
	return string(b)
}

func (h *harness) decode(t *testing.T, w *http.Response) map[string]any {
	t.Helper()
	var got map[string]any
	if err := json.Unmarshal([]byte(h.body(w)), &got); err != nil {
		t.Fatalf("body: %v", err)
	}
	return got
}

// newHarness wires one enabled team owning PLAT and one disabled team.
func newHarness(t *testing.T, tweak func(*config.Team)) *harness {
	t.Helper()
	team := &config.Team{
		Slug:          testSlug,
		WebhookSecret: testSecret,
		Projects:      []config.ProjectScope{{Key: "PLAT"}},
	}
	if tweak != nil {
		tweak(team)
	}
	rv := &fakeReviewer{}
	rt := teams.NewRuntime(team, rv, nil, nil, nil)

	var buf bytes.Buffer
	log := slog.New(logging.NewHandler(&buf, slog.LevelDebug, "test"))
	reg := teams.NewRegistry(
		map[string]*teams.Runtime{testSlug: rt},
		map[string]string{"payments": "LLM check failed"},
		log,
	)
	q := &fakeQueue{}
	srv := httpapi.New(reg.Status, log)
	Register(srv, Deps{Teams: reg, Queue: q, Log: log, BotUsername: testBot})
	return &harness{srv: srv, q: q, rv: rv, rt: rt, reg: reg, log: log, logs: &buf}
}

func sign(body []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}

// post sends a signed delivery unless headers override it.
func (h *harness) post(t *testing.T, slug string, body []byte, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/webhook/"+slug, bytes.NewReader(body))
	for k, v := range headers {
		if v == "" {
			continue
		}
		r.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	h.srv.Handler().ServeHTTP(w, r)
	return w
}

func (h *harness) signedPost(t *testing.T, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	return h.post(t, testSlug, body, map[string]string{"X-Hub-Signature": sign(body, testSecret)})
}

// payloadJSON builds a delivery for one event on PLAT/svc.
func payloadJSON(event string, extra string) []byte {
	base := fmt.Sprintf(`{"eventKey":%q,
		"pullRequest":{"id":42,"title":"t","state":"OPEN",
		"fromRef":{"id":"refs/heads/f","displayId":"f","latestCommit":"src1",
			"repository":{"slug":"svc","project":{"key":"PLAT"}}},
		"toRef":{"id":"refs/heads/main","displayId":"main","latestCommit":"dst1",
			"repository":{"slug":"svc","project":{"key":"PLAT"}}},
		"author":{"user":{"name":"alice"}}}%s}`, event, extra)
	return []byte(base)
}

func decodeBody(t *testing.T, w *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var got map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("body %q: %v", w.Body.String(), err)
	}
	return got
}

func wantBody(t *testing.T, w *httptest.ResponseRecorder, code int, want map[string]any) {
	t.Helper()
	if w.Code != code {
		t.Errorf("status = %d, want %d (body %s)", w.Code, code, w.Body.String())
	}
	if got := decodeBody(t, w); !reflect.DeepEqual(got, want) {
		t.Errorf("body = %#v, want %#v", got, want)
	}
}

// --- team routing -----------------------------------------------------------

func TestWebhook_UnknownTeamIs404AndDisabledIs503(t *testing.T) {
	h := newHarness(t, nil)
	body := payloadJSON(webhook.EventOpened, "")

	w := h.post(t, "nope", body, map[string]string{"X-Hub-Signature": sign(body, testSecret)})
	wantBody(t, w, http.StatusNotFound, map[string]any{"detail": "unknown team"})

	w = h.post(t, "payments", body, map[string]string{"X-Hub-Signature": sign(body, testSecret)})
	wantBody(t, w, http.StatusServiceUnavailable, map[string]any{
		"detail": "team payments is disabled, see the noergler startup log",
	})
}

// The team is bound from the path before anything else, so even the ping
// 503s for a disabled team.
func TestWebhook_DisabledTeamRejectsEvenTheDiagnosticsPing(t *testing.T) {
	h := newHarness(t, nil)
	w := h.post(t, "payments", nil, map[string]string{"X-Event-Key": "diagnostics:ping"})
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
}

// --- the signature-free paths -----------------------------------------------

func TestWebhook_DiagnosticsPingAnsweredBeforeTheBodyIsRead(t *testing.T) {
	h := newHarness(t, nil)
	w := h.post(t, testSlug, []byte("not even json"), map[string]string{"X-Event-Key": "diagnostics:ping"})
	wantBody(t, w, http.StatusOK, map[string]any{"status": "ok"})
}

func TestWebhook_TestConnectionWithoutSignatureOrEventKey(t *testing.T) {
	h := newHarness(t, nil)
	w := h.post(t, testSlug, []byte(`{"test": true}`), nil)
	wantBody(t, w, http.StatusOK, map[string]any{"status": "ok"})
}

// A body that mentions eventKey is a real delivery, so a missing signature
// is a 401 rather than the test-connection shortcut.
func TestWebhook_MissingSignatureIs401WhenTheBodyLooksLikeADelivery(t *testing.T) {
	h := newHarness(t, nil)
	w := h.post(t, testSlug, payloadJSON(webhook.EventOpened, ""), nil)
	wantBody(t, w, http.StatusUnauthorized, map[string]any{"detail": "Missing signature"})
}

func TestWebhook_MissingSignatureWithAnEventKeyHeaderIs401(t *testing.T) {
	h := newHarness(t, nil)
	w := h.post(t, testSlug, []byte(`{}`), map[string]string{"X-Event-Key": "pr:opened"})
	wantBody(t, w, http.StatusUnauthorized, map[string]any{"detail": "Missing signature"})
}

// --- signature verification -------------------------------------------------

func TestWebhook_SignatureVerification(t *testing.T) {
	body := payloadJSON(webhook.EventOpened, "")
	good := sign(body, testSecret)

	cases := []struct {
		name string
		sig  string
		want int
	}{
		{"valid", good, http.StatusOK},
		{"valid without the sha256 prefix", strings.TrimPrefix(good, "sha256="), http.StatusOK},
		{"another team's secret", sign(body, "pay-secret"), http.StatusUnauthorized},
		// The comparison is on the hex strings, so uppercase hex must fail
		// even though it decodes to the same bytes. Only the digest is
		// uppercased:
		// uppercasing the whole header would fail on the prefix instead and
		// the case would prove nothing.
		{"uppercase hex", "sha256=" + strings.ToUpper(strings.TrimPrefix(good, "sha256=")), http.StatusUnauthorized},
		{"garbage", "sha256=not-hex", http.StatusUnauthorized},
		{"empty after the prefix", "sha256=", http.StatusUnauthorized},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, nil)
			w := h.post(t, testSlug, body, map[string]string{"X-Hub-Signature": c.sig})
			if w.Code != c.want {
				t.Errorf("status = %d, want %d (body %s)", w.Code, c.want, w.Body.String())
			}
		})
	}
}

// A signature over different bytes must fail: the HMAC covers the raw body.
func TestWebhook_SignatureIsOverTheExactBytes(t *testing.T) {
	h := newHarness(t, nil)
	body := payloadJSON(webhook.EventOpened, "")
	other := payloadJSON(webhook.EventMerged, "")
	w := h.post(t, testSlug, body, map[string]string{"X-Hub-Signature": sign(other, testSecret)})
	if w.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", w.Code)
	}
}

// --- payload handling -------------------------------------------------------

// A non-PR event is ignored with a 200 even though its shape would never
// satisfy the PR payload checks. The prefix test runs before validation.
func TestWebhook_NonPREventIsIgnoredNotRefused(t *testing.T) {
	h := newHarness(t, nil)
	body := []byte(`{"eventKey":"repo:refs_changed","changes":[{"ref":{"id":"x"}}]}`)
	w := h.signedPost(t, body)
	wantBody(t, w, http.StatusOK, map[string]any{
		"status": "ignored", "reason": "not a PR event: repo:refs_changed",
	})
}

func TestWebhook_UnparseablePRPayloadIs400(t *testing.T) {
	h := newHarness(t, nil)
	// A pr: event missing every required field.
	w := h.signedPost(t, []byte(`{"eventKey":"pr:opened"}`))
	wantBody(t, w, http.StatusBadRequest, map[string]any{"detail": "Invalid payload"})
}

func TestWebhook_MissingRepositoryIsIgnored(t *testing.T) {
	h := newHarness(t, nil)
	body := []byte(`{"eventKey":"pr:opened",
		"pullRequest":{"id":42,"title":"t",
		"fromRef":{"id":"a","displayId":"a"},
		"toRef":{"id":"b","displayId":"b"},
		"author":{"user":{"name":"alice"}}}}`)
	w := h.signedPost(t, body)
	wantBody(t, w, http.StatusOK, map[string]any{"status": "ignored", "reason": "missing repository"})
}

// --- the ownership check ----------------------------------------------------

// The invariant: holding a team's secret does not make another team's repo
// yours. Without this a team signs a payload naming a repo it does not own
// and has it reviewed on its own key.
func TestWebhook_SignedPayloadForAnotherTeamsRepoIs403(t *testing.T) {
	h := newHarness(t, nil)
	body := []byte(`{"eventKey":"pr:opened",
		"pullRequest":{"id":42,"title":"t",
		"fromRef":{"id":"a","displayId":"a","repository":{"slug":"wallet","project":{"key":"PAY"}}},
		"toRef":{"id":"b","displayId":"b","repository":{"slug":"wallet","project":{"key":"PAY"}}},
		"author":{"user":{"name":"alice"}}}}`)

	w := h.signedPost(t, body)
	wantBody(t, w, http.StatusForbidden, map[string]any{
		"detail": "repository PAY/wallet is not owned by team platform",
	})
	if len(h.q.keys) != 0 || len(h.q.tags) != 0 {
		t.Error("a rejected payload must not reach the queue")
	}
}

// --- exclude_repos ----------------------------------------------------------

func TestWebhook_ExcludedRepoGatesOnlyReviewStartingEvents(t *testing.T) {
	excluded := func(team *config.Team) { team.Review.ExcludeRepos = []string{"*-infra"} }

	// Review-starting events are ignored for an excluded repo.
	for _, event := range []string{webhook.EventOpened, webhook.EventFromRefUpdated, webhook.EventCommentAdded} {
		t.Run("ignored "+event, func(t *testing.T) {
			h := newHarness(t, excluded)
			body := infraPayload(event)
			w := h.signedPost(t, body)
			wantBody(t, w, http.StatusOK, map[string]any{
				"status": "ignored", "reason": "repo excluded by the team's exclude_repos",
			})
		})
	}

	// Lifecycle events still pass: a PR reviewed before the pattern was set
	// must still be marked merged and get its rollup.
	for _, event := range []string{webhook.EventMerged, webhook.EventDeclined, webhook.EventDeleted, webhook.EventCommentDeleted} {
		t.Run("passes "+event, func(t *testing.T) {
			h := newHarness(t, excluded)
			w := h.signedPost(t, infraPayload(event))
			if w.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200", w.Code)
			}
			if got := decodeBody(t, w)["status"]; got != "accepted" {
				t.Errorf("status = %v, want accepted (body %s)", got, w.Body.String())
			}
		})
	}
}

// An explicitly claimed repo beats the glob.
func TestWebhook_ExplicitRepoClaimBeatsExcludeRepos(t *testing.T) {
	h := newHarness(t, func(team *config.Team) {
		team.Projects = []config.ProjectScope{{Key: "PLAT", Repos: []string{"core-infra"}}}
		team.Review.ExcludeRepos = []string{"*-infra"}
	})
	w := h.signedPost(t, infraPayload(webhook.EventOpened))
	if w.Code != http.StatusOK || decodeBody(t, w)["status"] != "accepted" {
		t.Errorf("an explicitly claimed repo must be reviewed: %d %s", w.Code, w.Body.String())
	}
}

func infraPayload(event string) []byte {
	return []byte(fmt.Sprintf(`{"eventKey":%q,
		"pullRequest":{"id":42,"title":"t",
		"fromRef":{"id":"a","displayId":"a","repository":{"slug":"core-infra","project":{"key":"PLAT"}}},
		"toRef":{"id":"b","displayId":"b","repository":{"slug":"core-infra","project":{"key":"PLAT"}}},
		"author":{"user":{"name":"alice"}}},
		"comment":{"id":7,"text":"@noergler hi","author":{"name":"alice"}}}`, event))
}

// --- dispatch ---------------------------------------------------------------

func TestWebhook_DispatchTable(t *testing.T) {
	cases := []struct {
		event    string
		extra    string
		wantBody map[string]any
		wantCall string
	}{
		{webhook.EventOpened, "",
			map[string]any{"status": "accepted", "pr_id": float64(42), "queue": "queued"}, "review"},
		{webhook.EventFromRefUpdated, "",
			map[string]any{"status": "accepted", "pr_id": float64(42), "queue": "queued"}, "review"},
		{webhook.EventMerged, "",
			map[string]any{"status": "accepted", "reason": "merged-rollup", "queue": "queued"}, "merged"},
		{webhook.EventDeclined, "",
			map[string]any{"status": "accepted", "reason": "declined-rollup", "queue": "queued"}, "declined"},
		// These two keep their body shape: no queue key, even though both go
		// on the queue.
		{webhook.EventDeleted, "",
			map[string]any{"status": "accepted", "reason": "deleted-purge"}, "deleted"},
		{webhook.EventCommentDeleted, `,"comment":{"id":7,"text":"x","author":{"name":"bob"}}`,
			map[string]any{"status": "accepted", "reason": "comment-deleted"}, "comment-deleted"},
		{webhook.EventCommentAdded, `,"comment":{"id":7,"text":"@noergler please look","author":{"name":"bob"}}`,
			map[string]any{"status": "accepted", "reason": "mention", "queue": "queued"}, "mention"},
	}
	for _, c := range cases {
		t.Run(c.event, func(t *testing.T) {
			h := newHarness(t, nil)
			w := h.signedPost(t, payloadJSON(c.event, c.extra))
			wantBody(t, w, http.StatusOK, c.wantBody)

			if c.wantCall == "review" {
				want := store.PRKey{Project: "PLAT", Repo: "svc", PRID: 42}
				if len(h.q.keys) != 1 || h.q.keys[0] != want {
					t.Fatalf("Submit keys = %+v, want one %+v", h.q.keys, want)
				}
				return
			}
			if len(h.q.tags) != 1 || h.q.tags[0] != "PLAT/svc#42" {
				t.Fatalf("SubmitJob tags = %v, want one PLAT/svc#42", h.q.tags)
			}
			h.q.runAll()
			if got := h.rv.String(); got != c.wantCall {
				t.Errorf("reviewer calls = %q, want %q", got, c.wantCall)
			}
		})
	}
}

// Submit's outcome is passed through, so a superseded push reports it.
func TestWebhook_SupersededOutcomeIsReported(t *testing.T) {
	h := newHarness(t, nil)
	h.q.outcome = "superseded"
	w := h.signedPost(t, payloadJSON(webhook.EventFromRefUpdated, ""))
	wantBody(t, w, http.StatusOK, map[string]any{
		"status": "accepted", "pr_id": float64(42), "queue": "superseded",
	})
}

func TestWebhook_UnhandledPREventIsIgnoredAndWarns(t *testing.T) {
	h := newHarness(t, nil)
	w := h.signedPost(t, payloadJSON("pr:reviewer:approved", ""))
	wantBody(t, w, http.StatusOK, map[string]any{
		"status": "ignored", "reason": "unhandled event: pr:reviewer:approved",
	})
	// The signal that a Bitbucket hook is configured with events we ignore.
	if !strings.Contains(h.logs.String(), `Unhandled event \"pr:reviewer:approved\"`) {
		t.Errorf("missing the unhandled-event warning: %s", h.logs.String())
	}
	if len(h.q.tags) != 0 || len(h.q.keys) != 0 {
		t.Error("an unhandled event must not reach the queue")
	}
}

// --- the mention gate -------------------------------------------------------

// HandleMention does not check that the comment names the bot, so this gate
// is what stops every comment on every PR becoming an inference call.
func TestWebhook_MentionGate(t *testing.T) {
	cases := []struct {
		name   string
		text   string
		accept bool
	}{
		{"plain mention", "@noergler please review", true},
		{"uppercase", "@NOERGLER please review", true},
		{"mixed case", "@NoErGlEr hi", true},
		// A case-insensitive substring test, no word boundary.
		{"inside a longer word", "ask @noerglerbot about it", true},
		{"mid sentence", "cc @noergler on this", true},
		{"no mention", "looks good to me", false},
		{"another bot", "@someone-else look", false},
		{"empty text", "", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, nil)
			extra := fmt.Sprintf(`,"comment":{"id":7,"text":%q,"author":{"name":"bob"}}`, c.text)
			w := h.signedPost(t, payloadJSON(webhook.EventCommentAdded, extra))
			got := decodeBody(t, w)
			if c.accept {
				if got["status"] != "accepted" || got["reason"] != "mention" {
					t.Errorf("body = %v, want an accepted mention", got)
				}
				return
			}
			if got["status"] != "ignored" || got["reason"] != "comment without mention" {
				t.Errorf("body = %v, want ignored", got)
			}
			if len(h.q.tags) != 0 {
				t.Error("a comment without a mention must not reach the queue")
			}
		})
	}
}

// A comment event with no comment object must not panic.
func TestWebhook_CommentEventWithoutACommentObject(t *testing.T) {
	h := newHarness(t, nil)
	w := h.signedPost(t, payloadJSON(webhook.EventCommentAdded, ""))
	wantBody(t, w, http.StatusOK, map[string]any{
		"status": "ignored", "reason": "comment without mention",
	})
}

// An empty comment.text is accepted and answers "comment without mention"
// rather than refusing the payload with a 400.
func TestWebhook_EmptyCommentTextIsIgnoredNotRefused(t *testing.T) {
	h := newHarness(t, nil)
	extra := `,"comment":{"id":7,"text":"","author":{"name":"bob"}}`
	w := h.signedPost(t, payloadJSON(webhook.EventCommentAdded, extra))
	wantBody(t, w, http.StatusOK, map[string]any{
		"status": "ignored", "reason": "comment without mention",
	})
}

// The real Reviewer must satisfy the interface the fake stands in for.
var _ teams.Reviewer = (*review.Reviewer)(nil)
