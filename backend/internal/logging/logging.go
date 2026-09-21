// Package logging is the Splunk-friendly JSON log: one object per line,
// `timestamp` first, `msg`, `log_level`, `service`, `env`, then whatever the
// context binds (team, request_id, pr_tag, ...) and the record's own attrs.
//
// Field names Splunk treats as built-in input metadata are renamed to
// `splunk_<key>` rather than dropped: using them as JSON keys makes Splunk
// silently overwrite the value with the forwarder's (`source`, `host`, `index`,
// `sourcetype`, `time`/`_time`/`_raw`) or double-extract it (`event`).
package logging

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"
)

const serviceName = "noergler"

var splunkReserved = map[string]bool{
	"source": true, "sourcetype": true, "host": true, "index": true,
	"time": true, "_time": true, "_raw": true, "event": true,
}

// Keys the handler writes itself; an attr with one of these names would
// collide, so it is renamed the same way as a Splunk-reserved key.
var ownKeys = map[string]bool{
	"timestamp": true, "msg": true, "log_level": true, "service": true, "env": true,
}

type ctxKey struct{}

// With returns a context whose log lines carry attrs. Bound at each boundary
// (webhook route, queue worker, per-team startup) so every line about a team
// says `team=<slug>` without each call site repeating it. Later bindings win
// over earlier ones with the same key.
func With(ctx context.Context, attrs ...any) context.Context {
	if len(attrs) == 0 {
		return ctx
	}
	parent, _ := ctx.Value(ctxKey{}).([]slog.Attr)
	bound := make([]slog.Attr, 0, len(parent)+len(attrs)/2)
	bound = append(bound, parent...)
	var rec slog.Record
	rec.Add(attrs...)
	rec.Attrs(func(a slog.Attr) bool {
		bound = append(bound, a)
		return true
	})
	return context.WithValue(ctx, ctxKey{}, bound)
}

// WithTeam binds the team slug, the one field every team-related line needs.
func WithTeam(ctx context.Context, slug string) context.Context {
	return With(ctx, "team", slug)
}

// Bound returns the attrs a context carries, for tests and for handing the
// same fields to another logger.
func Bound(ctx context.Context) []slog.Attr {
	attrs, _ := ctx.Value(ctxKey{}).([]slog.Attr)
	return attrs
}

// ParseLevel reads LOG_LEVEL: DEBUG, INFO, WARNING (or WARN), ERROR, in any
// case. Anything else is INFO.
func ParseLevel(s string) slog.Level {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "DEBUG":
		return slog.LevelDebug
	case "WARNING", "WARN":
		return slog.LevelWarn
	case "ERROR":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

// Handler is the JSON handler. Safe for concurrent use.
type Handler struct {
	mu    *sync.Mutex
	w     io.Writer
	level slog.Level
	env   string
	now   func() time.Time
	// attrs and group carry what WithAttrs/WithGroup accumulated.
	attrs []slog.Attr
	group string
}

// Option tunes a Handler.
type Option func(*Handler)

// WithClock injects the timestamp source; tests pin it.
func WithClock(now func() time.Time) Option { return func(h *Handler) { h.now = now } }

// NewHandler writes JSON lines to w at level, stamping env on every line.
func NewHandler(w io.Writer, level slog.Level, env string, opts ...Option) *Handler {
	h := &Handler{mu: &sync.Mutex{}, w: w, level: level, env: env, now: time.Now}
	for _, o := range opts {
		o(h)
	}
	return h
}

// Setup installs the handler as slog's default, reading LOG_LEVEL and
// NOERGLER_ENV, and returns the logger. Called before config loads, because
// config loading logs.
func Setup() *slog.Logger {
	h := NewHandler(os.Stdout, ParseLevel(os.Getenv("LOG_LEVEL")), envName())
	l := slog.New(h)
	slog.SetDefault(l)
	return l
}

func envName() string {
	if v := os.Getenv("NOERGLER_ENV"); v != "" {
		return v
	}
	return "dev"
}

// Enabled reports whether the given level is enabled.
func (h *Handler) Enabled(_ context.Context, l slog.Level) bool { return l >= h.level }

// WithAttrs returns a new handler with the given attributes added.
func (h *Handler) WithAttrs(attrs []slog.Attr) slog.Handler {
	c := *h
	c.attrs = append(append([]slog.Attr(nil), h.attrs...), qualify(h.group, attrs)...)
	return &c
}

// WithGroup returns a new handler with the given group.
func (h *Handler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	c := *h
	if h.group != "" {
		c.group = h.group + "." + name
	} else {
		c.group = name
	}
	return &c
}

// qualify flattens a group into dotted keys: Splunk fields are flat, and a
// nested object would hide `team` from the field extractor.
func qualify(group string, attrs []slog.Attr) []slog.Attr {
	if group == "" {
		return attrs
	}
	out := make([]slog.Attr, len(attrs))
	for i, a := range attrs {
		out[i] = slog.Attr{Key: group + "." + a.Key, Value: a.Value}
	}
	return out
}

func levelName(l slog.Level) string {
	switch {
	case l >= slog.LevelError:
		return "error"
	case l >= slog.LevelWarn:
		return "warning"
	case l >= slog.LevelInfo:
		return "info"
	default:
		return "debug"
	}
}

// Handle renders one line. Key order is deliberate: Splunk's auto timestamp
// search only looks 128 chars into the event and falls back to index time
// when it misses, so `timestamp` is first, before any bound field.
func (h *Handler) Handle(ctx context.Context, r slog.Record) error {
	var buf bytes.Buffer
	buf.WriteByte('{')
	ts := r.Time
	if ts.IsZero() {
		ts = h.now()
	}
	writeField(&buf, "timestamp", ts.UTC().Format("2006-01-02T15:04:05.000000Z"), true)
	writeField(&buf, "msg", r.Message, false)
	writeField(&buf, "log_level", levelName(r.Level), false)
	writeField(&buf, "service", serviceName, false)
	writeField(&buf, "env", h.env, false)

	seen := map[string]bool{}
	write := func(a slog.Attr) {
		a.Value = a.Value.Resolve()
		if a.Equal(slog.Attr{}) {
			return
		}
		key := a.Key
		if splunkReserved[key] || ownKeys[key] {
			key = "splunk_" + key
		}
		if seen[key] {
			return
		}
		seen[key] = true
		writeField(&buf, key, attrValue(a.Value), false)
	}
	// Bound fields first, latest binding winning; then the handler's own
	// WithAttrs, then the record's. A record attr never overrides a bound
	// one: `team` is the boundary's to set.
	bound := Bound(ctx)
	last := make(map[string]int, len(bound))
	for i, a := range bound {
		last[a.Key] = i
	}
	for i, a := range bound {
		if last[a.Key] == i {
			write(a)
		}
	}
	for _, a := range h.attrs {
		write(a)
	}
	r.Attrs(func(a slog.Attr) bool {
		write(slog.Attr{Key: qualifyKey(h.group, a.Key), Value: a.Value})
		return true
	})
	buf.WriteString("}\n")

	h.mu.Lock()
	defer h.mu.Unlock()
	_, err := h.w.Write(buf.Bytes())
	return err
}

func qualifyKey(group, key string) string {
	if group == "" {
		return key
	}
	return group + "." + key
}

func writeField(buf *bytes.Buffer, key string, v any, first bool) {
	if !first {
		buf.WriteByte(',')
	}
	k, _ := json.Marshal(key)
	buf.Write(k)
	buf.WriteByte(':')
	b, err := json.Marshal(v)
	if err != nil {
		b, _ = json.Marshal(err.Error())
	}
	buf.Write(b)
}

// attrValue turns a slog value into something json.Marshal renders the way
// the log format requires: durations as their string, errors as text, groups as
// objects.
func attrValue(v slog.Value) any {
	switch v.Kind() {
	case slog.KindString:
		return v.String()
	case slog.KindInt64:
		return v.Int64()
	case slog.KindUint64:
		return v.Uint64()
	case slog.KindFloat64:
		return v.Float64()
	case slog.KindBool:
		return v.Bool()
	case slog.KindDuration:
		return v.Duration().String()
	case slog.KindTime:
		return v.Time().UTC().Format(time.RFC3339Nano)
	case slog.KindGroup:
		m := map[string]any{}
		for _, a := range v.Group() {
			m[a.Key] = attrValue(a.Value.Resolve())
		}
		return m
	default:
		a := v.Any()
		if err, ok := a.(error); ok {
			return err.Error()
		}
		return a
	}
}
