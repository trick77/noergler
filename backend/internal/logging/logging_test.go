package logging

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"testing"
	"time"
)

func newTestLogger(level slog.Level) (*slog.Logger, *bytes.Buffer) {
	var buf bytes.Buffer
	fixed := func() time.Time { return time.Date(2026, 9, 19, 10, 0, 0, 0, time.UTC) }
	return slog.New(NewHandler(&buf, level, "test", WithClock(fixed))), &buf
}

func decode(t *testing.T, line string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		t.Fatalf("not JSON: %q: %v", line, err)
	}
	return m
}

func TestHandler_TimestampIsFirstAndSchemaFieldsPresent(t *testing.T) {
	log, buf := newTestLogger(slog.LevelInfo)
	log.Info("hello", "team", "platform")
	line := strings.TrimSuffix(buf.String(), "\n")
	if !strings.HasPrefix(line, `{"timestamp":"`) {
		t.Errorf("timestamp must be the first key: %s", line)
	}
	m := decode(t, line)
	for k, want := range map[string]any{"msg": "hello", "log_level": "info", "service": "noergler", "env": "test", "team": "platform"} {
		if m[k] != want {
			t.Errorf("%s = %v, want %v", k, m[k], want)
		}
	}
	if strings.Count(buf.String(), "\n") != 1 {
		t.Error("one line per record")
	}
}

func TestHandler_LevelNamesArePinned(t *testing.T) {
	log, buf := newTestLogger(slog.LevelDebug)
	log.Debug("d")
	log.Info("i")
	log.Warn("w")
	log.Error("e")
	var got []string
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		got = append(got, decode(t, line)["log_level"].(string))
	}
	if strings.Join(got, ",") != "debug,info,warning,error" {
		t.Errorf("levels = %v", got)
	}
}

func TestHandler_ReservedKeysAreRenamed(t *testing.T) {
	log, buf := newTestLogger(slog.LevelInfo)
	log.Info("x", "source", "jenkins", "host", "h", "event", "e", "timestamp", "fake", "msg", "fake")
	m := decode(t, buf.String())
	if m["splunk_source"] != "jenkins" || m["splunk_host"] != "h" || m["splunk_event"] != "e" {
		t.Errorf("reserved keys not renamed: %v", m)
	}
	if m["splunk_timestamp"] != "fake" || m["msg"] != "x" || m["splunk_msg"] != "fake" {
		t.Errorf("own keys must not be overridden by attrs: %v", m)
	}
	if _, ok := m["source"]; ok {
		t.Error("reserved key must not survive")
	}
}

func TestHandler_ContextBindingsAppearAndLatestWins(t *testing.T) {
	log, buf := newTestLogger(slog.LevelInfo)
	ctx := WithTeam(context.Background(), "a")
	ctx = With(ctx, "request_id", "r1", "team", "b")
	log.InfoContext(ctx, "x", "extra", 1)
	m := decode(t, buf.String())
	if m["team"] != "b" || m["request_id"] != "r1" || m["extra"] != float64(1) {
		t.Errorf("bindings: %v", m)
	}
	// A record attr never overrides a bound key.
	buf.Reset()
	log.InfoContext(ctx, "x", "team", "record")
	if m := decode(t, buf.String()); m["team"] != "b" {
		t.Errorf("record attr overrode the bound team: %v", m)
	}
	// The parent context is untouched.
	buf.Reset()
	log.InfoContext(WithTeam(context.Background(), "a"), "x")
	if m := decode(t, buf.String()); m["team"] != "a" || m["request_id"] != nil {
		t.Errorf("parent context leaked: %v", m)
	}
}

func TestHandler_LevelFilterAndValueKinds(t *testing.T) {
	log, buf := newTestLogger(slog.LevelWarn)
	log.Info("dropped")
	if buf.Len() != 0 {
		t.Fatal("info must be filtered at WARN")
	}
	log.With("component", "q").WithGroup("g").Warn("kept", "err", errors.New("boom"), "took", 1500*time.Millisecond, "n", 2.5)
	m := decode(t, buf.String())
	if m["component"] != "q" || m["g.err"] != "boom" || m["g.took"] != "1.5s" || m["g.n"] != 2.5 {
		t.Errorf("values: %v", m)
	}
}

func TestParseLevel(t *testing.T) {
	cases := map[string]slog.Level{"debug": slog.LevelDebug, "INFO": slog.LevelInfo, "Warning": slog.LevelWarn,
		"WARN": slog.LevelWarn, "error": slog.LevelError, "": slog.LevelInfo, "bogus": slog.LevelInfo}
	for in, want := range cases {
		if got := ParseLevel(in); got != want {
			t.Errorf("ParseLevel(%q) = %v, want %v", in, got, want)
		}
	}
}
