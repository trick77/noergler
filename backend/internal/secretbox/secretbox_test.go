package secretbox

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func box(t *testing.T, fill byte) *Box {
	t.Helper()
	b, err := New(bytes.Repeat([]byte{fill}, KeySize))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestRoundTrip(t *testing.T) {
	b := box(t, 1)
	nonce, ct, err := b.Seal([]byte("token"), []byte("platform|bitbucket"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(ct, []byte("token")) {
		t.Fatal("ciphertext holds the plaintext")
	}
	got, err := b.Open(nonce, ct, []byte("platform|bitbucket"))
	if err != nil || string(got) != "token" {
		t.Fatalf("Open = %q, %v", got, err)
	}
}

// A row copied to another team, or another kind, must not open.
func TestOpen_WrongAADFails(t *testing.T) {
	b := box(t, 1)
	nonce, ct, _ := b.Seal([]byte("token"), []byte("platform|bitbucket"))
	for _, aad := range []string{"payments|bitbucket", "platform|jira"} {
		if _, err := b.Open(nonce, ct, []byte(aad)); !errors.Is(err, ErrOpen) {
			t.Errorf("aad %q: err = %v, want ErrOpen", aad, err)
		}
	}
}

func TestOpen_WrongKeyFails(t *testing.T) {
	nonce, ct, _ := box(t, 1).Seal([]byte("token"), []byte("a|b"))
	if _, err := box(t, 2).Open(nonce, ct, []byte("a|b")); !errors.Is(err, ErrOpen) {
		t.Fatalf("err = %v, want ErrOpen", err)
	}
}

func TestOpen_DamagedRowFails(t *testing.T) {
	b := box(t, 1)
	nonce, ct, _ := b.Seal([]byte("token"), []byte("a|b"))
	ct[0] ^= 0xff
	if _, err := b.Open(nonce, ct, []byte("a|b")); !errors.Is(err, ErrOpen) {
		t.Fatalf("err = %v, want ErrOpen", err)
	}
	if _, err := b.Open(nonce[:3], ct, []byte("a|b")); !errors.Is(err, ErrOpen) {
		t.Fatalf("short nonce: err = %v, want ErrOpen", err)
	}
}

func TestParseKey(t *testing.T) {
	if _, err := ParseKey(strings.Repeat("ab", KeySize)); err != nil {
		t.Fatalf("valid key: %v", err)
	}
	for _, bad := range []string{"", "zz", strings.Repeat("ab", 16)} {
		if _, err := ParseKey(bad); err == nil {
			t.Errorf("ParseKey(%q) accepted", bad)
		}
	}
}
