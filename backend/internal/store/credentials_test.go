package store

import (
	"context"
	"reflect"
	"testing"
)

func TestCredentials_RoundTrip(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if got, err := s.ListCredentials(ctx); err != nil || len(got) != 0 {
		t.Fatalf("empty = %v %v", got, err)
	}
	bb := SealedCredential{Team: "platform", Kind: "bitbucket", Username: "plat-bot", Nonce: []byte{1}, Ciphertext: []byte{2}}
	jr := SealedCredential{Team: "platform", Kind: "jira", Nonce: []byte{3}, Ciphertext: []byte{4}}
	if err := s.PutCredentials(ctx, []SealedCredential{bb, jr}, "team:platform"); err != nil {
		t.Fatal(err)
	}
	got, _ := s.ListCredentials(ctx)
	if !reflect.DeepEqual(got, []SealedCredential{bb, jr}) {
		t.Fatalf("list = %+v", got)
	}

	// A second push replaces, never adds.
	bb.Username, bb.Ciphertext = "plat-bot-2", []byte{9}
	_ = s.PutCredentials(ctx, []SealedCredential{bb}, "team:platform")
	got, _ = s.ListCredentials(ctx)
	if len(got) != 2 || got[0].Username != "plat-bot-2" || got[0].Ciphertext[0] != 9 {
		t.Fatalf("after replace = %+v", got)
	}

	if ok, err := s.DeleteCredential(ctx, "platform", "jira", "team:platform"); !ok || err != nil {
		t.Fatalf("delete = %v %v", ok, err)
	}
	if ok, _ := s.DeleteCredential(ctx, "platform", "jira", "team:platform"); ok {
		t.Fatal("second delete reported a row")
	}
	got, _ = s.ListCredentials(ctx)
	if len(got) != 1 || got[0].Kind != "bitbucket" {
		t.Fatalf("after delete = %+v", got)
	}
}

// All or none: a bad second row takes the good first one down with it.
func TestCredentials_PutIsAllOrNone(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	err := s.PutCredentials(ctx, []SealedCredential{
		{Team: "a", Kind: "bitbucket", Username: "a-bot", Nonce: []byte{1}, Ciphertext: []byte{1}},
		{Team: "a", Kind: "riptide", Nonce: []byte{1}, Ciphertext: []byte{1}},
	}, "x")
	if err == nil {
		t.Fatal("kind outside bitbucket/jira accepted")
	}
	if got, _ := s.ListCredentials(ctx); len(got) != 0 {
		t.Errorf("half a push landed: %+v", got)
	}
}

func TestKeyCheck_RoundTrip(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if _, _, found, err := s.KeyCheck(ctx); found || err != nil {
		t.Fatalf("empty = %v %v", found, err)
	}
	_ = s.PutKeyCheck(ctx, []byte{1}, []byte{2})
	_ = s.PutKeyCheck(ctx, []byte{3}, []byte{4})
	nonce, ct, found, err := s.KeyCheck(ctx)
	if !found || err != nil || nonce[0] != 3 || ct[0] != 4 {
		t.Fatalf("key check = %v %v %v %v", nonce, ct, found, err)
	}
}
