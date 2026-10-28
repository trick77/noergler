package review

import (
	"context"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/store"
)

func (f *fakeBitbucket) FetchPRChangeSet(_ context.Context, _, _ string, _ int) (bitbucket.ChangeSet, error) {
	return f.changeSet, f.changeSetErr
}

func (f *fakeBitbucket) FetchFileBytes(_ context.Context, _, _, commit, path string) ([]byte, error) {
	f.byteReads++
	key := commit + ":" + path
	if err, ok := f.fileErr[key]; ok {
		return nil, err
	}
	if b, ok := f.blobs[key]; ok {
		return b, nil
	}
	return nil, &bitbucket.StatusError{Method: "GET", Path: "/raw/" + path, Status: 404}
}

func (f *fakeLLM) Vision() bool { return f.vision }

func (f *fakeStore) VisualChanges(_ context.Context, _ store.PRKey) (map[store.VisualPair]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return nil, err
	}
	out := map[store.VisualPair]string{}
	for k, v := range f.visual {
		out[k] = v
	}
	return out, nil
}

func (f *fakeStore) InsertVisualChange(_ context.Context, _ store.PRKey, p store.VisualPair, text, _ string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return err
	}
	if f.visual == nil {
		f.visual = map[store.VisualPair]string{}
	}
	if _, ok := f.visual[p]; !ok {
		f.visual[p] = text
	}
	f.VisualInserts = append(f.VisualInserts, p)
	return nil
}
