package review

import (
	"context"

	"github.com/trick77/noergler/internal/almanach"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
)

// --- fake LLM: the note call -------------------------------------------------

func (f *fakeLLM) FitCeiling() int { return f.window - inference.OutputTokenReserve }

func (f *fakeLLM) AlmanachNote(_ context.Context, req inference.AlmanachNoteRequest) inference.AlmanachNoteResult {
	f.Notes = append(f.Notes, req)
	return f.note
}

// --- fake store: the notes table ---------------------------------------------

// fakeNote is one almanach_notes row as the fake keeps it.
type fakeNote struct {
	store.AlmanachNote
	final    *store.AlmanachFinal
	emitted  bool
	issues   int
	suggests int
}

func (f *fakeStore) UpsertAlmanachNote(_ context.Context, n store.AlmanachNote) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return "", err
	}
	if f.noteErr != nil {
		return "", f.noteErr
	}
	if f.notes == nil {
		f.notes = map[store.PRKey]*fakeNote{}
	}
	row, ok := f.notes[n.Key]
	switch {
	case ok && row.emitted:
		return store.NoteAlreadyEmitted, nil
	case ok && row.RunID != 0 && row.RunID >= n.RunID:
		return store.NoteSuperseded, nil
	case ok:
		row.AlmanachNote = n
	default:
		f.notes[n.Key] = &fakeNote{AlmanachNote: n}
	}
	f.Notes = append(f.Notes, n)
	return store.NoteWritten, nil
}

func (f *fakeStore) FinalizeAlmanachNote(_ context.Context, k store.PRKey, team string, final store.AlmanachFinal) (bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return false, err
	}
	if f.notes == nil {
		f.notes = map[store.PRKey]*fakeNote{}
	}
	row, ok := f.notes[k]
	if !ok {
		row = &fakeNote{AlmanachNote: store.AlmanachNote{Key: k, TeamSlug: team, PRTitle: final.PRTitle}}
		f.notes[k] = row
	}
	if row.final != nil {
		return false, nil
	}
	fin := final
	row.final = &fin
	for _, fi := range f.Findings {
		if fi.PullRequestID == f.prID {
			if fi.Severity == "issue" {
				row.issues++
			} else {
				row.suggests++
			}
		}
	}
	f.Finalized = append(f.Finalized, final)
	return true, nil
}

func (f *fakeStore) DeleteAlmanachNote(_ context.Context, k store.PRKey) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.NoteDeletes++
	if row, ok := f.notes[k]; ok && row.final == nil {
		delete(f.notes, k)
	}
	return f.err()
}

func (f *fakeStore) MarkAlmanachEmitted(_ context.Context, k store.PRKey, runID int64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if row, ok := f.notes[k]; ok && row.RunID == runID && !row.emitted {
		row.emitted = true
		f.EmittedRuns = append(f.EmittedRuns, runID)
	}
	return f.err()
}

func (f *fakeStore) PendingAlmanachNotes(_ context.Context, team string, limit int) ([]store.AlmanachPending, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.err(); err != nil {
		return nil, err
	}
	var out []store.AlmanachPending
	// Oldest first: the order the rows were finalized, which the fake
	// approximates by PR id.
	for id := 0; id < 1000 && len(out) < limit; id++ {
		for k, row := range f.notes {
			if k.PRID != id || row.final == nil || row.emitted || row.Note == nil || row.TeamSlug != team {
				continue
			}
			p := store.AlmanachPending{AlmanachNote: row.AlmanachNote, MergedAt: row.final.MergedAt,
				MergeCommit: row.final.MergeCommit, SourceCommit: row.final.SourceCommit,
				IssuesPosted: row.issues, SuggestionsPosted: row.suggests}
			for _, c := range f.AlmanachCalls {
				if c.Key == k {
					p.Rounds++
					if c.RunID != nil && *c.RunID == row.RunID {
						p.NoteCallFound, p.NoteCallOutcome, p.NoteStatus = true, c.Outcome, c.NoteStatus
						if c.Call != nil {
							p.NoteCall = *c.Call
						}
					}
				}
			}
			out = append(out, p)
		}
	}
	return out, nil
}

func (f *fakeStore) InsertAlmanachCall(_ context.Context, c store.AlmanachCall) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.AlmanachCalls = append(f.AlmanachCalls, c)
	return f.err()
}

func (f *fakeStore) AlmanachNoteStatus(_ context.Context, k store.PRKey) (store.AlmanachNoteState, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	row, ok := f.notes[k]
	if !ok {
		return store.AlmanachNoteState{}, false, f.err()
	}
	return store.AlmanachNoteState{Finalized: row.final != nil, HasNote: row.Note != nil, Emitted: row.emitted, PROpen: true}, true, f.err()
}

// --- fake Almanach emitter ---------------------------------------------------

// fakeAlmanach records the notes sent; refuse makes every post a miss.
type fakeAlmanach struct {
	enabled bool
	refuse  bool
	Emitted []almanach.Note
}

func (f *fakeAlmanach) Enabled() bool { return f.enabled }
func (f *fakeAlmanach) EmitNote(_ context.Context, n almanach.Note) bool {
	f.Emitted = append(f.Emitted, n)
	return !f.refuse
}
