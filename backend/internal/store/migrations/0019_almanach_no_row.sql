-- A round's note call can find its row gone: the PR was declined or deleted
-- while the call ran unheld, and DeleteAlmanachNote took the row between the
-- upsert and the status read. That is neither "superseded" (no later round
-- exists) nor "already_emitted"; the ledger says so.
ALTER TABLE almanach_calls DROP CONSTRAINT almanach_calls_note_status_check;
ALTER TABLE almanach_calls ADD CONSTRAINT almanach_calls_note_status_check
    CHECK (note_status IN
        ('written', 'superseded', 'already_emitted', 'no_row', 'nothing_learned',
         'dropped_too_long', 'dropped_secret', 'dropped_invalid', 'no_answer'));
