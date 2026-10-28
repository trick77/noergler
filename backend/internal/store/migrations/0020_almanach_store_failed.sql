-- A round whose note the store refused (a database fault on the upsert) is
-- neither "no_row" (the PR was closed) nor anything the model did: the
-- ledger says store_failed, and the warn log has the error.
ALTER TABLE almanach_calls DROP CONSTRAINT almanach_calls_note_status_check;
ALTER TABLE almanach_calls ADD CONSTRAINT almanach_calls_note_status_check
    CHECK (note_status IN
        ('written', 'superseded', 'already_emitted', 'no_row', 'store_failed', 'nothing_learned',
         'dropped_too_long', 'dropped_secret', 'dropped_invalid', 'no_answer'));
