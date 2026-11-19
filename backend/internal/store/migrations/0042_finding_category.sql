-- What kind of defect a finding is, as the review named it: bug, security,
-- secret, guideline or contradiction (the four remit items of review.txt,
-- security split out). The summary's security count read keywords from the
-- comment before; a row from before this column (NULL) still does.
ALTER TABLE findings ADD COLUMN category TEXT;
