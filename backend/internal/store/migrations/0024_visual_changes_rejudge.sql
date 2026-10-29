-- The model now marks a screenshot pair it sees no difference in, and such a
-- pair gets no line (user, 2026-10-30: «kein klarer visueller Unterschied»
-- lines are noise). Lines stored before carry no such mark and would keep
-- showing; dropping them sends those pairs once more, to be judged anew.
-- change_text '' now also marks a pair the model saw no difference in.
DELETE FROM visual_changes WHERE change_text <> '';
