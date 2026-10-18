package render

import (
	"fmt"

	"github.com/trick77/noergler/internal/lang"
)

// The skip summaries in the PR author's language. The English texts are in
// skip.go; the German ones mirror them section for section. Env var and
// teams.yaml names stay as they are: they are what an operator types.

// OptOutBranchSummary is the skip summary for a branch carrying the opt-out
// keyword. No LLM or Jira call was made.
func OptOutBranchSummary(l lang.Lang, keyword, branch string) string {
	if l == lang.EN {
		return optOutBranchSummaryEN(keyword, branch)
	}
	return "### Review übersprungen — Opt-out-Keyword im Branch-Namen 🛑\n\n" +
		fmt.Sprintf("Der Source-Branch `%s` enthält das Opt-out-Keyword ", branch) +
		fmt.Sprintf("`%s`, deshalb wurde das Auto-Review übersprungen. Für diesen PR ", keyword) +
		"wurden keine LLM- oder Jira-Aufrufe gemacht.\n\n" +
		"**Was tun**\n" +
		"- Den Branch ohne das Keyword umbenennen und erneut pushen; das " +
		"nächste Webhook-Event löst ein volles Review aus.\n" +
		"- Oder den Bot in einem Kommentar explizit @erwähnen, um auf diesem " +
		"Branch einmalig ein Review zu bekommen — Mentions sind nicht stummgeschaltet.\n\n" +
		"**Konfiguration**\n" +
		"- Das Keyword wird über `REVIEW_OPT_OUT_BRANCH_KEYWORD` am " +
		"noergler-Service gesetzt (leerer String schaltet die Funktion ab).\n"
}

// AgentsMDMissingSummary is the skip summary when AGENTS.md is required and
// absent from both the PR branch and the target branch.
func AgentsMDMissingSummary(l lang.Lang) string {
	if l == lang.EN {
		return agentsMDMissingSummaryEN()
	}
	return "### Review übersprungen — keine `AGENTS.md` gefunden 🛑\n\n" +
		"Dieses Repository hat weder auf dem PR-Branch noch auf dem Ziel-Branch eine `AGENTS.md`. " +
		"Projektspezifische Review-Richtlinien sind **entscheidend** für gezieltes Feedback " +
		"mit hohem Nutzen — ohne sie bleibt nur generisches Kleinzeug, " +
		"deshalb wurde das Review nicht ausgeführt.\n\n" +
		"**Was tun**\n" +
		"- Eine `AGENTS.md` im Repo-Root anlegen, die Projektkonventionen, " +
		"verbotene Muster und Schwerpunkte für das Review beschreibt.\n" +
		"- Die Datei auf den PR-Branch pushen (oder in den Ziel-Branch mergen); " +
		"das nächste Webhook-Event auf diesem PR löst ein volles Review aus.\n\n" +
		"**Opt-out**\n" +
		"- Um PRs ohne `AGENTS.md` zu prüfen, `require_agents_md: false` " +
		"für dieses Team in der noergler-`teams.yaml` setzen (oder `REVIEW_REQUIRE_AGENTS_MD=false` " +
		"als Instanz-Default) und neu starten.\n"
}

// DiffTooLargeSummary is the skip summary when the PR diff exceeds the byte
// cap. limit is in bytes.
func DiffTooLargeSummary(l lang.Lang, limit int) string {
	if l == lang.EN {
		return diffTooLargeSummaryEN(limit)
	}
	return "### Review übersprungen — Diff zu gross 🛑\n\n" +
		fmt.Sprintf("Der PR-Diff ist grösser als %d MiB, mehr als in ein ", limit/(1024*1024)) +
		"Review passt, auch nach Komprimierung, deshalb wurde das Review nicht ausgeführt.\n\n" +
		"**Was tun**\n" +
		"- Die Änderung in kleinere PRs aufteilen, oder generierte Dateien, " +
		"vendored Code und Lockfile-Änderungen in einen eigenen PR verschieben.\n" +
		"- Die kleinere Änderung pushen; das nächste Webhook-Event auf diesem PR " +
		"löst ein volles Review aus.\n"
}

func agentsMDTooLargeSummaryDE(tokens, limit int, fr string) string {
	return "### Review übersprungen — `AGENTS.md` zu gross 🛑\n\n" +
		fmt.Sprintf("`AGENTS.md` umfasst ~%s Tokens und überschreitet das konfigurierte ", Fmt(tokens)) +
		fmt.Sprintf("harte Limit von %s Tokens. Zu grosse Agent-Anweisungen verdrängen ", Fmt(limit)) +
		"den eigentlichen Diff, verschlechtern die Review-Qualität (Context Rot) und treiben " +
		"die Inferenzkosten hoch — deshalb wurde das Review nicht ausgeführt.\n\n" +
		"**Was tun**\n" +
		"- `AGENTS.md` kürzen: Abschnitte zum Überfliegen für Menschen streichen, " +
		"README-Duplikate entfernen, Prosa durch knappe Regeln ersetzen.\n" +
		"- Ausführliches Referenzmaterial in separate Dateien auslagern und " +
		"verlinken; die Hauptdatei schlank halten.\n" +
		"- Die schlankere Datei pushen; das nächste Webhook-Event auf diesem PR " +
		"löst ein volles Review aus.\n\n" +
		"**Konfiguration**\n" +
		"- Das Limit über `REVIEW_AGENTS_MD_MAX_TOKENS` am noergler-Service " +
		"erhöhen, oder auf `0` setzen, um die harte Grenze ganz abzuschalten " +
		"(die weiche Warnung über `REVIEW_AGENTS_MD_WARN_TOKENS` bleibt).\n\n" +
		"**Weiterführende Links**\n" +
		fr + "\n"
}
