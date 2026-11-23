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
		"**Einstellungen**\n" +
		"- Das Keyword wird über `REVIEW_OPT_OUT_BRANCH_KEYWORD` am " +
		"noergler-Service gesetzt (leerer String schaltet die Funktion ab).\n"
}

// AgentsMDMissingSummary is the skip summary when the team's rules file
// (name: AGENTS.md unless the team chose another) is required and absent
// from both the PR branch and the target branch. A call to action: what
// belongs in it, never what does not (user, 2026-11-23).
func AgentsMDMissingSummary(l lang.Lang, name string) string {
	if l == lang.EN {
		return agentsMDMissingSummaryEN(name)
	}
	return fmt.Sprintf("### Review übersprungen — keine `%s` gefunden 🛑\n\n", name) +
		fmt.Sprintf("Dieses Repository hat weder auf dem PR-Branch noch auf dem Ziel-Branch eine `%s`, ", name) +
		"deshalb wurde das Review nicht ausgeführt.\n\n" +
		"**Was tun**\n" +
		fmt.Sprintf("- Eine `%s` im Repo-Root anlegen, mit dem, was im Code nicht steht: ", name) +
		"Invarianten, die nie brechen dürfen, Konventionen eures Teams, Entscheide und ihr Warum, " +
		"Fallen, aus denen ihr gelernt habt. Genau daran misst das Review euren Code; " +
		"ohne sie sieht es nur den Diff.\n" +
		"- Ein paar Zeilen genügen für den Anfang. Regeln, die einen Merge blockieren sollen, " +
		fmt.Sprintf("mit `[blocking]` markieren. Im Monorepo ergänzt eine `%s` im Unterordner ", name) +
		"Regeln für diesen Teil.\n" +
		fmt.Sprintf("- Keine Projektregeln gewünscht? Eine leere `%s` genügt; das Review ", name) +
		"läuft dann mit allgemeinen Regeln.\n" +
		"- Die Datei pushen; das nächste Webhook-Event auf diesem PR löst ein volles Review aus.\n"
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

func agentsMDTooLargeSummaryDE(name string, tokens, limit int, fr string) string {
	return fmt.Sprintf("### Review übersprungen — `%s` zu gross 🛑\n\n", name) +
		fmt.Sprintf("`%s` umfasst ~%s Tokens und überschreitet das konfigurierte ", name, Fmt(tokens)) +
		fmt.Sprintf("harte Limit von %s Tokens. Zu grosse Agent-Anweisungen verdrängen ", Fmt(limit)) +
		"den eigentlichen Diff, verschlechtern die Review-Qualität (Context Rot) und treiben " +
		"die Inferenzkosten hoch — deshalb wurde das Review nicht ausgeführt.\n\n" +
		"**Was tun**\n" +
		"- Behalten, was der Code nicht zeigt: Invarianten, Konventionen, Entscheide und ihr Warum. " +
		"Streichen, was Code oder README schon sagen; Prosa durch knappe Regeln ersetzen.\n" +
		"- Ausführliches Referenzmaterial in separate Dateien auslagern und " +
		"verlinken; die Hauptdatei schlank halten.\n" +
		"- Die schlankere Datei pushen; das nächste Webhook-Event auf diesem PR " +
		"löst ein volles Review aus.\n\n" +
		"**Einstellungen**\n" +
		"- Das Limit über `REVIEW_AGENTS_MD_MAX_TOKENS` am noergler-Service " +
		"erhöhen, oder auf `0` setzen, um die harte Grenze ganz abzuschalten " +
		"(die weiche Warnung über `REVIEW_AGENTS_MD_WARN_TOKENS` bleibt).\n\n" +
		"**Weiterführende Links**\n" +
		fr + "\n"
}
