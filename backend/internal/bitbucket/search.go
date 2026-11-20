package bitbucket

import (
	"context"
	"html"
	"net/http"
	"strings"
)

// searchPath is Bitbucket Data Center's code search, a resource of its own
// beside the REST API proper.
const searchPath = "/rest/search/latest/search"

// CodeHit is one file the search found, with the lines it returned around
// the matches.
type CodeHit struct {
	Project, Repo string
	File          string
	Lines         []CodeLine
}

// CodeLine is one line of a hit's context, as plain text: the search sends it
// HTML-escaped and wrapped in highlight markup, both taken off here.
type CodeLine struct {
	Line int
	Text string
}

// SearchCode runs one code search and returns up to limit files. The search
// indexes each repository's default branch only, files under 512 KiB, and
// what the token may read (Atlassian: "Search for code in Bitbucket Data
// Center"). One request, never paged: a caller that needs more asks a
// narrower query.
func (c *Client) SearchCode(ctx context.Context, query string, limit int) ([]CodeHit, error) {
	body := map[string]any{
		"query":    query,
		"entities": map[string]any{"code": map[string]any{"start": 0, "limit": limit}},
	}
	var resp struct {
		Code struct {
			Values []struct {
				Repository struct {
					Slug    string `json:"slug"`
					Project struct {
						Key string `json:"key"`
					} `json:"project"`
				} `json:"repository"`
				File        string `json:"file"`
				HitContexts [][]struct {
					Line int    `json:"line"`
					Text string `json:"text"`
				} `json:"hitContexts"`
			} `json:"values"`
		} `json:"code"`
	}
	// The search resource sits beside apiBase: a path of its own, sent
	// through the same client so it carries the team's token and observer.
	if err := c.do(ctx, http.MethodPost, searchPath, nil, body, &resp); err != nil {
		return nil, err
	}
	out := make([]CodeHit, 0, len(resp.Code.Values))
	for _, v := range resp.Code.Values {
		hit := CodeHit{Project: v.Repository.Project.Key, Repo: v.Repository.Slug, File: v.File}
		for _, ctxLines := range v.HitContexts {
			for _, l := range ctxLines {
				hit.Lines = append(hit.Lines, CodeLine{Line: l.Line, Text: plainHitText(l.Text)})
			}
		}
		out = append(out, hit)
	}
	return out, nil
}

// highlightTags are the markup the search wraps matches in. They come in
// empty pairs as well ("globalFunc.<em></em>printLog"), so they are cut out
// rather than trusted as the match.
var highlightTags = strings.NewReplacer("<em>", "", "</em>", "")

// plainHitText is a hit line as it stands in the file.
func plainHitText(s string) string {
	return html.UnescapeString(highlightTags.Replace(s))
}
