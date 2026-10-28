package diff

import (
	"fmt"
	"path"
	"slices"
	"strings"
)

// screenshotDirs are the directory names visual test runners commit their
// baselines under: Playwright's `<spec>-snapshots`, Vitest browser mode's
// `__screenshots__`, jest-image-snapshot's `__image_snapshots__`. Matched per
// path segment, lowercased.
var screenshotDirs = []string{"__screenshots__", "__image_snapshots__"}

// screenshotDirSuffix is Playwright's per-spec baseline directory.
const screenshotDirSuffix = "-snapshots"

// screenshotExts are the image types a screenshot is compared in.
var screenshotExts = []string{".png", ".jpg", ".jpeg"}

// variantPlatforms and variantBrowsers are the suffixes Playwright appends to
// a baseline's stem (`login-chromium-linux.png`). Stripped to group one
// screenshot's variants: they show the same change, and each costs as much
// to send as the first.
var (
	variantPlatforms = []string{"-linux", "-darwin", "-win32"}
	variantBrowsers  = []string{"-chromium", "-chrome", "-msedge", "-firefox", "-webkit"}
)

// ScreenshotChange says what the PR did to a screenshot.
type ScreenshotChange int

// The changes a PR can make to a screenshot.
const (
	ScreenshotModified ScreenshotChange = iota
	ScreenshotAdded
	ScreenshotDeleted
)

// Screenshot is one committed test screenshot the PR changed. OldPath is the
// before side's path, the same as Path unless the file was renamed.
type Screenshot struct {
	Path    string
	OldPath string
	Change  ScreenshotChange
	// BeforeID and AfterID are the versions' git blob ids when the source
	// knows them (a PR's changes do, a diff's header does not), else "".
	BeforeID, AfterID string
}

// IsScreenshot reports whether path is a committed visual test baseline:
// an image under one of the runners' snapshot directories. Any other image
// (a logo, an icon) is not one.
func IsScreenshot(p string) bool {
	p = strings.ToLower(p)
	if !hasAnySuffix(p, screenshotExts) {
		return false
	}
	segments := strings.Split(p, "/")
	for _, seg := range segments[:len(segments)-1] {
		if strings.HasSuffix(seg, screenshotDirSuffix) || slices.Contains(screenshotDirs, seg) {
			return true
		}
	}
	return false
}

// Screenshots lists the screenshots a whole-PR diff changes, in path order.
// Added and deleted are read off the header markers git and Bitbucket write
// for binary files; anything else is modified.
func Screenshots(prDiff string) []Screenshot {
	var out []Screenshot
	for _, fd := range SplitByFile(prDiff) {
		p := ExtractPath(fd)
		if p == "" || !IsScreenshot(p) {
			continue
		}
		s := Screenshot{Path: p, OldPath: p}
		head := firstRunes(fd, binaryMarkerScanRunes)
		switch {
		case strings.Contains(head, "\nnew file mode") || strings.Contains(head, "\n--- /dev/null") ||
			strings.Contains(head, "Binary files /dev/null and"):
			s.Change = ScreenshotAdded
		case strings.Contains(head, "\ndeleted file mode") || IsDeleted(fd) ||
			strings.Contains(head, " and /dev/null differ"):
			s.Change = ScreenshotDeleted
		}
		for _, line := range strings.Split(head, "\n") {
			if from, ok := strings.CutPrefix(line, "rename from "); ok {
				s.OldPath = strings.TrimRight(from, "\r")
			}
		}
		out = append(out, s)
	}
	slices.SortFunc(out, func(a, b Screenshot) int { return strings.Compare(a.Path, b.Path) })
	return out
}

// GroupVariants groups one screenshot's browser and platform variants, in
// path order of their first member, each group chromium first, then by
// path. The change is part of the key: a deleted variant beside a modified
// one is not the same change. A caller shows one per group and tries the
// next when the first turns out unchanged, so a change only one browser
// shows is not lost.
func GroupVariants(shots []Screenshot) [][]Screenshot {
	groups := map[string][]Screenshot{}
	var order []string
	for _, s := range shots {
		k := fmt.Sprintf("%d:%s", s.Change, variantGroup(s.Path))
		if _, seen := groups[k]; !seen {
			order = append(order, k)
		}
		groups[k] = append(groups[k], s)
	}
	chromium := func(s Screenshot) bool { return strings.Contains(strings.ToLower(path.Base(s.Path)), "-chromium") }
	out := make([][]Screenshot, 0, len(order))
	for _, k := range order {
		g := groups[k]
		slices.SortStableFunc(g, func(a, b Screenshot) int {
			switch {
			case chromium(a) && !chromium(b):
				return -1
			case chromium(b) && !chromium(a):
				return 1
			}
			return strings.Compare(a.Path, b.Path)
		})
		out = append(out, g)
	}
	return out
}

// variantGroup is the path with its extension and the trailing platform and
// browser suffixes stripped, lowercased.
func variantGroup(p string) string {
	p = strings.ToLower(p)
	stem := strings.TrimSuffix(p, path.Ext(p))
	for _, list := range [][]string{variantPlatforms, variantBrowsers} {
		for _, suffix := range list {
			if s, ok := strings.CutSuffix(stem, suffix); ok {
				stem = s
				break
			}
		}
	}
	return stem
}
