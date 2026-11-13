// Package gitrepo is noergler's one use of git itself: fetch a PR's two
// branches into memory, find their merge base, write a merge commit and
// push it. Smart HTTP through go-git's session API, never its Remote: that
// one takes its transport from a process-wide registry, and this needs the
// Bitbucket adapter's client (no redirects, no client timeout) with a byte
// count of its own per fetch.
//
// Nothing touches disk: the image has no writable directory, and a
// repository's objects stay in memory, bounded by Remote.MaxBytes.
package gitrepo

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"
	"github.com/go-git/go-git/v5/plumbing/format/packfile"
	"github.com/go-git/go-git/v5/plumbing/object"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/capability"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/sideband"
	"github.com/go-git/go-git/v5/plumbing/transport"
	githttp "github.com/go-git/go-git/v5/plumbing/transport/http"
	"github.com/go-git/go-git/v5/storage/memory"
)

// Errors a caller maps to an answer.
var (
	// ErrTooLarge is a fetch past Remote.MaxBytes.
	ErrTooLarge = errors.New("fetch exceeds the size limit")
	// ErrIncomplete is a merge base deeper than the history fetched.
	ErrIncomplete = errors.New("history too shallow for the merge base")
	// ErrNoBase is two branches without one merge base: none, or several
	// (a criss-cross merge, which git resolves recursively and this does
	// not).
	ErrNoBase = errors.New("no single merge base")
	// ErrMoved is a branch that is no longer at the commit the merge was
	// built on.
	ErrMoved = errors.New("branch moved")
	// ErrNoWrite is a push the account may not make (401/403).
	ErrNoWrite = errors.New("no write permission")
)

// Rejected is a push the server refused for its own reason (a hook).
type Rejected struct{ Status string }

func (e *Rejected) Error() string { return "push rejected: " + e.Status }

// Remote is one repository over smart HTTP as one account.
type Remote struct {
	// URL is the clone URL, without credentials.
	URL string
	// Username and Token authenticate as HTTP basic: Bitbucket Data
	// Center takes an HTTP access token as the password for git.
	Username, Token string
	// HTTP is the client every request goes through. Its redirect policy
	// stands; go-git's own is set to never follow.
	HTTP *http.Client
	// MaxBytes bounds what one fetch reads, 0 for no bound.
	MaxBytes int64
	// MaxInflated bounds what one fetch holds in memory, the objects
	// inflated, 0 for no bound.
	MaxInflated int64
}

func (r Remote) endpoint() (*transport.Endpoint, error) {
	u, err := url.Parse(r.URL)
	if err != nil {
		return nil, err
	}
	u.User = nil
	return transport.NewEndpoint(u.String())
}

func (r Remote) auth() *githttp.BasicAuth {
	return &githttp.BasicAuth{Username: r.Username, Password: r.Token}
}

// client is r's HTTP client with a byte count of its own: every response
// body read past max fails with ErrTooLarge.
func (r Remote) client(read *atomic.Int64) transport.Transport {
	base := r.HTTP
	if base == nil {
		base = &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	}
	rt := base.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}
	hc := &http.Client{
		Transport:     counting{rt: rt, read: read, max: r.MaxBytes},
		CheckRedirect: base.CheckRedirect,
		Jar:           base.Jar,
	}
	return githttp.NewClientWithOptions(hc, &githttp.ClientOptions{RedirectPolicy: githttp.NoFollowRedirects})
}

type counting struct {
	rt   http.RoundTripper
	read *atomic.Int64
	max  int64
}

func (c counting) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := c.rt.RoundTrip(req)
	if err != nil || resp.Body == nil {
		return resp, err
	}
	resp.Body = &countingBody{ReadCloser: resp.Body, c: c}
	return resp, nil
}

type countingBody struct {
	io.ReadCloser
	c counting
}

func (b *countingBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if total := b.c.read.Add(int64(n)); b.c.max > 0 && total > b.c.max {
		return n, ErrTooLarge
	}
	return n, err
}

// capped is memory storage that refuses objects past max inflated bytes:
// the wire count bounds the compressed pack, which inflates many times
// over in memory.
type capped struct {
	*memory.Storage
	stored, max int64
}

func (c *capped) SetEncodedObject(o plumbing.EncodedObject) (plumbing.Hash, error) {
	if c.max > 0 {
		if c.stored += o.Size(); c.stored > c.max {
			return plumbing.ZeroHash, ErrTooLarge
		}
	}
	return c.Storage.SetEncodedObject(o)
}

// Repo is fetched objects in memory.
type Repo struct {
	st *capped
	// trees are the tree objects Files walked: on the remote already, so a
	// commit rebuilding one sends nothing for it, even once detached.
	trees map[plumbing.Hash]bool
	// Read is what the fetch read off the wire.
	Read int64
}

// Detach is r without its fetched objects: what a merge commit needs once
// the files are read (their entries and the trees Files walked), so the
// fetch's memory is not held while the model is asked.
func (r *Repo) Detach() *Repo {
	return &Repo{st: &capped{Storage: memory.NewStorage()}, trees: r.trees, Read: r.Read}
}

// Fetch reads branches (names without refs/heads/) at most depth commits
// deep into memory and returns their heads, in order. A branch the remote
// does not have is an error.
func Fetch(ctx context.Context, r Remote, depth int, branches ...string) (*Repo, []plumbing.Hash, error) {
	ep, err := r.endpoint()
	if err != nil {
		return nil, nil, err
	}
	var read atomic.Int64
	sess, err := r.client(&read).NewUploadPackSession(ep, r.auth())
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = sess.Close() }()
	adv, err := sess.AdvertisedReferencesContext(ctx)
	if err != nil {
		return nil, nil, wrapRead(err)
	}
	refs, err := adv.AllReferences()
	if err != nil {
		return nil, nil, err
	}
	heads := make([]plumbing.Hash, len(branches))
	req := packp.NewUploadPackRequestFromCapabilities(adv.Capabilities)
	seen := map[plumbing.Hash]bool{}
	for i, b := range branches {
		ref, err := refs.Reference(plumbing.NewBranchReferenceName(b))
		if err != nil {
			return nil, nil, fmt.Errorf("branch %s: %w", b, err)
		}
		heads[i] = ref.Hash()
		if !seen[ref.Hash()] {
			seen[ref.Hash()] = true
			req.Wants = append(req.Wants, ref.Hash())
		}
	}
	if depth > 0 {
		req.Depth = packp.DepthCommits(depth)
		if err := req.Capabilities.Set(capability.Shallow); err != nil {
			return nil, nil, err
		}
	}
	if adv.Capabilities.Supports(capability.NoProgress) {
		_ = req.Capabilities.Set(capability.NoProgress)
	}
	resp, err := sess.UploadPack(ctx, req)
	if err != nil {
		return nil, nil, wrapRead(err)
	}
	defer func() { _ = resp.Close() }()
	st := &capped{Storage: memory.NewStorage(), max: r.MaxInflated}
	if len(resp.Shallows) > 0 {
		if err := st.SetShallow(resp.Shallows); err != nil {
			return nil, nil, err
		}
	}
	if err := packfile.UpdateObjectStorage(st, sidebandReader(req.Capabilities, resp)); err != nil {
		return nil, nil, wrapRead(err)
	}
	return &Repo{st: st, Read: read.Load()}, heads, nil
}

func wrapRead(err error) error {
	if errors.Is(err, ErrTooLarge) {
		return ErrTooLarge
	}
	return err
}

func sidebandReader(l *capability.List, r io.Reader) io.Reader {
	var t sideband.Type
	switch {
	case l.Supports(capability.Sideband64k):
		t = sideband.Sideband64k
	case l.Supports(capability.Sideband):
		t = sideband.Sideband
	default:
		return r
	}
	return sideband.NewDemuxer(t, r)
}

// MergeBase is the one best common ancestor of a and b, by git's
// paint-down: commits newest first, painted from either side, until only
// commits below a common one are left. A parent missing on a path that
// still counts (the shallow edge) is ErrIncomplete: fetch deeper.
func (r *Repo) MergeBase(a, b plumbing.Hash) (plumbing.Hash, error) {
	if a == b {
		return a, nil
	}
	const (
		p1 = 1 << iota
		p2
		stale
		result
	)
	flags := map[plumbing.Hash]int{}
	commits := map[plumbing.Hash]*object.Commit{}
	q := &commitQueue{}
	push := func(h plumbing.Hash, f int) error {
		c, ok := commits[h]
		if !ok {
			var err error
			c, err = object.GetCommit(r.st, h)
			if err != nil {
				return err
			}
			commits[h] = c
		}
		flags[h] |= f
		heap.Push(q, c)
		return nil
	}
	if err := push(a, p1); err != nil {
		return plumbing.ZeroHash, err
	}
	if err := push(b, p2); err != nil {
		return plumbing.ZeroHash, err
	}
	var results []plumbing.Hash
	// edge records a side's walk reaching the shallow edge: a parent
	// missing on a path only that side reaches.
	edge := map[int]bool{}
	for q.hasNonStale(flags, stale) {
		c := heap.Pop(q).(*object.Commit)
		f := flags[c.Hash] & (p1 | p2 | stale)
		if f == p1|p2 {
			if flags[c.Hash]&result == 0 {
				flags[c.Hash] |= result
				results = append(results, c.Hash)
			}
			f |= stale
		}
		for _, p := range c.ParentHashes {
			if flags[p]&f == f {
				continue
			}
			if err := push(p, f); err != nil {
				if !errors.Is(err, plumbing.ErrObjectNotFound) {
					return plumbing.ZeroHash, err
				}
				// Below a common commit nothing better hides.
				if f&stale == 0 {
					edge[f] = true
				}
			}
		}
	}
	// A commit hidden past one side's edge is no ancestor of the other
	// side when that side's walk ended without an edge: it would have been
	// reached and fetched. Only both edges, or no base found, can hide one.
	if (edge[p1] && edge[p2]) || (len(edge) > 0 && len(results) == 0) {
		return plumbing.ZeroHash, ErrIncomplete
	}
	if len(results) != 1 {
		return plumbing.ZeroHash, ErrNoBase
	}
	return results[0], nil
}

// commitQueue is a max-heap on committer time.
type commitQueue []*object.Commit

func (q commitQueue) Len() int { return len(q) }
func (q commitQueue) Less(i, j int) bool {
	return q[i].Committer.When.After(q[j].Committer.When)
}
func (q commitQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *commitQueue) Push(x any)   { *q = append(*q, x.(*object.Commit)) }
func (q *commitQueue) Pop() any {
	old := *q
	c := old[len(old)-1]
	*q = old[:len(old)-1]
	return c
}

func (q commitQueue) hasNonStale(flags map[plumbing.Hash]int, stale int) bool {
	for _, c := range q {
		if flags[c.Hash]&stale == 0 {
			return true
		}
	}
	return false
}

// Entry is one path of a tree: its object and mode.
type Entry struct {
	Hash plumbing.Hash
	Mode filemode.FileMode
}

// Files is the commit's tree, flattened: every path to its entry. A
// submodule is an entry too (its hash a commit of another repository),
// never walked.
func (r *Repo) Files(commit plumbing.Hash) (map[string]Entry, error) {
	c, err := object.GetCommit(r.st, commit)
	if err != nil {
		return nil, err
	}
	out := map[string]Entry{}
	if r.trees == nil {
		r.trees = map[plumbing.Hash]bool{}
	}
	var walk func(h plumbing.Hash, dir string) error
	walk = func(h plumbing.Hash, dir string) error {
		t, err := object.GetTree(r.st, h)
		if err != nil {
			return err
		}
		r.trees[h] = true
		for _, e := range t.Entries {
			p := path.Join(dir, e.Name)
			if e.Mode == filemode.Dir {
				if err := walk(e.Hash, p); err != nil {
					return err
				}
				continue
			}
			out[p] = Entry{Hash: e.Hash, Mode: e.Mode}
		}
		return nil
	}
	return out, walk(c.TreeHash, "")
}

// Blob is a blob's bytes, an error past max bytes.
func (r *Repo) Blob(h plumbing.Hash, max int64) ([]byte, error) {
	b, err := object.GetBlob(r.st, h)
	if err != nil {
		return nil, err
	}
	if max > 0 && b.Size > max {
		return nil, ErrTooLarge
	}
	rd, err := b.Reader()
	if err != nil {
		return nil, err
	}
	defer func() { _ = rd.Close() }()
	return io.ReadAll(rd)
}

// Signature is a commit's author and committer.
type Signature struct{ Name, Email string }

// Commit writes the tree files and a commit of it on parents, with the
// blobs of content (path to bytes) as files' new objects. It returns the
// commit and the objects the remote does not have yet: what a push sends.
func (r *Repo) Commit(files map[string]Entry, content map[string][]byte, parents []plumbing.Hash, sig Signature, message string, when time.Time) (plumbing.Hash, []plumbing.Hash, error) {
	var created []plumbing.Hash
	store := func(o plumbing.EncodedObject) (plumbing.Hash, error) {
		h := o.Hash()
		if r.st.HasEncodedObject(h) == nil || (o.Type() == plumbing.TreeObject && r.trees[h]) {
			return h, nil
		}
		if _, err := r.st.SetEncodedObject(o); err != nil {
			return plumbing.ZeroHash, err
		}
		created = append(created, h)
		return h, nil
	}
	files = maps.Clone(files)
	for p, b := range content {
		o := r.st.NewEncodedObject()
		o.SetType(plumbing.BlobObject)
		o.SetSize(int64(len(b)))
		w, err := o.Writer()
		if err != nil {
			return plumbing.ZeroHash, nil, err
		}
		if _, err := w.Write(b); err != nil {
			return plumbing.ZeroHash, nil, err
		}
		if err := w.Close(); err != nil {
			return plumbing.ZeroHash, nil, err
		}
		h, err := store(o)
		if err != nil {
			return plumbing.ZeroHash, nil, err
		}
		e, ok := files[p]
		if !ok {
			return plumbing.ZeroHash, nil, fmt.Errorf("content for %s, which the tree does not have", p)
		}
		files[p] = Entry{Hash: h, Mode: e.Mode}
	}
	root, err := r.writeTree(files, store)
	if err != nil {
		return plumbing.ZeroHash, nil, err
	}
	s := object.Signature{Name: sig.Name, Email: sig.Email, When: when}
	c := &object.Commit{Author: s, Committer: s, Message: message, TreeHash: root, ParentHashes: parents}
	o := r.st.NewEncodedObject()
	if err := c.Encode(o); err != nil {
		return plumbing.ZeroHash, nil, err
	}
	h, err := store(o)
	if err != nil {
		return plumbing.ZeroHash, nil, err
	}
	return h, created, nil
}

// writeTree builds the nested trees of a flat path map, bottom up, and
// returns the root's hash.
func (r *Repo) writeTree(files map[string]Entry, store func(plumbing.EncodedObject) (plumbing.Hash, error)) (plumbing.Hash, error) {
	type dir struct {
		entries map[string]object.TreeEntry
		subdirs map[string]bool
	}
	dirs := map[string]*dir{"": {entries: map[string]object.TreeEntry{}, subdirs: map[string]bool{}}}
	var ensure func(p string) *dir
	ensure = func(p string) *dir {
		if d, ok := dirs[p]; ok {
			return d
		}
		d := &dir{entries: map[string]object.TreeEntry{}, subdirs: map[string]bool{}}
		dirs[p] = d
		parent := path.Dir(p)
		if parent == "." {
			parent = ""
		}
		ensure(parent).subdirs[path.Base(p)] = true
		return d
	}
	for p, e := range files {
		parent := path.Dir(p)
		if parent == "." {
			parent = ""
		}
		ensure(parent).entries[path.Base(p)] = object.TreeEntry{Name: path.Base(p), Mode: e.Mode, Hash: e.Hash}
	}
	var build func(p string) (plumbing.Hash, error)
	build = func(p string) (plumbing.Hash, error) {
		d := dirs[p]
		entries := make([]object.TreeEntry, 0, len(d.entries)+len(d.subdirs))
		for _, e := range d.entries {
			entries = append(entries, e)
		}
		for name := range d.subdirs {
			h, err := build(path.Join(p, name))
			if err != nil {
				return plumbing.ZeroHash, err
			}
			entries = append(entries, object.TreeEntry{Name: name, Mode: filemode.Dir, Hash: h})
		}
		// git's order: a directory sorts as if its name ended in "/".
		key := func(e object.TreeEntry) string {
			if e.Mode == filemode.Dir {
				return e.Name + "/"
			}
			return e.Name
		}
		sort.Slice(entries, func(i, j int) bool { return key(entries[i]) < key(entries[j]) })
		t := &object.Tree{Entries: entries}
		o := r.st.NewEncodedObject()
		if err := t.Encode(o); err != nil {
			return plumbing.ZeroHash, err
		}
		return store(o)
	}
	return build("")
}

// Push moves branch from old to commit, sending objects. Never a force:
// a branch no longer at old is ErrMoved, here or in the server's own
// compare-and-swap.
func Push(ctx context.Context, r Remote, repo *Repo, branch string, old, commit plumbing.Hash, objects []plumbing.Hash) error {
	ep, err := r.endpoint()
	if err != nil {
		return err
	}
	var read atomic.Int64
	sess, err := r.client(&read).NewReceivePackSession(ep, r.auth())
	if err != nil {
		return err
	}
	defer func() { _ = sess.Close() }()
	adv, err := sess.AdvertisedReferencesContext(ctx)
	if err != nil {
		return wrapWrite(err)
	}
	refs, err := adv.AllReferences()
	if err != nil {
		return err
	}
	name := plumbing.NewBranchReferenceName(branch)
	if cur, err := refs.Reference(name); err != nil || cur.Hash() != old {
		return ErrMoved
	}
	req := packp.NewReferenceUpdateRequestFromCapabilities(adv.Capabilities)
	req.Commands = []*packp.Command{{Name: name, Old: old, New: commit}}
	rd, wr := io.Pipe()
	req.Packfile = rd
	done := make(chan error, 1)
	go func() {
		_, err := packfile.NewEncoder(wr, repo.st, false).Encode(objects, 10)
		if err != nil {
			done <- wr.CloseWithError(err)
			return
		}
		done <- wr.Close()
	}()
	// A refused command comes back as the report and its error both: the
	// report says why.
	rs, err := sess.ReceivePack(ctx, req)
	if rs == nil {
		if err == nil {
			// A server without report-status accepted the pack.
			return <-done
		}
		// Closing the pipe ends the encoder, whose error (the closed pipe)
		// lands in the buffered done unread: the push's is the one.
		_ = rd.Close()
		return wrapWrite(err)
	}
	if err := <-done; err != nil {
		return err
	}
	for _, cs := range rs.CommandStatuses {
		if cs.Status == "ok" {
			continue
		}
		// git's own words for a ref that moved under the push; a hook's
		// "branch is locked" is a refusal, never "ask again".
		s := strings.ToLower(cs.Status)
		if strings.Contains(s, "fetch first") || strings.Contains(s, "stale info") ||
			strings.Contains(s, "non-fast-forward") || strings.Contains(s, "failed to lock") ||
			strings.Contains(s, "cannot lock ref") {
			return ErrMoved
		}
		return &Rejected{Status: cs.Status}
	}
	if rs.UnpackStatus != "ok" {
		return &Rejected{Status: "unpack " + rs.UnpackStatus}
	}
	return nil
}

func wrapWrite(err error) error {
	if errors.Is(err, transport.ErrAuthenticationRequired) || errors.Is(err, transport.ErrAuthorizationFailed) {
		return fmt.Errorf("%w: %w", ErrNoWrite, err)
	}
	return err
}
