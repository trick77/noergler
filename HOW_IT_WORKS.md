# How noergler works

This document describes the review pipeline in detail: how diffs are fetched, processed, enriched with context, and sent to the AI model.

## Pipeline overview

```
Webhook event
  |
  v
Signature validation (HMAC-SHA256)
  |
  v
Event routing (pr:opened, pr:from_ref_updated, pr:comment:added, pr:merged)
  |
  v
Author check (optional allowlist)
  |
  v
Incremental review detection (for pr:from_ref_updated)
  |
  v
Diff fetch (full PR diff or incremental commit diff)
  |
  v
File splitting and filtering
  |
  v
Content enrichment (full file content fetch)
  |
  v
Context expansion (asymmetric + dynamic scope detection)
  |
  v
Cross-file context analysis (symbol reference mapping)
  |
  v
Token-aware chunking and compression
  |
  v
AI review (GitHub Models API)
  |
  v
Post-processing (dedup, sort, limit)
  |
  v
Post inline comments + summary
```

## 1. Webhook reception and routing

**File:** `app/main.py`

noergler exposes a `/webhook` endpoint that receives Bitbucket Server webhook events. Every request is validated via HMAC-SHA256 using the `X-Hub-Signature` header.

Events are routed as follows:

| Event | Handler |
|---|---|
| `pr:opened` | Full review |
| `pr:from_ref_updated` | Incremental or full review |
| `pr:merged` | Usefulness stats collection |
| `pr:comment:added` | Mention Q&A or feedback handling |
| `diagnostics:ping` | Health check acknowledgment |

Comment events are further routed based on content: if the comment contains the mention trigger (default `@noergler`), it goes to the Q&A handler; if it's a reply to an existing noergler comment, it goes to the feedback handler.

**Example:** A developer opens a PR and pushes two more commits. noergler receives three events: one `pr:opened` (full review) and two `pr:from_ref_updated` (incremental reviews of just the new commits).

## 2. Incremental review detection

**File:** `app/reviewer.py`

On `pr:from_ref_updated` events, noergler checks if a previous review exists by looking for a commit hash embedded as an HTML comment in the summary:

```
<!-- noergler:last_reviewed_commit=abc123def456 -->
```

If found, it requests an incremental diff from Bitbucket's compare API (`/compare/diff?from={lastCommit}&to={currentCommit}`) covering only changes since the last review. This avoids re-reviewing the entire PR when new commits are pushed.

**Fallback:** If the compare API fails (e.g., after a force-push that removed the old commit), noergler falls back to a full PR review.

**Empty diff:** If the incremental diff is empty (push with no code changes), the review is skipped entirely.

**Example:** A PR was reviewed at commit `a1b2c3d`. The developer pushes two more commits, landing at `e4f5g6h`. noergler finds `<!-- noergler:last_reviewed_commit=a1b2c3d -->` in the summary comment, calls `/compare/diff?from=a1b2c3d&to=e4f5g6h`, and only reviews the changes between those two commits.

## 3. Diff fetch and file splitting

**Files:** `app/bitbucket.py`, `app/llm_client.py`

The raw diff (either full PR diff or incremental) is split into per-file chunks using `diff --git` boundaries. Each file chunk is then filtered:

**Skipped by extension:** Binary files (`.png`, `.jar`, `.exe`, etc.), lock files (`.lock`), minified files (`.min.js`), data files (`.json`, `.csv`), and build config (`.xml`, `.properties`).

**Skipped by directory:** `node_modules`, `build`, `target`, `dist`, `__pycache__`, and dot-directories (`.git`, `.next`, etc.).

**Skipped by content:** Files with `binary files ... differ` markers.

**Example:** A PR touches `src/service.py`, `package-lock.json`, `dist/bundle.min.js`, and `logo.png`. Only `src/service.py` passes filtering — the lock file, minified bundle, and image are all skipped.

## 4. Content enrichment

**File:** `app/reviewer.py` (`_prepare_files`)

For each reviewable file, noergler fetches the **full file content** at the source commit from Bitbucket. This gives the AI complete context, not just the diff. The full content is included alongside the diff in the prompt.

**Size limit:** Files exceeding `REVIEW_MAX_FILE_LINES` (default 1000) are reviewed with diff only (no full content). This prevents a single huge file from consuming the entire token budget.

The AI sees each file presented as:

```
## File: src/service.py
### Full file content (new version):
```python
<entire file>
```
### Changes (diff):
```diff
<unified diff>
```
```

## 5. Context expansion

**File:** `app/context_expansion.py`

Diffs from Bitbucket arrive with zero context lines (only changed lines). noergler expands each hunk with surrounding context from the full file content:

### Asymmetric context

By default, 3 lines before and 2 lines after each hunk. The asymmetry reflects that code before a change (function signature, class declaration) is usually more informative than code after.

### Dynamic scope detection

For supported languages, noergler searches backwards from each hunk to find the enclosing function or class definition. If found, the context window is extended to include it.

Language-specific patterns:

| Language | Scope detection pattern |
|---|---|
| Python | `def`, `class`, `async def` |
| JVM (Java/Kotlin) | `public`, `private`, `fun`, `class`, `interface`, `enum`, `@` annotations, `{` |
| TypeScript/JS | `function`, `class`, `interface`, `enum`, `export`, `const ... = (` |
| Other | `def`, `func`, `fn`, `{` |

Config, docs, CSS, and HTML files skip dynamic scope detection (no meaningful enclosing scopes).

### Hunk merging

After expansion, overlapping or adjacent hunks are merged to avoid duplicated context lines.

**Example:** A diff hunk changes line 25 inside a method. The raw diff shows only line 25. After asymmetric expansion: lines 22-27. Dynamic scope detection finds `def process_order():` at line 18 and extends the window to lines 18-27. The AI now sees the full method signature alongside the change.

## 6. Cross-file context analysis

**File:** `app/cross_file_context.py`

When a PR modifies multiple files, changes in one file often affect others (e.g., a renamed function parameter breaks callers in other files). noergler builds a cross-file relationship map:

### How it works

1. **Symbol extraction:** For each changed file, added lines matching language-specific scope patterns (function/class definitions) are parsed to extract symbol names.

2. **Reference search:** Each extracted symbol is searched for (word-boundary match) in all other PR files' content.

3. **Relationship map:** Symbols with cross-file references are collected into a structured map.

### What gets sent to the model

The relationship map is rendered as a markdown section appended to the supplementary context:

```markdown
## Cross-file relationships

The following symbols were changed and are referenced in other files
in this PR. Pay special attention to whether callers/consumers are
updated consistently.

**`get_user`** (changed in `service.py`) is referenced in:
- `controller.py:45` -- user = get_user(request.user_id)
- `test_service.py:12` -- result = get_user(1)
```

This enables the AI to detect:
- Mismatched function signatures between definition and call sites
- Missing import updates after renames
- Broken call sites after parameter changes
- Incomplete refactors across files

**Limits:** Max 5 references per symbol, max 30 relationship lines total. Symbols shorter than 3 characters are ignored to avoid noise.

**Example:** A developer renames `get_user(id)` to `get_user(user_id, include_deleted=False)` in `service.py`. Cross-file analysis finds `controller.py:4` still calls `get_user(request.id)` with the old signature. The relationship map tells the AI about this, and it flags the mismatched call site.

## 7. Large PR compression

**File:** `app/diff_compression.py`

When a PR's total tokens (with context expansion) exceed the model's budget, noergler compresses:

### File classification

Each file is classified by:
- **Language:** Python, JVM, TypeScript, HTML, CSS, config, docs, build-config, other
- **Role:** Source file vs. test file (detected by path patterns like `_test.py`, `.spec.ts`, `/tests/`)

### Priority sorting

Files are sorted for inclusion priority:
1. Source files in the PR's primary language(s)
2. Test files in the same languages
3. Deprioritized files (config, docs, build-config, other)

Within each group, larger files (by token count) are prioritized — they likely contain more significant changes.

### Token budget

90% of available tokens (after prompt overhead) are allocated to files. Files are added in priority order until the budget is exhausted. Remaining files are listed as "other modified files" in the prompt so the AI knows the full PR scope.

### Deletion handling

- **Deleted files** are separated and listed by path only (no diff content).
- **Renamed files** (100% similarity) are listed by path only.
- **Deletion-only hunks** (hunks with only removed lines, no additions) are stripped from active files since there's nothing new to review.

**Example:** A 40-file PR exceeds the token budget. The PR is mostly Python with 2 config files. The 30 Python source files are included first (prioritized), then 8 Python test files. The 2 config files and any remaining files are listed under "Other modified files" so the AI knows they exist but doesn't review them in detail.

## 8. What gets sent to the AI model

**File:** `app/llm_client.py`

The final prompt sent to the model is assembled from the template (`prompts/review.txt`) with these placeholders filled:

| Placeholder | Content |
|---|---|
| `{files}` | Rendered file entries (full content + diff) + supplementary context |
| `{repo_instructions}` | Contents of `AGENTS.md` from the repo (if present) |
| `{tone}` | Tone preset (default: friendly senior engineer; ramsay: brutally condescending) |
| `{ticket_context}` | Jira ticket details (title, description, acceptance criteria) |
| `{compliance_instructions}` | Instructions for evaluating ticket compliance (when Jira is enabled) |

### Supplementary context (appended after file entries)

```markdown
## Other modified files (not included in detail)
- path/to/large_file.py
- path/to/config.yaml

## Renamed files (no content changes)
- old_name.py -> new_name.py

## Deleted files
- removed_feature.py

## Cross-file relationships
**`get_user`** (changed in `service.py`) is referenced in:
- `controller.py:45` -- user = get_user(request.user_id)
```

### Token-aware chunking

If files don't fit in a single API call, they're split into groups that fit within the token budget. Each group is reviewed independently. The supplementary context (other files list, cross-file relationships) is included in the first chunk only.

### 413 handling (payload too large)

If the API returns 413, noergler:
1. Reads the `Max size` from the response and adjusts the token budget
2. Splits the current group in half and retries each half
3. For single-file groups, retries without full file content (diff only)
4. After 3 levels of bisection, skips the file

## 9. AI response parsing

**File:** `app/llm_client.py`

The model responds with JSON containing:

```json
{
  "findings": [
    {
      "file": "service.py",
      "line": 42,
      "severity": "critical",
      "confidence": 95,
      "comment": "Description of the issue",
      "suggestion": "corrected code (optional)"
    }
  ],
  "change_summary": [
    "Added user deletion support with soft-delete flag",
    "Updated API endpoint to accept optional parameters"
  ],
  "compliance_requirements": [
    {"requirement": "User can be soft-deleted", "met": true}
  ]
}
```

Only findings with confidence >= 80 are included. Severity levels:
- **critical** (90-100): Runtime bugs, security vulnerabilities, data loss risks
- **warning** (80-89): Likely bugs, error handling gaps, performance issues

## 10. Post-processing and posting

**File:** `app/reviewer.py`

### Deduplication

Before posting, findings are checked against existing noergler comments on the PR (by file + line + severity). Duplicates are dropped. This prevents re-posting the same finding on subsequent reviews.

### Sorting and limiting

Findings are sorted by severity (critical first), then capped at `REVIEW_MAX_COMMENTS` (default 25).

### Inline comments

Each finding is posted as an inline comment at the specific file and line, with severity emoji and optional code suggestion.

### Summary comment

A summary comment is posted (or updated if one already exists) containing:
- Review type indicator (initial review vs. incremental update with commit range)
- Issue counts by severity
- Security issue flag
- Change summary (AI-generated bullet points)
- Jira ticket compliance status (if applicable)
- Review effort estimate (1-5 scale)
- Files reviewed count (e.g., "8 of 12 files" when compression skips files)
- Diff size (additions / deletions)
- Cross-file dependencies analyzed (symbol names)
- Token usage breakdown
- Model name and elapsed time
- Commit tracking metadata (invisible HTML comment)

**Example inline comment:**

```
🔴 **Critical** — `service.py:42`

`user_id` can be `None` when called from the batch endpoint, causing an
unhandled `TypeError` on the database query.

**Suggestion:**
```python
if not user_id:
    raise ValueError("user_id is required")
```
```

**Example summary comment (excerpt):**

```
### Review summary
- 1 critical ❌
- 2 warnings ⚠️

### What changed
- Added user deletion endpoint with soft-delete flag
- Updated UserService to accept optional `include_deleted` parameter

### Info
- Estimated review effort: 3/5 — Medium: multiple files, some logic changes 📊
- Reviewed 8 of 12 files (4 skipped: lock files, binaries, config) 📂
- Diff: +142 / -38 lines
- 2 cross-file dependencies analyzed (`get_user`, `UserCache`) 🔗
- Model: `gpt-4o` · 12'450↑ 890↓ (13'340 total) · ⏱️ 8.2s
<!-- noergler:last_reviewed_commit=e4f5g6h789 -->
```

## 11. Mention Q&A

**Files:** `app/reviewer.py`, `app/llm_client.py`

When a developer mentions `@noergler` with a question in a PR comment, the Q&A pipeline:

1. Fetches the full PR diff and file contents (same as review)
2. Sends the question + files to the model with a Q&A-specific prompt
3. Posts the answer as a reply to the original comment

Special keywords (`review`, `re-review`, `rereview`) trigger a full review instead.

**Example:** A developer comments `@noergler Why does this endpoint return 404 for deleted users?` on a PR. noergler fetches the PR diff, sees the soft-delete logic, and replies explaining that the endpoint filters out soft-deleted users by default and suggests using the `include_deleted` query parameter.

## 12. Feedback collection

**File:** `app/reviewer.py`

When a developer replies to a noergler inline comment:
- If the reply contains "disagree" (or similar negative feedback), it's logged as a disagreement
- An emoji reaction is added to acknowledge the feedback
- Aggregate stats (disagreement rate) are tracked per PR

On PR merge, final usefulness stats are calculated: percentage of comments that weren't disagreed with.

## 13. Jira integration

**File:** `app/jira.py`

When a PR branch name or title contains a Jira ticket ID (e.g., `PROJ-123`):

1. The ticket is fetched from Jira (title, description, acceptance criteria, subtasks)
2. Parent-child relationships are resolved for subtasks
3. Ticket context is included in the review prompt
4. The AI evaluates compliance against acceptance criteria
5. Compliance status (fully/partially/not compliant) is shown in the summary

Acceptance criteria are extracted from ticket descriptions using configurable prefixes (AC, AK, DoD, Req, etc.).

**Example:** Branch `feature/PROJ-123-user-deletion` triggers a Jira lookup. Ticket PROJ-123 has acceptance criteria: "AC1: Users can be soft-deleted", "AC2: Deleted users are excluded from search results". The AI checks both against the PR changes and reports: AC1 met, AC2 not met (search query not updated).
