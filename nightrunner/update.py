"""Update check against the GitHub repository a build came from.

Qt-free so it can be unit-tested and used from the CLI. The GUI wraps it in a corner widget on the tab bar.

Three pieces:

* `local_commit()` reads the checked-out commit straight out of `.git` (no `git` executable needed), falling back
  to a `build_info.json` written beside the package for installs that came from an archive rather than a clone.
* `remote_head()` / `compare()` are thin wrappers over the GitHub REST API, stdlib `urllib` only.
* `check()` combines them into a `Status` the UI can render without knowing any of the above.

The repository is configurable (`NIGHTRUNNER_UPDATE_REPO`) so a fork does not have to patch the source. A token is
optional: without one the anonymous API is used, which is enough for a public repository. `GITHUB_TOKEN` / `GH_TOKEN`
or the GitHub CLI's stored auth let the check work against a private repository during development. A token is only
ever put in an Authorization header; it is never logged or written into a Status.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO = "ErickCHIN000/nightrunner"
DEFAULT_BRANCH = "main"
API = "https://api.github.com"
USER_AGENT = "nightrunner-update-check"
TIMEOUT = 10.0

ROOT = Path(__file__).resolve().parents[1]

#: state -> what the UI should say. "unknown" means we could not determine the local commit at all.
STATES = ("unknown", "up-to-date", "behind", "ahead", "local-only", "diverged", "error")


@dataclass
class Status:
    """The result of one check. Never carries a token or any other credential."""
    state: str = "unknown"
    local: str | None = None            # full sha of the checked-out commit
    remote: str | None = None           # full sha of the branch head upstream
    behind: int = 0                     # commits on the remote that we do not have
    ahead: int = 0                      # local commits the remote does not have
    branch: str = DEFAULT_BRANCH
    repo: str = DEFAULT_REPO
    error: str | None = None
    dirty: bool = False                 # a git clone we could not classify (e.g. detached, unknown ref)

    @property
    def short(self) -> str:
        return (self.local or "unknown")[:7]

    @property
    def compare_url(self) -> str:
        if self.local and self.remote and self.local != self.remote:
            return f"https://github.com/{self.repo}/compare/{self.local}...{self.remote}"
        return f"https://github.com/{self.repo}/commits/{self.branch}"

    @property
    def label(self) -> str:
        """One short string for the corner widget. Terse on purpose."""
        if self.state == "behind":
            return f"{self.short} · {self.behind} behind"
        if self.state == "ahead":
            return f"{self.short} · {self.ahead} ahead"
        if self.state == "local-only":
            return f"{self.short} · unpushed"
        if self.state == "diverged":
            return f"{self.short} · diverged"
        if self.state == "up-to-date":
            return f"{self.short} · latest"
        if self.state == "error":
            return f"{self.short} · offline"
        return self.short


# ---- local side ----------------------------------------------------------------------------------------------

def _read_packed_ref(git_dir: Path, ref: str) -> str | None:
    packed = git_dir / "packed-refs"
    if not packed.is_file():
        return None
    for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "^")):
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref:
            return sha
    return None


def local_commit(root: Path | None = None) -> str | None:
    """The commit this working tree is on, or None when it cannot be determined.

    Reads `.git` directly rather than shelling out, so it works without git installed and without a subprocess on
    every GUI start. `.git` may be a file (a worktree or submodule), in which case it points at the real dir.
    """
    root = Path(root or ROOT)
    git = root / ".git"
    if git.is_file():                                   # "gitdir: <path>" (worktree / submodule)
        try:
            target = git.read_text(encoding="utf-8").partition("gitdir:")[2].strip()
        except OSError:
            return None
        if not target:
            return None
        git = Path(target)
        if not git.is_absolute():
            git = (root / git).resolve()
    if not git.is_dir():
        return _build_info_commit(root)
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return _build_info_commit(root)
    if not head.startswith("ref:"):
        return head or None                             # detached HEAD holds the sha itself
    ref = head[4:].strip()
    loose = git / ref
    try:
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    return _read_packed_ref(git, ref)


def _build_info_commit(root: Path) -> str | None:
    """Archive installs have no .git; a build_info.json beside the package carries the commit instead."""
    p = root / "build_info.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("commit") or None
    except (OSError, ValueError):
        return None


def local_branch(root: Path | None = None) -> str | None:
    """The checked-out branch name, or None when detached / unavailable."""
    git = Path(root or ROOT) / ".git"
    if not git.is_dir():
        return None
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if head.startswith("ref: refs/heads/"):
        return head[len("ref: refs/heads/"):].strip() or None
    return None


# ---- remote side ---------------------------------------------------------------------------------------------

def default_token() -> str | None:
    """A token for the API, if one is available. Env first, then the GitHub CLI's stored auth.

    Only needed for a private repository. Returns None rather than raising when nothing is configured.
    """
    for var in ("NIGHTRUNNER_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return tok.strip()
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    tok = (out.stdout or "").strip()
    return tok if out.returncode == 0 and tok else None


def _get(url: str, token: str | None, timeout: float, opener) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    })
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with opener(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def remote_head(repo: str = DEFAULT_REPO, branch: str = DEFAULT_BRANCH, token: str | None = None,
                timeout: float = TIMEOUT, opener=urllib.request.urlopen) -> str:
    """The sha at the tip of *branch*. Raises urllib errors through to the caller."""
    return _get(f"{API}/repos/{repo}/commits/{branch}", token, timeout, opener)["sha"]


def compare(repo: str, base: str, head: str, token: str | None = None, timeout: float = TIMEOUT,
            opener=urllib.request.urlopen) -> dict:
    """GitHub's comparison of two commits: status, ahead_by, behind_by (all relative to *base*)."""
    return _get(f"{API}/repos/{repo}/compare/{base}...{head}", token, timeout, opener)


# ---- the one call the UI makes -------------------------------------------------------------------------------

def check(repo: str | None = None, branch: str | None = None, token: str | None = None, root: Path | None = None,
          timeout: float = TIMEOUT, opener=urllib.request.urlopen) -> Status:
    """Compare the local commit with the upstream branch head. Never raises; failures land in `Status.error`.

    Runs on a worker thread in the GUI: two HTTP round trips, or one when the local commit is already the head.
    """
    repo = repo or os.environ.get("NIGHTRUNNER_UPDATE_REPO") or DEFAULT_REPO
    branch = branch or os.environ.get("NIGHTRUNNER_UPDATE_BRANCH") or DEFAULT_BRANCH
    st = Status(repo=repo, branch=branch, local=local_commit(root))
    try:
        st.remote = remote_head(repo, branch, token, timeout, opener)
    except Exception as exc:                            # offline, rate-limited, private without a token, 404…
        st.state = "error"
        st.error = _describe(exc)
        return st
    if not st.local:
        st.state = "unknown"
        st.error = "local commit unknown (not a git checkout and no build_info.json)"
        return st
    if st.local == st.remote:
        st.state = "up-to-date"
        return st
    try:
        cmp_ = compare(repo, st.local, st.remote, token, timeout, opener)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # remote_head just succeeded, so the repo is reachable: it is our own commit the remote has never
            # seen. Normal while local work is unpushed, and not an error worth showing as "offline".
            st.state = "local-only"
            st.error = "local commit is not on the remote (unpushed?)"
            return st
        st.state = "error"
        st.error = _describe(exc)
        return st
    except Exception as exc:
        st.state = "error"
        st.error = _describe(exc)
        return st
    # GitHub reports ahead_by/behind_by from base (our commit) to head (the remote tip): what the remote has that
    # we do not is ahead_by, what we have that it does not is behind_by. Flipped here to the user's point of view.
    st.behind = int(cmp_.get("ahead_by") or 0)
    st.ahead = int(cmp_.get("behind_by") or 0)
    status = cmp_.get("status")
    if status == "ahead":
        st.state = "behind"
    elif status == "behind":
        st.state = "ahead"
    elif status == "identical":
        st.state = "up-to-date"
    else:
        st.state = "diverged"
    return st


# ---- changelog -----------------------------------------------------------------------------------------------

#: commit subject prefix -> heading. Conventional-commit prefixes first, then a few plain-English verbs, so a
#: repository that does not use the convention still groups sensibly instead of dumping everything in "OTHER".
_FIXED = ("fix", "fixes", "fixed", "bugfix", "hotfix", "patch", "revert")
_IMPROVED = ("feat", "feature", "add", "added", "improve", "improved", "perf", "refactor", "change", "changed",
             "update", "updated", "support")

MAX_LISTED = 12          # how many subjects the dialog shows before "+ N more changes included"


def _subject(message: str) -> str:
    return (message or "").strip().splitlines()[0].strip() if (message or "").strip() else ""


def categorise(subject: str) -> str:
    """FIXED / IMPROVED / OTHER for one commit subject. Cheap prefix match, never raises."""
    s = subject.strip().lower()
    head = s.split(":", 1)[0] if ":" in s[:24] else s
    word = head.split("(", 1)[0].strip().strip("!").split()[0] if head.split() else ""
    if word in _FIXED:
        return "FIXED"
    if word in _IMPROVED:
        return "IMPROVED"
    return "OTHER"


def changelog(repo: str, base: str, head: str, token: str | None = None, timeout: float = TIMEOUT,
              opener=urllib.request.urlopen, limit: int = MAX_LISTED) -> dict:
    """Commit subjects between *base* and *head*, grouped for the update dialog.

    Returns {"groups": {heading: [subject, ...]}, "listed": n, "total": n, "more": n}. `total` is the true number
    of commits in the range; the compare API returns at most 250 of them, so `more` can exceed what we counted and
    is taken from the caller's `behind` when that is larger.
    """
    data = compare(repo, base, head, token, timeout, opener)
    commits = data.get("commits") or []
    total = int(data.get("total_commits") or len(commits))
    subjects = []
    for c in reversed(commits):                          # newest first, the way the dialog reads
        s = _subject((c.get("commit") or {}).get("message", ""))
        if s and s not in subjects:
            subjects.append(s)
    groups: dict[str, list[str]] = {}
    for s in subjects[:limit]:
        groups.setdefault(categorise(s), []).append(s)
    ordered = {k: groups[k] for k in ("FIXED", "IMPROVED", "OTHER") if k in groups}
    listed = sum(len(v) for v in ordered.values())
    return {"groups": ordered, "listed": listed, "total": total, "more": max(0, total - listed)}


# ---- applying ------------------------------------------------------------------------------------------------

class UpdateError(RuntimeError):
    """The update could not be applied. The message names the exact reason."""


def _git(root: Path, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def can_apply(root: Path | None = None) -> tuple[bool, str]:
    """Whether `apply_update` could run here, and why not when it could not.

    Refuses rather than risking someone's work: no git, not a clone, a dirty tree, or a detached HEAD all stop it.
    """
    root = Path(root or ROOT)
    if not (root / ".git").exists():
        return False, "not a git checkout - download the new version manually"
    try:
        r = _git(root, "rev-parse", "--is-inside-work-tree", timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False, "git is not installed"
    if r.returncode != 0:
        return False, "not a git checkout - download the new version manually"
    if local_branch(root) is None:
        return False, "HEAD is detached - check out a branch first"
    try:
        r = _git(root, "status", "--porcelain", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False, "git is not installed"
    if r.returncode != 0:
        return False, (r.stderr or "git status failed").strip()
    if r.stdout.strip():
        n = len([l for l in r.stdout.splitlines() if l.strip()])
        return False, f"{n} uncommitted change(s) - commit or stash them first"
    return True, ""


def apply_update(root: Path | None = None, branch: str | None = None) -> str:
    """Fast-forward the checkout to the upstream branch head. Returns the new commit sha.

    Deliberately `--ff-only`: an update must never merge, rebase or discard anything. Anything that cannot be a
    fast-forward raises `UpdateError` naming the reason and leaves the tree exactly as it was.
    """
    root = Path(root or ROOT)
    ok, why = can_apply(root)
    if not ok:
        raise UpdateError(why)
    branch = branch or local_branch(root) or DEFAULT_BRANCH
    try:
        r = _git(root, "fetch", "origin", branch, timeout=300)
        if r.returncode != 0:
            raise UpdateError((r.stderr or "git fetch failed").strip().splitlines()[-1])
        r = _git(root, "merge", "--ff-only", "FETCH_HEAD", timeout=120)
        if r.returncode != 0:
            raise UpdateError((r.stderr or "git merge failed").strip().splitlines()[-1])
    except subprocess.SubprocessError as exc:
        raise UpdateError(f"git failed: {type(exc).__name__}") from exc
    return local_commit(root) or ""


def _describe(exc: Exception) -> str:
    """A short, credential-free description of a failed request."""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return "repository or branch not found (private without a token?)"
        if exc.code in (401, 403):
            return f"access denied or rate limited (HTTP {exc.code})"
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return "no connection"
    return type(exc).__name__
