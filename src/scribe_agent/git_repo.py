from __future__ import annotations

import logging
import re
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from git import Repo

logger = logging.getLogger(__name__)

_GITHUB_HTTPS = re.compile(
    r"https?://(?:[^@]+@)?github\.com/([^/]+)/([^/.]+)(?:\.git)?",
    re.I,
)
_GITHUB_SSH = re.compile(
    r"git@github\.com:([^/]+)/([^/.]+)(?:\.git)?\s*$",
    re.I,
)


@dataclass(frozen=True)
class GitSource:
    owner: str
    repo: str
    clone_url: str
    revision: str | None
    default_branch_hint: str | None


def git_source_from_clone_url(clone_url: str, branch: str) -> GitSource | None:
    """Build GitSource from configured clone URL and branch."""
    url = clone_url.strip()
    if not url:
        return None
    b = branch.strip()
    parsed = _owner_repo_from_clone_url(url)
    if not parsed:
        logger.warning("Could not parse owner/repo from SCRIBE_GIT_CLONE_URL: %s", url)
        return None
    owner, repo = parsed
    return GitSource(
        owner=owner,
        repo=repo,
        clone_url=url,
        revision=b or None,
        default_branch_hint=b or None,
    )


def _owner_repo_from_clone_url(clone_url: str) -> tuple[str, str] | None:
    u = clone_url.strip()
    m = _GITHUB_HTTPS.search(u)
    if m:
        return m.group(1), m.group(2)
    m = _GITHUB_SSH.match(u)
    if m:
        return m.group(1), m.group(2)
    if u.startswith("git@"):
        tail = u.split(":", 1)[-1]
        parts = tail.replace(".git", "").strip("/").split("/")
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        return None
    path = urlparse(u).path.strip("/")
    parts = [x for x in path.split("/") if x]
    if len(parts) >= 2:
        return parts[-2], parts[-1].removesuffix(".git")
    return None


def _authenticated_clone_url(clone_url: str, token: str | None) -> str:
    if not token:
        return clone_url
    if not clone_url.startswith("https://github.com/"):
        return clone_url
    rest = clone_url.removeprefix("https://")
    user = urllib.parse.quote("x-access-token", safe="")
    tok = urllib.parse.quote(token, safe="")
    return f"https://{user}:{tok}@{rest}"


def clone_repository(
    source: GitSource,
    dest: Path,
    token: str | None,
    depth: int,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Workspace not empty: {dest}")

    url = _authenticated_clone_url(source.clone_url, token)
    rev = (source.revision or "").strip()
    if rev:
        logger.info("Cloning %s into %s (branch/ref %r, shallow)", source.clone_url, dest, rev)
        try:
            Repo.clone_from(
                url,
                dest,
                depth=depth,
                branch=rev,
                single_branch=True,
            )
        except Exception as e:
            logger.warning(
                "Shallow clone with --branch %r failed (%s); falling back to default clone + checkout",
                rev,
                e,
            )
            repo = Repo.clone_from(url, dest, depth=depth)
            try:
                repo.git.fetch("origin", rev, depth=depth)
            except Exception as e2:
                logger.warning("Fetch %s failed: %s", rev, e2)
            try:
                repo.git.checkout(rev)
            except Exception:
                try:
                    repo.git.checkout("FETCH_HEAD")
                except Exception as e3:
                    logger.warning("Checkout %s failed: %s", rev, e3)
    else:
        logger.info("Cloning %s into %s (default branch)", source.clone_url, dest)
        Repo.clone_from(url, dest, depth=depth)

    return dest


def git_repo_summary(repo_path: Path, max_lines: int = 200) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_path), "log", "-n", "20", "--oneline"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        return f"(git log failed: {e})"
    lines = out.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)
