from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from llama_stack_client import LlamaStackClient

from scribe_agent.config import Settings
from scribe_agent.git_repo import GitSource, clone_repository, git_repo_summary, git_source_from_clone_url
from scribe_agent.llama_tools import run_tool_assisted_fix
from scribe_agent.mcp_github import create_issue_via_mcp, parse_json_loose
from scribe_agent.state_store import StateStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are **Scribe**, an agent that turns plain-text notes into well-scoped **GitHub issues**.

You receive the full text of a `.txt` file and metadata about the **target GitHub repository** (owner/repo and
branch). A **local git clone** of that repository on the configured branch is available under workspace tools.

Your job:
1. Read repository context when helpful: use `workspace_list_files`, `workspace_read_file` (e.g. README, docs),
   and **GitHub MCP tools** for anything else you need from GitHub.
2. Decompose the source text into **one or more** discrete issues. Each issue should be actionable and
   self-contained where possible.
3. Decide for **each** issue whether an **AI coding agent** (e.g. Developer agent) is expected to implement it.
   Set `"assign_to_agent": true` when the source text says that an AI agent, coding agent, **Developer** agent,
   automated agent, or similar must implement the change. Human-only or informational items should use
   `"assign_to_agent": false`.

**Do not** create GitHub issues yourself with MCP during this run; the service will create them from your plan.

When you are finished researching, respond with a **single JSON object** and **no tool calls** in that final
turn. The JSON must have this exact shape:
```json
{"issues":[{"title":"...","body":"...","assign_to_agent":true or false}]}
```

Rules for the JSON:
- `issues` is a non-empty array unless the source text truly contains no actionable items (then use `[]`).
- Every issue must have a non-empty `title` and a `body` (markdown allowed) that includes enough context from
  the source text.
- Use `assign_to_agent: true` whenever the source assigns implementation to an AI/Developer/automated agent.

Ignore unrelated tool groups (e.g. Kubernetes) unless the text explicitly requires them."""


def _register_mcp_endpoints(client: LlamaStackClient, settings: Settings) -> None:
    for reg in settings.parsed_mcp_registrations():
        try:
            client.toolgroups.register(
                toolgroup_id=reg.toolgroup_id,
                provider_id=reg.provider_id,
                mcp_endpoint={"uri": reg.mcp_uri},
            )
            logger.info("Registered MCP toolgroup %s", reg.toolgroup_id)
        except Exception as e:
            logger.warning(
                "Could not register MCP toolgroup %s (may already exist): %s",
                reg.toolgroup_id,
                e,
            )


def _resolve_model_id(client: LlamaStackClient, configured: str | None) -> str:
    if configured:
        return configured
    models = client.models.list()
    if not models:
        raise RuntimeError("LLAMA_STACK_MODEL_ID is unset and Llama Stack returned no models")
    mid = models[0].id
    logger.info("Using first available Llama Stack model: %s", mid)
    return mid


def _readme_excerpt(repo_path: Path, max_chars: int) -> str | None:
    if max_chars <= 0:
        return None
    for name in ("README.md", "README.rst", "README.txt", "README"):
        p = repo_path / name
        if p.is_file():
            t = p.read_text(encoding="utf-8", errors="replace")
            if len(t) > max_chars:
                return t[:max_chars] + "\n\n... (truncated)"
            return t
    return None


def _parse_issue_plan(model_text: str) -> list[dict[str, Any]]:
    parsed = parse_json_loose(model_text)
    if not isinstance(parsed, dict):
        return []
    issues = parsed.get("issues")
    if not isinstance(issues, list):
        return []
    out: list[dict[str, Any]] = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        body = str(item.get("body") or "").strip()
        raw_flag = item.get("assign_to_agent")
        if raw_flag is None:
            raw_flag = item.get("for_agent")
        if isinstance(raw_flag, bool):
            assign_to_agent = raw_flag
        else:
            assign_to_agent = str(raw_flag).lower() in ("true", "1", "yes")
        out.append(
            {
                "title": title,
                "body": body or "(no description)",
                "assign_to_agent": assign_to_agent,
            }
        )
    return out


def _discover_txt_files(watch: Path, recursive: bool) -> list[Path]:
    if not watch.is_dir():
        return []
    if recursive:
        return sorted({p.resolve() for p in watch.rglob("*.txt") if p.is_file()})
    return sorted(watch.glob("*.txt"))


def _build_user_prompt(
    source_path: Path,
    source_text: str,
    git_summary: str,
    repo_path: Path,
    branch: str,
    owner: str,
    repo: str,
    readme_excerpt: str | None,
) -> str:
    readme_block = (
        f"## README excerpt (if present)\n\n{readme_excerpt}\n"
        if readme_excerpt
        else "## README excerpt\n\n(not found or empty clone)\n"
    )
    return f"""## Source file
Path: `{source_path}`

## Target GitHub repository
- **owner:** `{owner}`
- **repo:** `{repo}`
- **branch (clone / context):** `{branch}`

## Local clone
Path: `{repo_path}`

Recent commits:
```
{git_summary}
```

{readme_block}

## Plain text to decompose into issues

```
{source_text}
```
"""


def process_text_file(
    settings: Settings,
    state: StateStore,
    client: LlamaStackClient,
    model_id: str,
    src: GitSource,
    txt_path: Path,
    content: str,
    content_sha256: str,
) -> None:
    ws = Path(settings.workspace_root) / f"scribe-{content_sha256[:24]}"
    if ws.exists():
        shutil.rmtree(ws)

    clone_ok = True
    try:
        clone_repository(src, ws, settings.github_token, settings.git_clone_depth)
    except Exception as e:
        clone_ok = False
        logger.warning(
            "Clone failed for %s/%s (%s). Continuing with empty workspace; use GitHub MCP for repo context.",
            src.owner,
            src.repo,
            e,
        )
        ws.mkdir(parents=True, exist_ok=True)

    summary = git_repo_summary(ws) if clone_ok else "(no clone)"
    readme = _readme_excerpt(ws, settings.readme_excerpt_chars) if clone_ok else None

    user_prompt = _build_user_prompt(
        txt_path.resolve(),
        content,
        summary,
        ws,
        settings.git_branch,
        src.owner,
        src.repo,
        readme,
    )

    logger.info("Invoking Llama Stack (model=%s) for %s", model_id, txt_path)
    try:
        model_reply = run_tool_assisted_fix(
            client=client,
            model_id=model_id,
            tool_group_ids=settings.tool_group_id_list,
            repo_root=ws,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_iterations=settings.max_llm_iterations,
        )
    except Exception:
        logger.exception("Llama Stack run failed for %s", txt_path)
        state.mark_content_processed(
            content_sha256,
            {
                "reason": "llm_failed",
                "source_path": str(txt_path.resolve()),
            },
        )
        return
    finally:
        if ws.exists():
            try:
                shutil.rmtree(ws)
            except OSError as e:
                logger.warning("Could not remove workspace %s: %s", ws, e)

    logger.info("Model reply (excerpt): %s", model_reply[:2000])
    planned = _parse_issue_plan(model_reply)
    if not planned:
        logger.warning("No issues parsed from model output for %s; marking processed to avoid a tight loop", txt_path)
        state.mark_content_processed(
            content_sha256,
            {
                "reason": "no_issues_parsed",
                "source_path": str(txt_path.resolve()),
                "model_excerpt": model_reply[:8000],
            },
        )
        return

    agent_label = settings.agent_implementation_label.strip() or "agent"
    created_urls: list[str] = []

    for item in planned:
        labels = [agent_label] if item["assign_to_agent"] else []
        title = item["title"][:250]
        body = (
            f"Created by **Scribe** from `{txt_path.name}`.\n\n"
            f"---\n\n{item['body']}"
        )
        if settings.dry_run_no_issues:
            logger.info(
                "SCRIBE_DRY_RUN_NO_ISSUES: would create issue %r labels=%s",
                title,
                labels,
            )
            created_urls.append(f"(dry-run) {title}")
            continue
        try:
            url = create_issue_via_mcp(
                client,
                settings,
                owner=src.owner,
                repo=src.repo,
                title=title,
                body=body,
                labels=labels,
            )
            created_urls.append(url)
            logger.info("Created issue: %s", url)
        except Exception:
            logger.exception("create_issue MCP failed for %r", title)
            state.mark_content_processed(
                content_sha256,
                {
                    "reason": "create_issue_failed",
                    "source_path": str(txt_path.resolve()),
                    "partial_created": created_urls,
                    "failed_title": title,
                },
            )
            return

    state.mark_content_processed(
        content_sha256,
        {
            "source_path": str(txt_path.resolve()),
            "issues_created": len(created_urls),
            "issue_urls": created_urls,
        },
    )


def run_forever(settings: Settings, state: StateStore) -> None:
    src = git_source_from_clone_url(settings.git_clone_url, settings.git_branch)
    if not src:
        raise RuntimeError(
            "Could not derive GitHub owner/repo from SCRIBE_GIT_CLONE_URL; check the URL format."
        )

    watch = Path(settings.watch_directory)
    if not watch.is_dir():
        raise RuntimeError(f"SCRIBE_WATCH_DIRECTORY is not a directory: {watch}")

    client = LlamaStackClient(
        base_url=settings.llama_stack_base_url,
        api_key=settings.llama_stack_api_key,
        timeout=600.0,
    )
    _register_mcp_endpoints(client, settings)
    model_id = _resolve_model_id(client, settings.llama_stack_model_id)

    while True:
        try:
            files = _discover_txt_files(watch, settings.watch_recursive)
            pending: list[tuple[Path, str, str]] = []
            for path in files:
                raw = path.read_bytes()
                h = hashlib.sha256(raw).hexdigest()
                if not state.is_content_processed(h):
                    text = raw.decode("utf-8", errors="replace")
                    if settings.max_source_text_chars > 0 and len(text) > settings.max_source_text_chars:
                        text = (
                            text[: settings.max_source_text_chars]
                            + "\n\n... (truncated; SCRIBE_MAX_SOURCE_TEXT_CHARS)"
                        )
                    pending.append((path, text, h))

            if pending:
                logger.info(
                    "Poll: %s .txt file(s) in %s, %s new by content hash",
                    len(files),
                    watch,
                    len(pending),
                )
            else:
                logger.info(
                    "Poll: no new .txt content under %s; sleeping %ss",
                    watch,
                    settings.poll_interval_seconds,
                )

            for path, text, h in pending:
                logger.info("Processing %s (sha256=%s...)", path, h[:12])
                process_text_file(settings, state, client, model_id, src, path, text, h)
        except Exception:
            logger.exception("Poll iteration failed")

        time.sleep(settings.poll_interval_seconds)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    settings = Settings()
    state = StateStore(settings.state_file_path)
    run_forever(settings, state)


if __name__ == "__main__":
    main()
