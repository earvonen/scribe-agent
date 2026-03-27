# Scribe (scribe-agent)

**Scribe** is a Python service that **watches a directory for plain text files** (`.txt`). When new file **content** appears (by SHA-256 of the file bytes), it:

1. Optionally **clones** the configured GitHub repository (branch from **`SCRIBE_GIT_BRANCH`**) so the model can read **README** and docs via workspace tools.
2. Runs **Llama Stack** with your registered model (e.g. **Minimax** via `LLAMA_STACK_MODEL_ID`) and **GitHub MCP** tools—**no direct `api.github.com` calls** from this app for listing or creating issues; creation uses Llama Stack **`tool_runtime.invoke_tool`** like the previous Developer agent did for PRs.
3. Parses the model’s JSON plan and **creates one or more GitHub issues** on the configured repo.

If the source text says an **AI agent**, **coding agent**, or **Developer** agent must implement the work, Scribe adds the GitHub label **`agent`** (configurable via **`SCRIBE_AGENT_IMPLEMENTATION_LABEL`**).

## Flow

1. Every **`SCRIBE_POLL_INTERVAL_SECONDS`**, scans **`SCRIBE_WATCH_DIRECTORY`** for `*.txt` (optional **`SCRIBE_WATCH_RECURSIVE=true`**).
2. Skips content whose SHA-256 is already in **`SCRIBE_STATE_FILE`** under `processed_txt_sha256`.
3. Shallow **git clone** of **`SCRIBE_GIT_CLONE_URL`** at **`SCRIBE_GIT_BRANCH`** (`GITHUB_TOKEN` optional for private HTTPS clone)—**git is only for local README/docs context**, not for GitHub issue APIs. (This is a direct Git connection to GitHub, separate from Llama Stack; if clone fails, Scribe continues with an empty workspace and the model can rely on **GitHub MCP** for repository context.)
4. Runs the **Llama Stack** chat loop with MCP + workspace tools; the model returns a JSON object `{"issues":[...]}` with `assign_to_agent` flags.
5. For each issue, calls MCP tool **`SCRIBE_MCP_CREATE_ISSUE_TOOL`** (default `issue_write`) with `labels: ["agent"]` when `assign_to_agent` is true.

## Prerequisites

- **Llama Stack** at **`LLAMA_STACK_BASE_URL`**, with GitHub MCP registered under **`SCRIBE_TOOL_GROUP_IDS`**. Use **`SCRIBE_MCP_REGISTRATIONS_JSON`** if the stack needs MCP SSE registration at startup.
- A model id (e.g. Minimax) in **`LLAMA_STACK_MODEL_ID`** if the stack does not expose a single default model.

## MCP tool names and kwargs

Defaults assume common GitHub MCP parameters for `issue_write`: `owner`, `repo`, `title`, `body`, and optionally `labels` as a **list of strings**. If your server differs, set **`SCRIBE_MCP_CREATE_ISSUE_EXTRA_JSON`** to merge extra kwargs or override names.

## Run locally

```bash
cd scribe-agent
pip install -e .

export SCRIBE_WATCH_DIRECTORY=/path/to/inbox
export SCRIBE_GIT_CLONE_URL=https://github.com/org/application.git
export SCRIBE_GIT_BRANCH=main
export LLAMA_STACK_BASE_URL=http://localhost:8321
export LLAMA_STACK_MODEL_ID=minimax-m2  # example; use your stack’s model id
export SCRIBE_TOOL_GROUP_IDS=mcp-github
export GITHUB_TOKEN=ghp_...   # optional: private git clone

scribe-agent
# or: python -m scribe_agent
```

## Container image

```bash
podman build -f Containerfile -t scribe-agent:latest .
```

## Configuration (environment / ConfigMap)

| Variable | Required | Default | Meaning |
|----------|----------|---------|---------|
| `SCRIBE_WATCH_DIRECTORY` | **Yes** | — | Directory to scan for `.txt` files. |
| `SCRIBE_GIT_CLONE_URL` | **Yes** | — | Clone URL; used to resolve `owner/repo` and for local clone. |
| `SCRIBE_GIT_BRANCH` | **Yes** | — | Branch to check out for README/docs context. |
| `LLAMA_STACK_BASE_URL` | **Yes** | — | Llama Stack HTTP base URL. |
| `SCRIBE_TOOL_GROUP_IDS` | **Yes** | — | Comma-separated tool group IDs (include GitHub MCP). |
| `SCRIBE_WATCH_RECURSIVE` | No | `false` | If `true`, scan subdirectories for `*.txt`. |
| `SCRIBE_MCP_CREATE_ISSUE_TOOL` | No | `issue_write` | MCP tool name for creating issues. |
| `SCRIBE_MCP_CREATE_ISSUE_EXTRA_JSON` | No | — | JSON object merged into create-issue kwargs. |
| `SCRIBE_MCP_INVOKE_TOOL_GROUP_ID` | No | — | Tool **group** id that serves GitHub MCP (same as one entry in `SCRIBE_TOOL_GROUP_IDS`). If unset, Scribe picks the first group that lists `SCRIBE_MCP_CREATE_ISSUE_TOOL` (default `issue_write`). Required by Llama Stack when `invoke_tool` must not conflate tool names with group ids. |
| `SCRIBE_AGENT_IMPLEMENTATION_LABEL` | No | `agent` | Label when the text assigns work to an AI/Developer agent. |
| `SCRIBE_POLL_INTERVAL_SECONDS` | No | `60` | Sleep between scans. |
| `SCRIBE_STATE_FILE` | No | `/tmp/scribe-agent-state.json` | JSON state file. |
| `SCRIBE_WORKSPACE_ROOT` | No | `/tmp/scribe-workspaces` | Clone parent directory. |
| `SCRIBE_GIT_CLONE_DEPTH` | No | `50` | Shallow clone depth. |
| `SCRIBE_MAX_LLM_ITERATIONS` | No | `40` | Max tool loop rounds. |
| `SCRIBE_MAX_SOURCE_TEXT_CHARS` | No | `120000` | Truncate very large `.txt` inputs. |
| `SCRIBE_README_EXCERPT_CHARS` | No | `24000` | Max README chars injected into the prompt. |
| `SCRIBE_DRY_RUN_NO_ISSUES` | No | `false` | Log planned issues without calling MCP create. |
| `SCRIBE_MCP_REGISTRATIONS_JSON` | No | — | Optional MCP SSE registrations at startup. |
| `GITHUB_TOKEN` | No | — | **Git HTTPS only** (clone); not used for GitHub REST. |
| `LLAMA_STACK_API_KEY` | No | — | Optional Llama Stack API key. |
| `LLAMA_STACK_MODEL_ID` | No | — | Optional model id (e.g. Minimax). |

## State file

Processed items are keyed by **SHA-256 of file contents**, so renaming a file does not re-trigger processing; changing the file content does.

## Layout

| Path | Role |
|------|------|
| `src/scribe_agent/main.py` | Watch loop, orchestration |
| `src/scribe_agent/mcp_github.py` | MCP invoke for create issue |
| `src/scribe_agent/git_repo.py` | Clone URL → `GitSource`, shallow clone |
| `src/scribe_agent/llama_tools.py` | Llama Stack chat + MCP tool loop |
| `src/scribe_agent/config.py` | Settings |
| `src/scribe_agent/state_store.py` | Processed content hashes |
