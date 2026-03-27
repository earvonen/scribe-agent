from __future__ import annotations

import json
import logging
import re
from typing import Any

from llama_stack_client import LlamaStackClient

from scribe_agent.config import Settings
from scribe_agent.llama_tools import tool_invocation_content_as_text

logger = logging.getLogger(__name__)


def resolve_tool_group_for_tool_name(
    client: LlamaStackClient,
    tool_group_ids: list[str],
    tool_name: str,
) -> str | None:
    """Find which registered tool group exposes ``tool_name`` (first match)."""
    for gid in tool_group_ids:
        try:
            defs = client.tool_runtime.list_tools(tool_group_id=gid)
        except Exception as e:
            logger.debug("list_tools failed for group %r: %s", gid, e)
            continue
        for d in defs:
            n = getattr(d, "name", None) or (d.get("name") if isinstance(d, dict) else None)
            if n == tool_name:
                return gid
    return None


def invoke_mcp_tool(
    client: LlamaStackClient,
    tool_name: str,
    kwargs: dict[str, Any],
    tool_group_id: str,
) -> str:
    inv = client.tool_runtime.invoke_tool(
        tool_name=tool_name,
        kwargs=kwargs,
        extra_body={"tool_group_id": tool_group_id},
    )
    if inv.error_message:
        raise RuntimeError(f"MCP tool {tool_name!r} failed: {inv.error_message}")
    return tool_invocation_content_as_text(inv.content)


def parse_json_loose(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _ensure_issue_write_method(kwargs: dict[str, Any], tool: str, method_setting: str | None) -> None:
    """GitHub MCP ``issue_write`` requires a ``method`` argument (e.g. ``create``)."""
    if "method" in kwargs and kwargs["method"] is not None and str(kwargs["method"]).strip():
        return
    explicit = (method_setting or "").strip()
    if explicit:
        kwargs["method"] = explicit
    elif tool.strip() == "issue_write":
        kwargs["method"] = "create"


def _response_looks_like_tool_failure(text: str) -> bool:
    """MCP sometimes returns HTTP 200 with a plain-text error in the body."""
    t = text.strip().lower()
    if not t:
        return False
    return "missing required parameter" in t or "required parameter:" in t


def _extract_issue_url_from_parsed(parsed: Any) -> str | None:
    if isinstance(parsed, dict):
        if "html_url" in parsed and isinstance(parsed["html_url"], str):
            return parsed["html_url"]
        for k in ("issue", "data", "result"):
            if k in parsed:
                u = _extract_issue_url_from_parsed(parsed[k])
                if u:
                    return u
    return None


def create_issue_via_mcp(
    client: LlamaStackClient,
    settings: Settings,
    owner: str,
    repo: str,
    title: str,
    body: str,
    labels: list[str],
) -> str:
    tool = settings.mcp_create_issue_tool.strip()
    if not tool:
        raise ValueError("SCRIBE_MCP_CREATE_ISSUE_TOOL must be non-empty")

    kwargs: dict[str, Any] = {
        "owner": owner,
        "repo": repo,
        "title": title,
        "body": body,
    }
    lab = [x.strip() for x in labels if x.strip()]
    if lab:
        kwargs["labels"] = lab

    if settings.mcp_create_issue_extra_json:
        extra = json.loads(settings.mcp_create_issue_extra_json)
        if not isinstance(extra, dict):
            raise ValueError("SCRIBE_MCP_CREATE_ISSUE_EXTRA_JSON must be a JSON object")
        kwargs.update(extra)

    raw_labels = kwargs.get("labels")
    if isinstance(raw_labels, str):
        kwargs["labels"] = [raw_labels] if raw_labels.strip() else []
    elif raw_labels is None:
        kwargs.pop("labels", None)
    elif isinstance(raw_labels, list):
        kwargs["labels"] = [str(x) for x in raw_labels if str(x).strip()]

    _ensure_issue_write_method(kwargs, tool, settings.mcp_create_issue_method)

    if settings.mcp_invoke_tool_group_id:
        group_id = settings.mcp_invoke_tool_group_id.strip()
    else:
        group_id = resolve_tool_group_for_tool_name(
            client, settings.tool_group_id_list, tool
        )
    if not group_id:
        raise RuntimeError(
            f"MCP tool {tool!r} not found in any of SCRIBE_TOOL_GROUP_IDS="
            f"{settings.tool_group_ids!r}. Set SCRIBE_MCP_INVOKE_TOOL_GROUP_ID to the GitHub MCP group id."
        )

    text = invoke_mcp_tool(client, tool, kwargs, tool_group_id=group_id)
    if _response_looks_like_tool_failure(text):
        raise RuntimeError(text.strip())
    parsed = parse_json_loose(text)
    if parsed is not None:
        url = _extract_issue_url_from_parsed(parsed)
        if url:
            return url
    if text.strip().startswith("http"):
        return text.strip().split()[0]
    return text.strip() or "(MCP returned no issue URL; check tool output)"
