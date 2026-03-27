from __future__ import annotations

import json
import logging
import re
from typing import Any

from llama_stack_client import LlamaStackClient

from scribe_agent.config import Settings
from scribe_agent.llama_tools import tool_invocation_content_as_text

logger = logging.getLogger(__name__)


def invoke_mcp_tool(client: LlamaStackClient, tool_name: str, kwargs: dict[str, Any]) -> str:
    inv = client.tool_runtime.invoke_tool(tool_name=tool_name, kwargs=kwargs)
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

    text = invoke_mcp_tool(client, tool, kwargs)
    parsed = parse_json_loose(text)
    if parsed is not None:
        url = _extract_issue_url_from_parsed(parsed)
        if url:
            return url
    if text.strip().startswith("http"):
        return text.strip().split()[0]
    return text.strip() or "(MCP returned no issue URL; check tool output)"
