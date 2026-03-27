from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class McpRegistration:
    """Optional MCP registration applied at startup (Llama Stack toolgroups.register)."""

    toolgroup_id: str
    provider_id: str
    mcp_uri: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    watch_directory: str = Field(
        ...,
        description="Directory to scan for .txt files",
        validation_alias="SCRIBE_WATCH_DIRECTORY",
    )
    watch_recursive: bool = Field(False, validation_alias="SCRIBE_WATCH_RECURSIVE")

    git_clone_url: str = Field(
        ...,
        description="HTTPS or SSH clone URL for the GitHub repository (target for new issues)",
        validation_alias="SCRIBE_GIT_CLONE_URL",
    )
    git_branch: str = Field(
        ...,
        description="Branch to check out in the local clone (context for README/docs)",
        validation_alias="SCRIBE_GIT_BRANCH",
    )

    poll_interval_seconds: int = Field(60, validation_alias="SCRIBE_POLL_INTERVAL_SECONDS")
    state_file_path: str = Field(
        "/tmp/scribe-agent-state.json",
        validation_alias="SCRIBE_STATE_FILE",
    )

    llama_stack_base_url: str = Field(..., validation_alias="LLAMA_STACK_BASE_URL")
    llama_stack_api_key: str | None = Field(None, validation_alias="LLAMA_STACK_API_KEY")
    llama_stack_model_id: str | None = Field(None, validation_alias="LLAMA_STACK_MODEL_ID")

    tool_group_ids: str = Field(
        ...,
        description="Comma-separated Llama Stack tool group IDs (include GitHub MCP)",
        validation_alias="SCRIBE_TOOL_GROUP_IDS",
    )

    mcp_create_issue_tool: str = Field(
        "create_issue",
        description="MCP tool name (GitHub) to create issues; invoked via Llama Stack",
        validation_alias="SCRIBE_MCP_CREATE_ISSUE_TOOL",
    )
    mcp_create_issue_extra_json: str | None = Field(
        None,
        description="Optional JSON object merged into create-issue MCP kwargs",
        validation_alias="SCRIBE_MCP_CREATE_ISSUE_EXTRA_JSON",
    )

    agent_implementation_label: str = Field(
        "agent",
        description="GitHub label when the source asks an AI/Developer agent to implement",
        validation_alias="SCRIBE_AGENT_IMPLEMENTATION_LABEL",
    )

    mcp_registrations_json: str | None = Field(
        None,
        validation_alias="SCRIBE_MCP_REGISTRATIONS_JSON",
        description='Optional JSON list: [{"toolgroup_id":"mcp::x","provider_id":"model-context-protocol","mcp_uri":"http://host/sse"}]',
    )

    github_token: str | None = Field(
        None,
        validation_alias="GITHUB_TOKEN",
        description="Optional: HTTPS git clone for local context (not GitHub REST from this app).",
    )
    git_clone_depth: int = Field(50, validation_alias="SCRIBE_GIT_CLONE_DEPTH")
    workspace_root: str = Field("/tmp/scribe-workspaces", validation_alias="SCRIBE_WORKSPACE_ROOT")

    max_llm_iterations: int = Field(40, validation_alias="SCRIBE_MAX_LLM_ITERATIONS")
    max_source_text_chars: int = Field(120_000, validation_alias="SCRIBE_MAX_SOURCE_TEXT_CHARS")
    readme_excerpt_chars: int = Field(24_000, validation_alias="SCRIBE_README_EXCERPT_CHARS")

    dry_run_no_issues: bool = Field(False, validation_alias="SCRIBE_DRY_RUN_NO_ISSUES")

    @property
    def tool_group_id_list(self) -> list[str]:
        return [x.strip() for x in self.tool_group_ids.split(",") if x.strip()]

    @field_validator("poll_interval_seconds", "git_clone_depth", "max_llm_iterations")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("max_source_text_chars", "readme_excerpt_chars")
    @classmethod
    def _non_negative_size(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    def parsed_mcp_registrations(self) -> list[McpRegistration]:
        if not self.mcp_registrations_json:
            return []
        raw: list[Any] = json.loads(self.mcp_registrations_json)
        out: list[McpRegistration] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("SCRIBE_MCP_REGISTRATIONS_JSON must be a JSON list of objects")
            out.append(
                McpRegistration(
                    toolgroup_id=str(item["toolgroup_id"]),
                    provider_id=str(item.get("provider_id") or "model-context-protocol"),
                    mcp_uri=str(item["mcp_uri"]),
                )
            )
        return out
