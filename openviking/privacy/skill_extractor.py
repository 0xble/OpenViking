# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Extract sensitive values from skill content."""

from dataclasses import dataclass, field

from openviking.privacy.skill_placeholder import placeholderize_skill_content_with_blocks
from openviking.prompts import render_prompt
from openviking_cli.utils.config import get_openviking_config
from openviking_cli.utils.llm import parse_json_from_response


@dataclass
class SkillPrivacyExtractionResult:
    values: dict[str, str]
    original_content: str
    sanitized_content: str
    original_content_blocks: list[str] = field(default_factory=list)
    replacement_content_blocks: list[str] = field(default_factory=list)


def _extract_structured_values(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in content.splitlines():
        delimiter_positions = [pos for pos in (line.find(":"), line.find("=")) if pos >= 0]
        if not delimiter_positions:
            continue
        delimiter_index = min(delimiter_positions)
        key = line[:delimiter_index].strip()
        value = line[delimiter_index + 1 :].strip()
        if not key or not value or " " in key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            values[key] = value
    return values


async def extract_skill_privacy_values(
    *,
    skill_name: str,
    skill_description: str,
    content: str,
) -> SkillPrivacyExtractionResult:
    prompt = render_prompt(
        "skill.privacy_extraction",
        {
            "skill_name": skill_name,
            "skill_description": skill_description,
            "skill_content": content,
        },
    )
    response = await get_openviking_config().vlm.get_completion_async(prompt)
    data = parse_json_from_response(response) or {}

    values: dict[str, str] = {}
    if isinstance(data, dict):
        raw_values = data.get("values", {})
        if isinstance(raw_values, dict):
            values = {
                str(key): "" if value is None else str(value)
                for key, value in raw_values.items()
                if str(key).strip()
            }
    if not values:
        values = _extract_structured_values(content)

    placeholder_result = placeholderize_skill_content_with_blocks(content, skill_name, values)
    return SkillPrivacyExtractionResult(
        values=placeholder_result.replaced_values,
        original_content=content,
        sanitized_content=placeholder_result.sanitized_content,
        original_content_blocks=placeholder_result.original_content_blocks,
        replacement_content_blocks=placeholder_result.replacement_content_blocks,
    )
