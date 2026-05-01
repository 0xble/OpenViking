# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Placeholder helpers for skill privacy values."""

from dataclasses import dataclass, field


@dataclass
class SkillPrivacyPlaceholderizationResult:
    sanitized_content: str
    original_content_blocks: list[str] = field(default_factory=list)
    replacement_content_blocks: list[str] = field(default_factory=list)
    replaced_values: dict[str, str] = field(default_factory=dict)


def build_placeholder(skill_name: str, field_name: str) -> str:
    return f"{{{{ov_privacy:skill:{skill_name}:{field_name}}}}}"


def _replace_structured_value(content: str, raw_value: str, placeholder: str) -> tuple[str, bool]:
    lines = content.splitlines(keepends=True)
    output: list[str] = []
    replaced = False

    for line in lines:
        newline = ""
        body = line
        if body.endswith("\r\n"):
            body = body[:-2]
            newline = "\r\n"
        elif body.endswith("\n"):
            body = body[:-1]
            newline = "\n"
        elif body.endswith("\r"):
            body = body[:-1]
            newline = "\r"

        delimiter_positions = [pos for pos in (body.find(":"), body.find("=")) if pos >= 0]
        if not delimiter_positions:
            output.append(line)
            continue

        delimiter_index = min(delimiter_positions)
        value_start = delimiter_index + 1
        while value_start < len(body) and body[value_start] == " ":
            value_start += 1

        prefix = body[:value_start]
        value = body[value_start:]
        quote = ""
        inner_value = value
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            quote = value[0]
            inner_value = value[1:-1]

        if inner_value == raw_value:
            replacement = f"{quote}{placeholder}{quote}" if quote else placeholder
            output.append(f"{prefix}{replacement}{newline}")
            replaced = True
        else:
            output.append(line)

    return "".join(output), replaced


def placeholderize_skill_content_with_blocks(
    content: str, skill_name: str, values: dict[str, str]
) -> SkillPrivacyPlaceholderizationResult:
    sanitized = content
    original_content_blocks: list[str] = []
    replacement_content_blocks: list[str] = []
    replaced_values: dict[str, str] = {}
    replacements = sorted(values.items(), key=lambda item: len(str(item[1])), reverse=True)

    for field_name, raw_value in replacements:
        if not raw_value:
            continue
        raw_value_str = str(raw_value)
        placeholder = build_placeholder(skill_name, field_name)
        sanitized, replaced = _replace_structured_value(sanitized, raw_value_str, placeholder)
        if replaced:
            original_content_blocks.append(raw_value_str)
            replacement_content_blocks.append(placeholder)
            replaced_values[field_name] = raw_value_str

    return SkillPrivacyPlaceholderizationResult(
        sanitized_content=sanitized,
        original_content_blocks=original_content_blocks,
        replacement_content_blocks=replacement_content_blocks,
        replaced_values=replaced_values,
    )


def placeholderize_skill_content(content: str, skill_name: str, values: dict[str, str]) -> str:
    return placeholderize_skill_content_with_blocks(content, skill_name, values).sanitized_content
