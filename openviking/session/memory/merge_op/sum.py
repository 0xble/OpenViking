# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Sum merge operation - numeric addition.
"""

from typing import Any, Type

from openviking.session.memory.merge_op.base import (
    FieldType,
    MergeOp,
    MergeOpBase,
    get_python_type_for_field,
)


class SumOp(MergeOpBase):
    """Sum merge operation - numeric addition."""

    op_type = MergeOp.SUM

    def get_output_schema_type(self, field_type: FieldType) -> Type[Any]:
        return get_python_type_for_field(field_type, default=int)

    def get_output_schema_description(self, field_description: str) -> str:
        return f"add for '{field_description}'"

    def apply(self, current_value: Any, patch_value: Any) -> Any:
        # None 或空值保留原值
        if patch_value is None or patch_value == "":
            return current_value
        if current_value is None:
            return patch_value
        should_use_float = isinstance(current_value, float) or isinstance(patch_value, float)
        number_type = float if should_use_float else int
        try:
            current_number = number_type(current_value)
        except (ValueError, TypeError):
            return patch_value
        try:
            patch_number = number_type(patch_value)
        except (ValueError, TypeError):
            return current_number
        return current_number + patch_number
