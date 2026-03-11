# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Source metadata tests for context indexing."""

import pytest
from unittest.mock import AsyncMock

from openviking.core.context import Context
from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking.utils.source_utils import infer_source
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig


@pytest.mark.parametrize(
    ("uri", "context_type", "expected"),
    [
        ("viking://session/acme__alice__helper/session-123", "resource", "sessions"),
        ("viking://resources/sources/documents/acme/file.md", "resource", "documents"),
        ("viking://resources/sources/imessages/acme/chat-1.md", "resource", "imessages"),
        ("viking://agent/skills/example/SKILL.md", "skill", "skill"),
        ("viking://agent/memories/events/foo.md", "memory", "memory"),
        ("viking://resources/manual/notes/today.md", "resource", "resource"),
    ],
)
def test_infer_source(uri, context_type, expected):
    assert infer_source(uri, context_type) == expected


def test_context_to_dict_includes_source():
    context = Context(
        uri="viking://resources/sources/contacts/acme/jane-doe.md",
        abstract="Jane Doe contact card",
        context_type="resource",
    )

    payload = context.to_dict()

    assert payload["source"] == "contacts"


class DummyCollection:
    def __init__(self, fields, scalar_index):
        self._meta = {"Fields": list(fields), "Description": "context"}
        self._index_meta = {"ScalarIndex": list(scalar_index)}

    def get_meta_data(self):
        return self._meta

    def update(self, fields=None, description=None):
        if fields is not None:
            self._meta["Fields"] = list(fields)
        if description is not None:
            self._meta["Description"] = description

    def get_index_meta_data(self, _index_name):
        return self._index_meta

    def update_index(self, _index_name, scalar_index, _description=None):
        self._index_meta["ScalarIndex"] = list(scalar_index)


@pytest.mark.asyncio
async def test_ensure_collection_schema_adds_source_field_and_scalar_index(monkeypatch, tmp_path):
    config = VectorDBBackendConfig(
        backend="local",
        path=str(tmp_path),
        name="context",
        dimension=2,
    )
    backend = VikingVectorIndexBackend(config)
    original_schema = CollectionSchemas.context_collection("context", 2)
    original_schema["Fields"] = [
        field for field in original_schema["Fields"] if field["FieldName"] != "source"
    ]
    original_schema["ScalarIndex"] = [
        field for field in original_schema["ScalarIndex"] if field != "source"
    ]
    collection = DummyCollection(original_schema["Fields"], original_schema["ScalarIndex"])

    monkeypatch.setattr(backend, "collection_exists", AsyncMock(return_value=True))
    monkeypatch.setattr(backend, "_get_collection", lambda: collection)
    monkeypatch.setattr(backend, "_get_meta_data", lambda coll: coll.get_meta_data())
    monkeypatch.setattr(backend, "_refresh_meta_data", lambda coll: None)

    changed = await backend.ensure_collection_schema(
        CollectionSchemas.context_collection("context", 2)
    )

    field_names = [field["FieldName"] for field in collection.get_meta_data()["Fields"]]
    index_meta = collection.get_index_meta_data("default")

    assert changed is True
    assert "source" in field_names
    assert "source" in (index_meta.get("ScalarIndex") or [])
