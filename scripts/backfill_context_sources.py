#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Backfill canonical `source` values into existing context vector records."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, Iterable, List

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.vikingdb_manager import VikingDBManager
from openviking.utils.source_utils import infer_source
from openviking_cli.utils.config import get_openviking_config


def _iter_records(manager: VikingDBManager) -> Iterable[Dict[str, Any]]:
    collection = manager._get_collection()
    inner_collection = getattr(collection, "_Collection__collection", None)
    store_mgr = getattr(inner_collection, "store_mgr", None)

    if store_mgr is not None:
        for candidate in store_mgr.get_all_cands_data():
            if not candidate.fields:
                continue
            record = json.loads(candidate.fields)
            if candidate.vector:
                record["vector"] = candidate.vector
            if candidate.sparse_raw_terms and candidate.sparse_values:
                record["sparse_vector"] = dict(
                    zip(candidate.sparse_raw_terms, candidate.sparse_values, strict=False)
                )
            yield record
        return

    raise RuntimeError(
        "Unable to enumerate vector records for backfill: local store manager is unavailable"
    )


async def backfill_sources(dry_run: bool) -> Dict[str, Any]:
    config = get_openviking_config()
    manager = VikingDBManager(vectordb_config=config.storage.vectordb)
    collection_name = config.storage.vectordb.name
    schema = CollectionSchemas.context_collection(collection_name, config.embedding.dimension)
    schema_changed = await manager.ensure_collection_schema(schema)

    total = await manager.count()
    updated = 0
    skipped = 0
    errors: List[str] = []

    try:
        for record in _iter_records(manager):
            record.pop("_score", None)
            uri = record.get("uri", "")
            expected_source = infer_source(uri, record.get("context_type"))
            if record.get("source") == expected_source:
                skipped += 1
                continue

            record["source"] = expected_source
            if not dry_run:
                record_id = await manager.upsert(record)
                if not record_id:
                    errors.append(uri)
                    continue
            updated += 1

        return {
            "collection": collection_name,
            "total": total,
            "updated": updated,
            "skipped": skipped,
            "schema_changed": schema_changed,
            "dry_run": dry_run,
            "errors": errors,
        }
    finally:
        await manager.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report required changes without writing updates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(backfill_sources(dry_run=args.dry_run))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
