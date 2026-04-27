# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""VolcEngine VLM backend implementation."""

import base64
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union

from openviking.telemetry import tracer
from openviking.utils.model_retry import retry_async, retry_sync
from openviking_cli.utils.async_utils import run_async

from ..base import ToolCall, VLMResponse
from .openai_vlm import OpenAIVLM

logger = logging.getLogger(__name__)


class _ResponseIDCache:
    def __init__(self, max_size: int = 1024):
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._values: OrderedDict[str, str] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._values:
                return None
            value = self._values.pop(key)
            self._values[key] = value
            return value

    def set(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._values:
                self._values.pop(key)
            self._values[key] = value
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)


def _coerce_response_cache_max_size(value: Any, default: int = 1024) -> int:
    try:
        max_size = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max_size if max_size > 0 else default


class VolcEngineVLM(OpenAIVLM):
    """VolcEngine VLM backend with Chat Completions API support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        config = {**(config or {}), **kwargs}
        super().__init__(config)
        self._sync_client = None
        self._async_client = None
        self._response_cache = _ResponseIDCache(
            max_size=_coerce_response_cache_max_size(config.get("response_cache_max_size", 1024))
        )
        self.provider = "volcengine"

        if not self.api_base:
            self.api_base = "https://ark.cn-beijing.volces.com/api/v3"
        if not self.model:
            self.model = "doubao-seed-2-0-pro-260215"

    def _get_response_id_cache_key(
        self,
        messages: List[Dict[str, Any]],
        segment_index: int = 0,
    ) -> str:
        parts: list[str] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
            else:
                parts.append(str(content))
        text = " ".join(parts).lower()
        words = re.findall(r"[a-z0-9]+", text)
        slug = "_".join(words[:4]) if words else "empty"
        digest = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:8]
        return f"prefix:seg{segment_index}_{slug}_{digest}"

    def _get_or_create_from_segments(self, segments: List[List[Dict[str, Any]]], end_idx: int):
        """Return cached response id synchronously when possible, otherwise create it."""
        if end_idx <= 0 or not segments:
            return None
        end_idx = min(end_idx, len(segments))
        first_key = self._get_response_id_cache_key(segments[0], 0)
        first_cached = self._response_cache.get(first_key)
        if end_idx == 1 and first_cached:
            return first_cached
        return run_async(
            self._get_or_create_from_segments_async(
                segments,
                end_idx,
                first_cached=(first_key, first_cached),
            )
        )

    async def _get_or_create_from_segments_async(
        self,
        segments: List[List[Dict[str, Any]]],
        end_idx: int,
        first_cached: Optional[tuple[str, Optional[str]]] = None,
    ) -> Optional[str]:
        if end_idx <= 0 or not segments:
            return None
        end_idx = min(end_idx, len(segments))
        previous_response_id = None
        client = None
        for idx in range(end_idx):
            key = self._get_response_id_cache_key(segments[idx], idx)
            if idx == 0 and first_cached and first_cached[0] == key:
                cached = first_cached[1]
            else:
                cached = self._response_cache.get(key)
            if cached:
                previous_response_id = cached
                continue

            if client is None:
                client = self.get_async_client()
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": segments[idx],
            }
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            response = await self._create_cached_response_async(client, kwargs)
            previous_response_id = response.id
            self._response_cache.set(key, previous_response_id)
        return previous_response_id

    async def _create_cached_response_async(self, client: Any, kwargs: Dict[str, Any]) -> Any:
        responses = getattr(client, "responses", None)
        create = getattr(responses, "create", None)
        if create is None:
            raise AttributeError(
                "VolcEngine SDK client does not expose responses.create, which is required "
                "for previous_response_id cache chaining. Upgrade volcenginesdkarkruntime "
                "to a version with Responses API support."
            )

        async def _do_call():
            return await create(**kwargs)

        return await retry_async(
            _do_call,
            max_retries=self.max_retries,
            operation_name="VolcEngineVLM.responses.create",
            logger=logger,
        )

    def _parse_tool_calls(self, message) -> List[ToolCall]:
        """Parse tool calls from VolcEngine response message."""
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return tool_calls

    def _build_vlm_response(self, response, has_tools: bool) -> Union[str, VLMResponse]:
        """Build response from Chat Completions response. Returns str or VLMResponse based on has_tools."""
        choice = response.choices[0]
        message = choice.message
        tracer.info(f"message.content={message.content}")
        if has_tools:
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "prompt_tokens_details": getattr(response.usage, "prompt_tokens_details", None),
                }

            return VLMResponse(
                content=message.content,
                tool_calls=self._parse_tool_calls(message),
                finish_reason=choice.finish_reason or "stop",
                usage=usage,
            )
        return message.content or ""

    def get_client(self):
        """Get sync client"""
        if self._sync_client is None:
            try:
                import volcenginesdkarkruntime
            except ImportError:
                raise ImportError(
                    "Please install volcenginesdkarkruntime: pip install volcenginesdkarkruntime"
                )
            self._sync_client = volcenginesdkarkruntime.Ark(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self._sync_client

    def get_async_client(self):
        """Get async client"""
        if self._async_client is None:
            try:
                import volcenginesdkarkruntime
            except ImportError:
                raise ImportError(
                    "Please install volcenginesdkarkruntime: pip install volcenginesdkarkruntime"
                )
            self._async_client = volcenginesdkarkruntime.AsyncArk(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self._async_client

    def get_completion(
        self,
        prompt: str = "",
        thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, VLMResponse]:
        """Get text completion via Chat Completions API."""
        kwargs_messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model or "doubao-seed-2-0-pro-260215",
            "messages": kwargs_messages,
            "temperature": self.temperature,
            "thinking": {"type": "disabled" if not thinking else "enabled"},
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        client = self.get_client()

        def _do_call():
            t0 = time.perf_counter()
            response = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            self._update_token_usage_from_response(response, duration_seconds=elapsed)
            result = self._build_vlm_response(response, has_tools=bool(tools))
            if tools:
                return result
            return self._clean_response(str(result))

        return retry_sync(
            _do_call,
            max_retries=self.max_retries,
            operation_name="VolcEngineVLM.get_completion",
            logger=logger,
        )

    @tracer("volcengine.vlm.call", ignore_result=True, ignore_args=False)
    async def get_completion_async(
        self,
        prompt: str = "",
        thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, VLMResponse]:
        """Get text completion asynchronously via Chat Completions API."""
        kwargs_messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model or "doubao-seed-2-0-pro-260215",
            "messages": kwargs_messages,
            "temperature": self.temperature,
            "thinking": {"type": "disabled" if not thinking else "enabled"},
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        # 用 tracer.info 打印请求
        tracer.info(f"request: {json.dumps(kwargs_messages, ensure_ascii=False, indent=2)}")

        client = self.get_async_client()

        async def _do_call():
            t0 = time.perf_counter()
            response = await client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            self._update_token_usage_from_response(response, duration_seconds=elapsed)
            result = self._build_vlm_response(response, has_tools=bool(tools))
            if tools:
                return result
            return self._clean_response(str(result))

        return await retry_async(
            _do_call,
            max_retries=self.max_retries,
            operation_name="VolcEngineVLM.get_completion_async",
            logger=logger,
        )

    def _detect_image_format(self, data: bytes) -> str:
        """Detect image format from magic bytes.

        Returns the MIME type, or raises ValueError for unsupported formats like SVG.

        Supported formats per VolcEngine docs:
        https://www.volcengine.com/docs/82379/1362931
        - JPEG, PNG, GIF, WEBP, BMP, TIFF, ICO, DIB, ICNS, SGI, JPEG2000, HEIC, HEIF
        """
        if len(data) < 12:
            logger.warning(f"[VolcEngineVLM] Image data too small: {len(data)} bytes")
            return "image/png"

        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        # JPEG: FF D8
        elif data[:2] == b"\xff\xd8":
            return "image/jpeg"
        # GIF: GIF87a or GIF89a
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        # WEBP: RIFF....WEBP
        elif data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
            return "image/webp"
        # BMP: BM
        elif data[:2] == b"BM":
            return "image/bmp"
        # TIFF (little-endian): 49 49 2A 00
        # TIFF (big-endian): 4D 4D 00 2A
        elif data[:4] == b"II*\x00" or data[:4] == b"MM\x00*":
            return "image/tiff"
        # ICO: 00 00 01 00
        elif data[:4] == b"\x00\x00\x01\x00":
            return "image/ico"
        # ICNS: 69 63 6E 73 ("icns")
        elif data[:4] == b"icns":
            return "image/icns"
        # SGI: 01 DA
        elif data[:2] == b"\x01\xda":
            return "image/sgi"
        # JPEG2000: 00 00 00 0C 6A 50 20 20 (JP2 signature)
        elif data[:8] == b"\x00\x00\x00\x0cjP  " or data[:4] == b"\xff\x4f\xff\x51":
            return "image/jp2"
        # HEIC/HEIF: ftyp box with heic/heif brand
        # 00 00 00 XX 66 74 79 70 68 65 69 63 (heic)
        # 00 00 00 XX 66 74 79 70 68 65 69 66 (heif)
        elif len(data) >= 12 and data[4:8] == b"ftyp":
            brand = data[8:12]
            if brand == b"heic":
                return "image/heic"
            elif brand == b"heif":
                return "image/heif"
            elif brand[:3] == b"mif":
                return "image/heif"
        # SVG (not supported)
        elif data[:4] == b"<svg" or (data[:5] == b"<?xml" and b"<svg" in data[:100]):
            raise ValueError(
                "SVG format is not supported by VolcEngine VLM API. "
                "Supported formats: JPEG, PNG, GIF, WEBP, BMP, TIFF, ICO, ICNS, SGI, JPEG2000, HEIC, HEIF"
            )

        # Unknown format - log and default to PNG
        logger.warning(f"[VolcEngineVLM] Unknown image format, magic bytes: {data[:16].hex()}")
        return "image/png"

    def _prepare_image(self, image: Union[str, Path, bytes]) -> Dict[str, Any]:
        """Prepare image data"""
        if isinstance(image, bytes):
            b64 = base64.b64encode(image).decode("utf-8")
            mime_type = self._detect_image_format(image)
            logger.info(
                f"[VolcEngineVLM] Preparing image from bytes, size={len(image)}, detected mime={mime_type}"
            )
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            }
        elif isinstance(image, Path) or (
            isinstance(image, str) and not image.startswith(("http://", "https://"))
        ):
            path = Path(image)
            suffix = path.suffix.lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".dib": "image/bmp",
                ".tiff": "image/tiff",
                ".tif": "image/tiff",
                ".ico": "image/ico",
                ".icns": "image/icns",
                ".sgi": "image/sgi",
                ".j2c": "image/jp2",
                ".j2k": "image/jp2",
                ".jp2": "image/jp2",
                ".jpc": "image/jp2",
                ".jpf": "image/jp2",
                ".jpx": "image/jp2",
                ".heic": "image/heic",
                ".heif": "image/heif",
            }.get(suffix, "image/png")
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            }
        else:
            return {"type": "image_url", "image_url": {"url": image}}

    def get_vision_completion(
        self,
        prompt: str = "",
        images: Optional[List[Union[str, Path, bytes]]] = None,
        thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, VLMResponse]:
        """Get vision completion via Chat Completions API."""
        if messages:
            kwargs_messages = messages
        else:
            content = []
            if images:
                content.extend(self._prepare_image(img) for img in images)
            if prompt:
                content.append({"type": "text", "text": prompt})
            kwargs_messages = [{"role": "user", "content": content}]

        kwargs = {
            "model": self.model or "doubao-seed-2-0-pro-260215",
            "messages": kwargs_messages,
            "temperature": self.temperature,
            "thinking": {"type": "disabled" if not thinking else "enabled"},
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        client = self.get_client()

        def _do_call():
            t0 = time.perf_counter()
            response = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            self._update_token_usage_from_response(response, duration_seconds=elapsed)
            result = self._build_vlm_response(response, has_tools=bool(tools))
            if tools:
                return result
            return self._clean_response(str(result))

        return retry_sync(
            _do_call,
            max_retries=self.max_retries,
            operation_name="VolcEngineVLM.get_vision_completion",
            logger=logger,
        )

    async def get_vision_completion_async(
        self,
        prompt: str = "",
        images: Optional[List[Union[str, Path, bytes]]] = None,
        thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, VLMResponse]:
        """Get vision completion asynchronously via Chat Completions API."""
        if messages:
            kwargs_messages = messages
        else:
            content = []
            if images:
                content.extend(self._prepare_image(img) for img in images)
            if prompt:
                content.append({"type": "text", "text": prompt})
            kwargs_messages = [{"role": "user", "content": content}]

        kwargs = {
            "model": self.model or "doubao-seed-2-0-pro-260215",
            "messages": kwargs_messages,
            "temperature": self.temperature,
            "thinking": {"type": "disabled" if not thinking else "enabled"},
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        client = self.get_async_client()

        async def _do_call():
            t0 = time.perf_counter()
            response = await client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            self._update_token_usage_from_response(response, duration_seconds=elapsed)
            result = self._build_vlm_response(response, has_tools=bool(tools))
            if tools:
                return result
            return self._clean_response(str(result))

        return await retry_async(
            _do_call,
            max_retries=self.max_retries,
            operation_name="VolcEngineVLM.get_vision_completion_async",
            logger=logger,
        )
