"""Chat CompletionsのSSEを読み、本文と分割されたツール呼び出しを組み立てる。"""

import asyncio
import json
import logging
import time

import httpx

from .errors import RateLimitError, UpstreamError
from .request_diagnostics import phase, request_id

logger = logging.getLogger(__name__)
MAX_EVENT_CHARS = 262144


class StreamAccumulator:
    def __init__(self, on_text):
        self.on_text = on_text
        self.text = ""
        self.calls = {}
        self.finish_reason = None
        self.usage = {}
        self.last_active = None

    async def accept(self, event: str) -> bool:
        if event.strip() == "[DONE]":
            return True
        data = json.loads(event)
        if "error" in data:
            raise UpstreamError("LLM（ストリームエラー）")
        if isinstance(data.get("usage"), dict):
            self.usage = data["usage"]
        if data.get("prompt_progress"):
            phase("llm_prefill")
        for choice in data.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            content = delta.get("content") or delta.get("refusal") or ""
            tool_calls = delta.get("tool_calls") or []
            if content or tool_calls or delta.get("reasoning_content"):
                if self.last_active is None:
                    logger.info("LLM stream first delta: request=%s", request_id())
                self.last_active = time.monotonic()
            if delta.get("reasoning_content"):
                phase("llm_reasoning")  # 思考本文は表示・記録しない。
            if content:
                if not isinstance(content, str):
                    raise ValueError("invalid content")
                self.text += content
                if len(self.text) > 65536:
                    raise ValueError("content limit")
                phase("llm_streaming")
                await self.on_text(self.text)
            for item in tool_calls:
                index = item.get("index", 0)
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 8:
                    raise ValueError("tool count limit")
                call = self.calls.setdefault(
                    index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                call["id"] += item.get("id") or ""
                function = item.get("function") or {}
                for key in ("name", "arguments"):
                    call["function"][key] += function.get(key) or ""
                if (
                    len(call["id"]) > 64
                    or len(call["function"]["name"]) > 64
                    or len(call["function"]["arguments"]) > 8000
                ):
                    raise ValueError("tool field limit")
                phase("llm_tool_arguments")
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]
        return False

    def result(self) -> dict:
        if self.finish_reason is None:
            raise UpstreamError("LLM（ストリームが途中で切断されました）")
        calls = [self.calls[index] for index in sorted(self.calls)]
        if calls and self.finish_reason != "tool_calls":
            raise UpstreamError("LLM（ツール呼び出しが未完了です）")
        for call in calls:
            if (
                not call["id"]
                or not call["function"]["name"]
                or not isinstance(json.loads(call["function"]["arguments"]), dict)
            ):
                raise ValueError("incomplete tool call")
        return {"choices": [{"message": {"content": self.text or None, "tool_calls": calls}}], "usage": self.usage}


async def read_chat_stream(settings, payload, on_text) -> dict:
    started = time.monotonic()
    first_deadline = started + settings.first_token_timeout_seconds
    accumulator = StreamAccumulator(on_text)
    timeout = httpx.Timeout(
        max(settings.first_token_timeout_seconds, settings.timeout_seconds),
        connect=min(10, settings.first_token_timeout_seconds),
        write=30,
        pool=10,
    )
    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    phase("llm_stream_connect")
    logger.info(
        "LLM stream started: request=%s first_timeout=%.1f idle_timeout=%.1f total_timeout=%.1f",
        request_id(),
        settings.first_token_timeout_seconds,
        settings.timeout_seconds,
        settings.stream_timeout_seconds,
    )
    try:
        async with asyncio.timeout(settings.stream_timeout_seconds):
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                request = client.build_request(
                    "POST",
                    settings.base_url.rstrip("/") + "/chat/completions",
                    json={**payload, "stream": True},
                    headers=headers,
                )
                response = await asyncio.wait_for(
                    client.send(request, stream=True), settings.first_token_timeout_seconds
                )
                try:
                    if response.status_code == 429:
                        logger.warning("LLM stream HTTP error: request=%s status=429", request_id())
                        raise RateLimitError()
                    if response.status_code >= 400:
                        logger.warning(
                            "LLM stream HTTP error: request=%s status=%d", request_id(), response.status_code
                        )
                        raise UpstreamError(f"LLM HTTP {response.status_code}")
                    if "text/event-stream" not in response.headers.get("content-type", "").lower():
                        # streamを無視して通常JSONを返す互換サーバーにも対応する。
                        await asyncio.wait_for(response.aread(), max(0.001, first_deadline - time.monotonic()))
                        data = response.json()
                        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                        if isinstance(text, str):
                            await on_text(text)
                        return data
                    lines = response.aiter_lines().__aiter__()
                    event = []
                    event_size = 0
                    phase("llm_wait_first_token")
                    while True:
                        deadline = (
                            accumulator.last_active + settings.timeout_seconds
                            if accumulator.last_active is not None
                            else first_deadline
                        )
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError()
                        try:
                            line = await asyncio.wait_for(anext(lines), remaining)
                        except StopAsyncIteration:
                            if event:
                                await accumulator.accept("\n".join(event))
                            break
                        if not line:
                            if event and await accumulator.accept("\n".join(event)):
                                break
                            event, event_size = [], 0
                        elif line.startswith("data:"):
                            value = line[5:].lstrip(" ")
                            event_size += len(value)
                            if event_size > MAX_EVENT_CHARS:
                                raise ValueError("event limit")
                            event.append(value)
                    return accumulator.result()
                finally:
                    await response.aclose()
    except (TimeoutError, httpx.TimeoutException) as exc:
        logger.warning(
            "LLM stream timeout: request=%s seconds=%.3f error=%s",
            request_id(),
            time.monotonic() - started,
            type(exc).__name__,
        )
        raise UpstreamError("LLM（ストリーム応答タイムアウト）") from exc
    except httpx.HTTPError as exc:
        logger.warning("LLM stream transport error: request=%s error=%s", request_id(), type(exc).__name__)
        raise UpstreamError("LLM（接続エラー）") from exc
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        logger.warning("LLM stream protocol error: request=%s error=%s", request_id(), type(exc).__name__)
        raise UpstreamError("LLM（ストリーム応答の形式不正）") from exc
    finally:
        logger.info("LLM stream closed: request=%s seconds=%.3f", request_id(), time.monotonic() - started)
