import asyncio
import json
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
MODEL_NAME = os.environ.get("XAI_MODEL") or os.environ.get("GROK_MODEL") or "grok-4.3"
XAI_CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"


class LLMError(RuntimeError):
    """Raised when Grok/xAI does not return a usable response."""


def _api_key() -> str:
    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not api_key:
        raise LLMError("XAI_API_KEY is not configured. Add your Grok/xAI key as XAI_API_KEY in backend/.env")
    return api_key


def _extract_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    chunks: list[str] = []
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "\n".join(chunks).strip()


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _json_object_slice(raw: str) -> str:
    text = _strip_code_fence(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError("LLM response did not contain a JSON object")
    return text[start : end + 1]


def parse_json_object(raw: str) -> dict[str, Any]:
    candidates = [_strip_code_fence(raw)]
    sliced = _json_object_slice(raw)
    if sliced not in candidates:
        candidates.append(sliced)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise LLMError("LLM response JSON was not an object")
            return parsed
        except json.JSONDecodeError as exc:
            last_error = exc

    raise LLMError(f"LLM returned malformed JSON: {last_error}")


def parse_markdown_bullets(raw: str) -> str:
    return raw.strip()


async def call_claude_json(system_prompt: str, payload: dict[str, Any], max_tokens: int = 4096) -> dict[str, Any]:
    text = await call_claude_text(system_prompt, payload, max_tokens=max_tokens, json_mode=True)
    return parse_json_object(text)


async def call_claude_text(
    system_prompt: str,
    payload: dict[str, Any],
    max_tokens: int = 2048,
    json_mode: bool = False,
) -> str:
    api_key = _api_key()
    body = json.dumps(payload, ensure_ascii=False)
    last_error: Exception | None = None
    
    # Determine the completions URL and model dynamically based on API key prefix
    api_url = XAI_CHAT_COMPLETIONS_URL
    model = MODEL_NAME
    provider = "Grok"
    
    if api_key.startswith("gsk_"):
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        provider = "Groq"
        if "grok" in model.lower():
            model = "llama-3.3-70b-versatile"

    request_body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": body},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if json_mode:
        request_body["response_format"] = {"type": "json_object"}

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=90.0, trust_env=False) as client:
                response = await client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )
                response.raise_for_status()
                response_json = response.json()
            text = _extract_text(response_json)
            if not text:
                raise LLMError("LLM returned an empty response")
            return text
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                await asyncio.sleep(1.5)
                continue
            raise LLMError(f"Could not connect to {provider} API: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if attempt < max_attempts - 1 and exc.response.status_code in {408, 429, 500, 502, 503, 504}:
                sleep_time = 2.0
                if exc.response.status_code == 429:
                    # Attempt to extract rate limit retry time from headers or body
                    retry_after = exc.response.headers.get("retry-after") or exc.response.headers.get("x-ratelimit-reset")
                    if retry_after:
                        try:
                            sleep_time = float(retry_after)
                        except ValueError:
                            sleep_time = 6.0
                    else:
                        try:
                            body_json = exc.response.json()
                            msg = body_json.get("error", {}).get("message", "")
                            match = re.search(r"try again in ([\d\.]+)s", msg)
                            if match:
                                sleep_time = float(match.group(1)) + 0.5
                            else:
                                sleep_time = 6.0
                        except Exception:
                            sleep_time = 6.0
                print(f"Rate limited or transient error ({exc.response.status_code}) on attempt {attempt + 1}. Sleeping for {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
                continue
            detail = exc.response.text[:800]
            raise LLMError(f"{provider} request failed ({exc.response.status_code}): {detail}") from exc

    raise LLMError(f"{provider} call failed: {last_error}")
