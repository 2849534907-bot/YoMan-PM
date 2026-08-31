"""
EchoMind-PM — LLM 适配层（OpenAI 兼容接口）

由 EchoMind 的 Anthropic 直连改造成统一适配层：
- 使用 OpenAI Python SDK（AsyncOpenAI），支持任意 OpenAI 兼容接口。
- 通过 base_url 切换：豆包（火山方舟 Ark）/ 官方 OpenAI / 第三方中转站。
- 通过 model 切换模型：doubao-* / gpt-* / 第三方自定义模型名。

所有需要调用 LLM 的模块（Agent、意图识别、记忆、工具、评测）统一走本层，
避免各处直接依赖具体 SDK，便于后续更换或扩展模型后端。

模型后端选择（优先级从高到低，按环境变量）：
  1. 豆包（火山方舟 Ark）：DOUBAO_API_KEY / DOUBAO_BASE_URL / DOUBAO_MODEL
  2. 通用 OpenAI 兼容：OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
"""
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# 豆包（火山方舟 Ark）OpenAI 兼容接口默认地址
DOUBAO_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
# 默认模型（可在 .env 中改为自己的推理接入点 ID ep-xxx，或开通后按需改）
DOUBAO_DEFAULT_MODEL = "doubao-1-5-pro-32k-250115"


def llm_env_config() -> Dict[str, str]:
    """
    从环境变量解析 LLM 配置。

    优先使用豆包（火山方舟 Ark）配置；未配置时回退到通用 OPENAI_* 配置。
    返回: {"api_key": ..., "base_url": ..., "model": ..., "api_mode": ...}
    """
    key = (os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY") or
           os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("DOUBAO_BASE_URL") or os.getenv("ARK_BASE_URL") or
                os.getenv("OPENAI_BASE_URL") or DOUBAO_DEFAULT_BASE_URL).strip()
    model = (os.getenv("DOUBAO_MODEL") or os.getenv("ARK_MODEL") or
             os.getenv("OPENAI_MODEL") or DOUBAO_DEFAULT_MODEL).strip()
    api_mode = (os.getenv("DOUBAO_API_MODE") or os.getenv("ARK_API_MODE") or
                os.getenv("OPENAI_API_MODE") or "chat").strip().lower()
    if api_mode not in ("chat", "responses"):
        api_mode = "chat"
    fast_model = (os.getenv("DOUBAO_FAST_MODEL") or os.getenv("ARK_FAST_MODEL") or
                  os.getenv("OPENAI_FAST_MODEL") or "").strip()
    return {"api_key": key, "base_url": base_url, "model": model, "api_mode": api_mode, "fast_model": fast_model}


class LLMClient:
    """OpenAI 兼容的异步 LLM 客户端封装，支持 Chat Completions 和 Responses 两种接口。"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = DOUBAO_DEFAULT_MODEL,
        api_mode: str = "chat",
    ):
        """
        api_mode: "chat"（默认，/chat/completions，所有模型通用）
                  "responses"（/responses，新模型如 doubao-seed-evolving 官方示例用）
        """
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self.model = model
        self.api_mode = api_mode if api_mode in ("chat", "responses") else "chat"
        # 保存原始配置，供 Responses 模式用 httpx 直连（部分 openai SDK 版本无异步 responses 属性）
        self._api_key = api_key
        self._base_url = (base_url or "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")

    @staticmethod
    def _build_messages(
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        full: List[Dict[str, str]] = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        return full

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> str:
        """
        调用 LLM，返回回复文本。根据 api_mode 自动选择 Chat Completions 或 Responses 接口。

        messages: [{"role": "user"/"assistant", "content": "..."}, ...]
        system:  可选的 system prompt，会插入到消息最前面。
        model:   可选，覆盖默认模型（用于双模型切换：快速模型/深度思考模型）。
        """
        full = self._build_messages(messages, system)
        effective_model = model or self.model
        if self.api_mode == "responses":
            return await self._call_responses(full, max_tokens=max_tokens, temperature=temperature, model=effective_model)
        return await self._call_chat(full, max_tokens=max_tokens, temperature=temperature, model=effective_model)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式调用 LLM，逐块 yield 回复文本。
        - chat 模式：使用 stream=True，逐 token 返回。
        - responses 模式：暂不支持原生流式，回退为一次性返回完整文本。
        model: 可选，覆盖默认模型。
        """
        full = self._build_messages(messages, system)
        effective_model = model or self.model

        if self.api_mode == "responses":
            text = await self._call_responses(full, max_tokens=max_tokens, temperature=temperature, model=effective_model)
            yield text
            return

        try:
            stream = await self._client.chat.completions.create(
                model=effective_model,
                messages=full,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                if (chunk.choices and len(chunk.choices) > 0
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content):
                    yield chunk.choices[0].delta.content
        except Exception as ex:
            logger.error(f"LLM 流式调用失败: {ex}")
            raise

    async def _call_chat(
        self,
        full: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> str:
        try:
            kwargs: Dict[str, Any] = {
                "model": model or self.model,
                "messages": full,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format
            resp = await self._client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            return text
        except Exception as ex:
            logger.error(f"LLM chat.completions 调用失败: {ex}")
            raise

    async def _call_responses(
        self,
        full: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        model: Optional[str] = None,
    ) -> str:
        """
        调用 Responses API（/responses），新模型如 doubao-seed-evolving 官方示例用此接口。

        与火山方舟官方示例对齐：使用 openai SDK 的 responses.create，
        content 格式为 [{"type": "input_text", "text": "..."}]。
        要求 openai>=3.0（异步客户端需有 responses 属性）。
        """
        # 将标准 messages 格式转换为 Responses API 的 input 格式
        responses_input = []
        for msg in full:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = [{"type": "input_text", "text": content}]
            responses_input.append({"role": msg["role"], "content": content})

        try:
            resp = await self._client.responses.create(
                model=model or self.model,
                input=responses_input,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            # 优先取 output_text，兜底遍历 output 列表
            text = getattr(resp, "output_text", None)
            if not text and getattr(resp, "output", None):
                for item in resp.output:
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") == "output_text":
                            text = getattr(c, "text", "")
                            break
                    if text:
                        break
            if not text:
                # 深度思考模型可能把输出 token 全用于推理，未生成正文
                has_message = any(
                    getattr(item, "type", "") == "message"
                    for item in getattr(resp, "output", [])
                )
                if not has_message:
                    logger.warning("Responses API 仅返回推理过程，未生成正文（可能 max_output_tokens 不足）")
            return text or ""
        except Exception as ex:
            logger.error(f"LLM responses 调用失败: {ex}")
            raise

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        与 chat 相同，但要求模型输出 JSON。
        - chat 模式：通过 response_format 约束；不支持时自动降级为普通调用。
        - responses 模式：通过 system/instructions 约束输出 JSON（Responses API 的 JSON 处理方式不同）。
        """
        full = self._build_messages(messages, system)

        if self.api_mode == "responses":
            # Responses API：在 system 中追加 JSON 输出约束
            json_instruction = "请严格以 JSON 格式输出，不要包含任何额外文字或 Markdown 代码块标记。"
            if full and full[0].get("role") == "system":
                full[0]["content"] = full[0]["content"] + "\n" + json_instruction
            else:
                full.insert(0, {"role": "system", "content": json_instruction})
            return await self._call_responses(full, max_tokens=max_tokens, temperature=temperature)

        # chat 模式
        try:
            return await self._call_chat(
                full,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            # 部分兼容站不支持 response_format，降级为普通调用
            return await self._call_chat(full, max_tokens=max_tokens, temperature=temperature)

    @property
    def base_url(self) -> Optional[str]:
        return getattr(self._client, "base_url", None)

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model}, base_url={self.base_url})"
