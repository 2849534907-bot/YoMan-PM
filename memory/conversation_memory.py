"""
EchoMind-PM — 多轮对话记忆管理

由 EchoMind 客服记忆管理改造而来，业务领域替换为项目管理。

三级记忆架构，模拟人类记忆机制：
  1. 工作记忆（Redis / 内存兜底）—— 当前会话的最近 N 条消息
  2. 情景记忆（ChromaDB，可选）—— 跨会话的历史对话，按语义相似度检索
  3. 用户画像（ChromaDB，可选）—— 从对话中提炼的长期偏好和实体

关键设计：
  - 上下文构建时三级记忆融合，按重要性 + 时效性排序
  - 工作记忆超过阈值时自动压缩（LLM 摘要），防止 context 爆炸
  - Redis / ChromaDB 为可选依赖：不可用时自动降级为内存/本地存储，
    保证在未安装 redis、chromadb（或缺少 C++ 编译工具）的环境也能运行。

降级策略：
  - 无 Redis  → 工作记忆与摘要存进程内存（重启丢失）
  - 无 ChromaDB → 情景记忆与用户画像降级为无操作，仅保留工作记忆
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from core.llm import LLMClient

# 可选依赖探测
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]  # 用户画像：偏好、常用实体
    summary:          str             # 当前会话摘要（压缩后）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=True)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（有 Redis 时，TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    Redis / ChromaDB 不可用时自动降级，不阻塞主流程。
    """

    WORKING_MAX   = 20    # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "doubao-1-5-pro-32k-250115",
        api_mode:     str = "chat",
    ):
        self._client = LLMClient(api_key=api_key, base_url=base_url, model=model, api_mode=api_mode)
        self._model  = model

        # ── 工作记忆后端：Redis 优先，否则进程内存 ──────────────────────────
        self._memory_backend = "memory"
        self._wm_store: Dict[str, List[Dict[str, Any]]] = {}     # (user,conv) -> msgs
        self._summary_store: Dict[str, str] = {}                 # (user,conv) -> summary
        self._redis = None
        if REDIS_AVAILABLE:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                self._memory_backend = "redis"
                logger.info(f"Redis 已连接: {redis_url}")
            except Exception as ex:
                self._redis = None
                logger.warning(f"Redis 不可用，使用内存记忆: {ex}")
        if self._memory_backend == "memory":
            logger.info("使用进程内存作为工作记忆（重启后丢失）")

        # ── 情景记忆 / 用户画像：ChromaDB 优先，否则降级为无操作 ────────────
        self._episodic = None
        self._profile  = None
        self._profile_store: Dict[str, Dict[str, Any]] = {}      # 内存画像兜底
        if CHROMADB_AVAILABLE:
            try:
                chroma = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )
                chroma.heartbeat()  # 测试连接
                logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
            except Exception:
                logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
                chroma = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )

            self._episodic = chroma.get_or_create_collection("episodic")
            self._profile  = chroma.get_or_create_collection("user_profile")
        else:
            logger.warning("未安装 chromadb，情景记忆/用户画像降级为不可用（仅保留工作记忆）")

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)

        if self._memory_backend == "redis":
            key = self._wm_key(user_id, conv_id)
            self._redis.lpush(key, json.dumps({
                "role":      msg.role.value,
                "content":   msg.content,
                "ts":        msg.timestamp.isoformat(),
                "metadata":  msg.metadata,
            }))
            self._redis.expire(key, 86400)  # 24h TTL
            if self._redis.llen(key) >= self.COMPRESS_AT:
                await self._compress(user_id, conv_id)
        else:
            mkey = self._wm_key(user_id, conv_id)
            bucket = self._wm_store.setdefault(mkey, [])
            bucket.append({
                "role":      msg.role.value,
                "content":   msg.content,
                "ts":        msg.timestamp.isoformat(),
                "metadata":  msg.metadata,
            })
            if len(bucket) >= self.COMPRESS_AT:
                await self._compress(user_id, conv_id)

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """
        从当前工作记忆中提炼用户偏好，更新用户画像。
        有 ChromaDB 时存向量库；否则写入内存画像兜底。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return

        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in messages[-10:]))
        prompt = f"""从以下对话中提炼用户偏好和关键实体，返回 JSON。
对话:
{text}

返回格式: {{"preferences": ["..."], "entities": {{"项目": [], "任务": [], "关注点": []}}}}"""
        prompt = self._safe_text(prompt)

        try:
            raw = await self._client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            s, e = raw.find("{"), raw.rfind("}") + 1
            profile_data = json.loads(raw[s:e])
        except Exception as ex:
            logger.warning(f"提炼用户画像失败: {ex}")
            return

        if self._profile is not None:
            try:
                doc_id = f"{user_id}_profile_{conv_id}"
                doc_text = self._safe_text(json.dumps(profile_data, ensure_ascii=False))
                try:
                    self._profile.delete(ids=[doc_id])
                except Exception:
                    pass
                self._profile.add(
                    ids=[doc_id],
                    documents=[doc_text],
                    metadatas=[{"user_id": user_id, "conv_id": conv_id,
                                "ts": datetime.now().isoformat()}],
                )
            except Exception as ex:
                logger.warning(f"写入用户画像失败: {ex}")
        else:
            self._profile_store[user_id] = profile_data
            logger.info(f"用户画像已更新（内存模式）: {user_id}")

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        history = await self._search_episodic(user_id, query or (recent[-1].content if recent else ""))

        profile = await self._get_profile(user_id)

        summary = ""
        if self._memory_backend == "redis":
            summary = self._redis.get(self._summary_key(user_id, conv_id)) or ""
        else:
            summary = self._summary_store.get(self._summary_key(user_id, conv_id), "")

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩：
          1. 用 LLM 对旧消息生成摘要
          2. 摘要存后端
          3. 旧消息存入情景记忆（有 ChromaDB 时）
          4. 工作记忆只保留最近 5 条
        """
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]   # 保留最近 5 条
        keep        = messages[-5:]

        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))
        prompt = self._safe_text(f"用 2-3 句话总结以下对话的关键信息：\n{text}")
        try:
            summary = (await self._client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )).strip()
        except Exception:
            summary = f"对话包含 {len(to_compress)} 条消息（摘要生成失败）"

        skey = self._summary_key(user_id, conv_id)
        if self._memory_backend == "redis":
            old_summary = self._redis.get(skey) or ""
            new_summary = self._safe_text(f"{old_summary}\n{summary}").strip()
            self._redis.setex(skey, 86400, new_summary)
        else:
            old_summary = self._summary_store.get(skey, "")
            self._summary_store[skey] = self._safe_text(f"{old_summary}\n{summary}").strip()

        if self._episodic is not None:
            await self._store_episodic(user_id, conv_id, text, summary)

        # 重置工作记忆为最近 5 条
        key = self._wm_key(user_id, conv_id)
        if self._memory_backend == "redis":
            self._redis.delete(key)
            for m in reversed(keep):
                self._redis.lpush(key, json.dumps({
                    "role": m.role.value, "content": m.content,
                    "ts": m.timestamp.isoformat(), "metadata": m.metadata,
                }))
            self._redis.expire(key, 86400)
        else:
            self._wm_store[key] = [{
                "role": m.role.value, "content": m.content,
                "ts": m.timestamp.isoformat(), "metadata": m.metadata,
            } for m in keep]
        logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key = self._wm_key(user_id, conv_id)
        if self._memory_backend == "redis":
            raws = self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        else:
            raws = self._wm_store.get(key, [])[-self.WORKING_MAX:][::-1]

        msgs = []
        for raw in reversed(raws):
            if isinstance(raw, str):
                d = json.loads(raw)
            else:
                d = raw
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, query: str) -> List[str]:
        """语义检索情景记忆。需要 ChromaDB，不可用时返回空。"""
        if self._episodic is None:
            return []
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            results = self._episodic.query(
                query_texts=[query_text],
                n_results=self.HISTORY_TOP_K,
                where={"user_id": self._safe_text(user_id)},
            )
            docs = results["documents"][0] if results["documents"] else []
            return [self._safe_text(doc) for doc in docs if isinstance(doc, str) and doc.strip()]
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> None:
        """将压缩后的对话片段存入情景记忆。需要 ChromaDB，不可用时跳过。"""
        if self._episodic is None:
            return
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            self._episodic.add(
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat(), "full_text": self._safe_text(text[:500])}],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像。"""
        if self._profile is not None:
            try:
                results = self._profile.get(where={"user_id": user_id}, limit=1)
                if results["documents"]:
                    return json.loads(results["documents"][0])
            except Exception:
                pass
        return self._profile_store.get(user_id, {})

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免后端读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
