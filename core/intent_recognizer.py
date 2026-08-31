"""
EchoMind-PM — 端到端意图识别（项目管理版）

由 EchoMind 客服意图识别改造而来，业务领域替换为项目管理。

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from core.llm import LLMClient

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    GREETING   = "greeting"    # 问候
    PLAN       = "plan"        # 项目规划：拆解目标、定义任务
    SCHEDULE   = "schedule"    # 排期：工期、排期、依赖、分配
    TRACK      = "track"       # 进度跟踪：状态、完成度、逾期
    RISK       = "risk"        # 风险与阻塞
    REPORT     = "report"      # 汇报总结：周报、摘要、复盘
    QUERY      = "query"       # 通用查询：项目/任务信息询问
    ESCALATION = "escalation"  # 要求升级/转人工
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.GREETING:   ["你好", "嗨，帮我看看项目", "早上好"],
    IntentCategory.PLAN:       ["帮我规划这个项目", "把这个需求拆解成任务", "制定项目里程碑", "帮我做 WBS 分解"],
    IntentCategory.SCHEDULE:   ["帮我排一下工期", "安排一下任务顺序", "这个任务需要多久", "分配任务给负责人"],
    IntentCategory.TRACK:      ["项目进度怎么样了", "查询任务状态", "这个任务完成了吗", "有哪些任务逾期了"],
    IntentCategory.RISK:       ["有什么项目风险", "这个任务卡住了", "识别一下阻塞点", "进度有风险吗"],
    IntentCategory.REPORT:     ["生成周报", "汇总一下项目情况", "写个进度总结", "开站会要说什么", "项目复盘"],
    IntentCategory.QUERY:      ["这个项目有哪些任务", "任务列表是什么", "项目背景是什么"],
    IntentCategory.ESCALATION: ["我要找负责人", "转人工处理", "这个需要领导审批"],
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻", "火烧眉毛"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now", "明天截止"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点", "本周"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器（项目管理版）。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "doubao-1-5-pro-32k-250115",
        confidence_threshold: float = 0.5,
        api_mode: str = "chat",
        use_llm: bool = True,
    ):
        self.client    = LLMClient(api_key=api_key, base_url=base_url, model=model, api_mode=api_mode)
        self.model     = model
        self.threshold = confidence_threshold
        self.use_llm   = use_llm   # False 时仅用关键词模式匹配，跳过 LLM 意图识别（大幅提速）
        # 第三方兼容 API（如豆包中转站）通常不支持 Embedding，禁用该策略。
        # 使用稳定的本地字符 n-gram 向量作为轻量兜底，保证三路融合链路真实可跑。
        self._embedding_enabled = not bool(base_url)

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        pat = self._pattern_recognize(message)

        if self.use_llm:
            # LLM 和 Embedding 并行（Embedding 不可用时跳过）
            llm_task = asyncio.create_task(self._llm_recognize(message, history))
            emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None

            if emb_task:
                llm, emb = await asyncio.gather(llm_task, emb_task)
            else:
                llm = await llm_task
                emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

            intent = self._vote(llm, emb, pat)
            confidence = llm["confidence"]
            reasoning = llm.get("reasoning", "")
        else:
            # 仅用关键词模式匹配（毫秒级，跳过 LLM 意图识别）
            intent = pat["intent"]
            confidence = pat["confidence"]
            reasoning = "pattern_only"
        entities = await self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            entities=entities,
            reasoning=reasoning,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是项目管理意图分析专家。根据示例判断用户意图，返回 JSON。

示例:
{examples}

{ctx}
用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            raw = await self.client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1,
            )
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        patterns = {
            IntentCategory.ESCALATION: ["负责人", "领导", "转人工", "审批", "supervisor"],
            IntentCategory.PLAN:       ["规划", "拆解", "分解", "wbs", "里程碑", "计划", "plan"],
            IntentCategory.SCHEDULE:   ["排期", "工期", "安排", "多久", "什么时候", "schedule", "依赖"],
            IntentCategory.TRACK:      ["进度", "状态", "完成", "逾期", "延迟", "跟踪", "progress", "status"],
            IntentCategory.RISK:       ["风险", "阻塞", "卡住", "问题", "隐患", "risk", "blocked"],
            IntentCategory.REPORT:     ["周报", "月报", "汇报", "总结", "摘要", "复盘", "站会", "report"],
            IntentCategory.QUERY:      ["?", "？", "有哪些", "列表", "是什么", "多少", "list"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
        }
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best_score, best_cat = score, cat
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> IntentCategory:
        """加权投票。embedding 不可用时权重自动转移到 LLM 和 Pattern。"""
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"]
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"]
            return IntentCategory.OTHER

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        return best if scores[best] >= self.threshold else IntentCategory.OTHER

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    async def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用 LLM 从消息中提取结构化实体（项目管理维度）。"""
        message = self._clean_text(message)
        prompt = f"""从项目管理相关消息中提取实体，返回 JSON（字段值为列表，没有则为空列表）:
消息: "{message}"
格式: {{"project":[],"task":[],"owner":[],"deadline":[],"priority":[],"status":[],"risk":[]}}"""
        prompt = self._clean_text(prompt)
        try:
            raw = await self.client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e])
        except Exception:
            return {"project": [], "task": [], "owner": [], "deadline": [], "priority": [], "status": [], "risk": []}

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client._client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="text-embedding-3-small", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.ESCALATION:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.RISK:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str) -> str:
        return self._clean_text(message)[:200]

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
