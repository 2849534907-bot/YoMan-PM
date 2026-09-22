"""
EchoMind-PM — 项目管理 AI Agent 编排器

由 EchoMind 智能客服多 Agent 编排改造而来，业务领域替换为项目管理。

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 GeneralAgent

并行协作：
  - 复杂问题（如"进度滞后 + 有风险"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低于阈值 → 自动升级到更高级 Agent 或转人工
"""
import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.llm import LLMClient
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL    = "general"     # 通用项目管理助手（兜底）
    PLANNER    = "planner"     # 项目规划：拆解任务、里程碑、排期
    TRACKER    = "tracker"     # 进度跟踪：状态、进度、逾期、风险
    REPORTER   = "reporter"    # 汇报总结：周报、摘要、复盘
    ESCALATION = "escalation"  # 人工升级（占位）


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None
    urgency:     Optional[UrgencyLevel]   = None
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_override: Optional[str] = None  # 用户手动指定模型，跳过自动选择
    images: Optional[List[str]] = None  # 图片 data URL 列表（多模态输入）


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    model:       str = ""  # 实际使用的模型


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str

    def __init__(self, client: LLMClient, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model  = model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()

    async def handle(self, req: Request, model: Optional[str] = None) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            content = await self._call_llm(req, model=model)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
            )

    async def _call_llm(self, req: Request, model: Optional[str] = None) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        if req.images:
            content = [{"type": "text", "text": _clean(req.message)}]
            for _url in req.images:
                content.append({"type": "image_url", "image_url": {"url": _url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": _clean(req.message)})

        return await self._client.chat(
            messages,
            system=self._build_system_prompt(req),
            max_tokens=2048,
            temperature=0.3,
            model=model,
        )

    async def handle_stream(self, req: Request, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        流式处理请求，逐块 yield 回复文本。
        与 handle() 相同的 prompt 构建逻辑，但使用 LLM 流式接口。
        """
        self.stats.total += 1
        t0 = time.monotonic()
        try:
            def _clean(s: str) -> str:
                return s.encode("utf-8", errors="ignore").decode("utf-8")

            messages = []
            if req.context:
                messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
                messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
            if req.images:
                content = [{"type": "text", "text": _clean(req.message)}]
                for _url in req.images:
                    content.append({"type": "image_url", "image_url": {"url": _url}})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": _clean(req.message)})

            async for chunk in self._client.chat_stream(
                messages,
                system=self._build_system_prompt(req),
                max_tokens=2048,
                temperature=0.3,
                model=model,
            ):
                yield chunk

            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 流式处理失败: {ex}")
            yield "\n\n[流式处理中断，请稍后重试]"

    def _build_system_prompt(self, req: Request) -> str:
        """把动态加载的 Skills 拼入 system prompt，让业务规则随请求生效。"""
        if self._skill_manager is None:
            return self.system_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        keywords = ["转人工", "人工介入", "负责人审批", "escalate", "specialist", "无法处理"]
        return any(kw in content for kw in keywords)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    system_prompt = (
        "你是 EchoMind-PM 项目管理助手。友好、简洁地回答用户关于项目管理的问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业项目管理功能。"
        "可协助用户了解项目整体情况、解释项目管理概念。"
        "当用户上传文件或图片时，请先读取其中的内容，并尽量结合项目管理/工作场景给出解读、总结或建议；"
        "如果内容与工作完全无关，可简短说明后礼貌引导回项目管理话题。"
    )


class PlannerAgent(BaseAgent):
    agent_type    = AgentType.PLANNER
    system_prompt = (
        "你是项目规划专家。专注于：项目目标拆解、WBS 工作分解、里程碑设定、"
        "任务依赖关系、资源与工期排期。"
        "输出清晰、可执行、可验收的规划结果，每个任务要包含：任务名称、负责人、工期、依赖、交付物。"
        "规划时先明确项目目标和范围，再逐层拆解，最后给出排期建议。"
    )


class TrackerAgent(BaseAgent):
    agent_type    = AgentType.TRACKER
    system_prompt = (
        "你是项目进度跟踪专家。专注于：任务状态查询、进度更新、逾期预警、风险与阻塞识别。"
        "回答必须基于当前项目的实际任务数据；信息不足时说明需要确认，不得编造进度。"
        "发现风险或阻塞时，要明确指出影响、建议的应对措施和需要谁跟进。"
    )


class ReporterAgent(BaseAgent):
    agent_type    = AgentType.REPORTER
    system_prompt = (
        "你是项目汇报专家。专注于：周报、项目进度摘要、站会要点、项目复盘。"
        "输出结构清晰、重点突出、数据可核对的汇报内容，按「已完成 / 进行中 / 风险阻塞 / 下一步」组织。"
        "数据要基于项目实际任务状态，不确定的部分明确标注。"
    )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器（项目管理版）。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 GeneralAgent
    """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.PLAN:      AgentType.PLANNER,
        IntentCategory.SCHEDULE:  AgentType.PLANNER,
        IntentCategory.TRACK:     AgentType.TRACKER,
        IntentCategory.RISK:      AgentType.TRACKER,
        IntentCategory.REPORT:    AgentType.REPORTER,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        # 其余意图（GREETING/QUERY/OTHER）→ GENERAL（默认）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "doubao-1-5-pro-32k-250115",
        skill_manager: Optional[Any] = None,
        api_mode: str = "chat",
        use_llm_intent: bool = True,
        fast_model: Optional[str] = None,
    ):
        client = LLMClient(api_key=api_key, base_url=base_url, model=model, api_mode=api_mode)

        self._intent_recognizer = IntentRecognizer(
            api_key=api_key, base_url=base_url, model=model,
            api_mode=api_mode, use_llm=use_llm_intent,
        )
        self._skill_manager = skill_manager
        self._deep_model = model
        self._fast_model = fast_model or model

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL:   [GeneralAgent(client, model, skill_manager)],
            AgentType.PLANNER:   [PlannerAgent(client, model, skill_manager)],
            AgentType.TRACKER:   [TrackerAgent(client, model, skill_manager)],
            AgentType.REPORTER:  [ReporterAgent(client, model, skill_manager)],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.urgency = intent_result.urgency

        # 外部信息类问题自动联网搜索（结果作为背景资料交给 Agent）
        if self._needs_search(req):
            search_bg = await self._search_for(req)
            if search_bg:
                req.context = (req.context + "\n\n" + search_bg) if req.context else search_bg

        # 并行协作已禁用：始终由"意图识别 → 单一最贴切 Agent"作答，
        # 避免多 Agent 同时回答导致响应慢、答案冗长。
        # collaboration = self._collaboration_targets(req)
        # if len(collaboration) > 1:
        #     return await self.run_parallel(req, collaboration)

        # 2. 路由：选择 Agent 类型
        agent_type = self._route(req.intent, req.urgency)

        # 3. 选择模型（自动复杂度评估 / 用户手动覆盖）
        effective_model = await self._select_model(req)

        # 4. 执行（含降级）
        response = await self._execute(req, agent_type, effective_model)

        # 5. 升级检查
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent == IntentCategory.ESCALATION:
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处创建工单、通知项目负责人

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            model=effective_model,
        )

    async def run_stream(self, req: Request) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理请求：先做意图识别和路由，然后逐块 yield 回复文本，
        最后 yield 一个包含元数据的 done 消息。

        Yields:
            {"type": "token", "content": "..."} — 回复文本片段
            {"type": "done", "intent": ..., "agent_type": ..., "latency_ms": ...,
             "escalated": ..., "full_response": "..."} — 结束元数据
        """
        t0 = time.monotonic()

        # 1. 意图识别
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent = intent_result.intent
            req.urgency = intent_result.urgency

        # 外部信息类问题自动联网搜索（结果作为背景资料交给 Agent）
        if self._needs_search(req):
            search_bg = await self._search_for(req)
            if search_bg:
                req.context = (req.context + "\n\n" + search_bg) if req.context else search_bg

        # 2. 路由（并行协作已禁用，始终单一 Agent 作答）
        # collaboration = self._collaboration_targets(req)
        # if len(collaboration) > 1:
        #     result = await self.run_parallel(req, collaboration)
        #     yield {"type": "token", "content": result.response}
        #     yield {
        #         "type": "done",
        #         "intent": result.intent.value if result.intent else "other",
        #         "agent_type": result.agent_type.value,
        #         "latency_ms": result.latency_ms,
        #         "escalated": result.escalated,
        #         "full_response": result.response,
        #     }
        #     return

        agent_type = self._route(req.intent, req.urgency)

        # 3. 选 Agent 并流式执行
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            yield {"type": "token", "content": "服务暂时不可用，请稍后重试。"}
            yield {"type": "done", "intent": "other", "agent_type": "general",
                   "latency_ms": (time.monotonic() - t0) * 1000, "escalated": False,
                   "full_response": "服务暂时不可用，请稍后重试。"}
            return

        # 收集完整回复，供后续升级检查和记忆使用
        full_parts: List[str] = []
        effective_model = await self._select_model(req)
        try:
            async for chunk in agent.handle_stream(req, model=effective_model):
                full_parts.append(chunk)
                yield {"type": "token", "content": chunk}
        except Exception as ex:
            logger.error(f"流式执行异常: {ex}")
            # 专属 Agent 流式失败，降级到 GeneralAgent
            if agent_type != AgentType.GENERAL:
                fallback = self._best_agent(AgentType.GENERAL)
                if fallback:
                    logger.warning(f"{agent_type.value} 流式失败，降级到 GeneralAgent")
                    full_parts = []
                    async for chunk in fallback.handle_stream(req, model=effective_model):
                        full_parts.append(chunk)
                        yield {"type": "token", "content": chunk}
                    agent_type = AgentType.GENERAL

        full_response = "".join(full_parts)

        # 4. 升级检查
        escalated = False
        if req.urgency == UrgencyLevel.CRITICAL or req.intent == IntentCategory.ESCALATION:
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")

        # 5. 结束元数据
        yield {
            "type": "done",
            "intent": req.intent.value if req.intent else "other",
            "agent_type": agent_type.value,
            "latency_ms": (time.monotonic() - t0) * 1000,
            "escalated": escalated,
            "model": effective_model,
            "full_response": full_response,
        }

    async def run_parallel(self, req: Request, agent_types: List[AgentType]) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及进度和风险）。
        """
        t0 = time.monotonic()
        effective_model = await self._select_model(req)
        tasks = [self._execute(req, at, effective_model) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并：拼接所有成功响应
        parts = []
        for r in responses:
            if isinstance(r, AgentResponse) and r.success:
                parts.append(f"[{r.agent_type.value}]\n{r.content}")

        combined = "\n\n".join(parts) if parts else "抱歉，所有 Agent 均处理失败。"
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=agent_types[0],
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            model=effective_model,
        )

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    # 简单意图直接用快速模型，跳过复杂度评估
    _FAST_INTENTS = {IntentCategory.GREETING, IntentCategory.OTHER}

    async def _select_model(self, req: Request) -> str:
        """
        智能选择模型：
        1. 用户手动指定了模型 → 直接使用
        2. 问候/闲聊类 → 直接用快速模型
        3. 其他问题 → 用快速模型做一次复杂度评估（~2秒），
           简单问题用快速模型回答，复杂问题用深度思考模型
        """
        if req.model_override:
            return req.model_override

        if req.intent in self._FAST_INTENTS:
            return self._fast_model

        # 复杂度评估：用快速模型快速判断，max_tokens 极小所以很快
        try:
            judge = await self._pool[AgentType.GENERAL][0]._client.chat(
                [{"role": "user", "content": (
                    "判断这个项目管理问题的复杂度，只回答 simple 或 complex，不要解释：\n"
                    + req.message[:500]
                )}],
                max_tokens=10,
                temperature=0.0,
                model=self._fast_model,
            )
            judge_lower = judge.strip().lower()
            if "complex" in judge_lower:
                logger.info(f"复杂度评估: complex → 使用深度思考模型 | 问题: {req.message[:50]}")
                return self._deep_model
            else:
                logger.info(f"复杂度评估: simple → 使用快速模型 | 问题: {req.message[:50]}")
                return self._fast_model
        except Exception as ex:
            logger.warning(f"复杂度评估失败，回退到深度思考模型: {ex}")
            return self._deep_model

    def _needs_search(self, req: Request) -> bool:
        """判断是否为外部信息查询：是则自动联网搜索。

        只对信息查询类意图（QUERY/OTHER）触发；内部项目管理场景不搜索；
        命中外部实体特征词（公司/科技/最新/行情等）才触发。
        """
        if req.intent not in (IntentCategory.QUERY, IntentCategory.OTHER):
            return False
        m = (req.message or "").strip()
        if len(m) < 4:
            return False
        internal_kws = ["帮我规划", "帮我跟踪", "生成周报", "我的项目", "我们项目",
                        "任务表", "进度如何", "风险应对", "排期", "里程碑", "复盘",
                        "计划", "安排任务", "拆解"]
        if any(k in m for k in internal_kws):
            return False
        external_kws = ["科技", "公司", "集团", "有限", "怎么样", "是什么", "有什么项目",
                        "最新", "新闻", "行情", "事件", "股票", "股价", "价格", "多少钱",
                        "谁", "哪里", "什么时候", "为什么", "产品", "业务", "发布",
                        "融资", "投资", "收购", "成立"]
        return any(k in m for k in external_kws)

    async def _search_for(self, req: Request) -> str:
        """执行联网搜索，返回格式化背景文本（失败/无结果返回空串，不阻断主流程）。

        搜索词策略：先用快速模型提取准确关键词（如"禾迈电力电子"→"禾迈股份"），
        再以清洗后的原词兜底，两者合并去重，提升中文实体命中率。
        """
        try:
            from mcp.web_search import web_search, clean_query, format_results
            keywords = await self._extract_search_keywords(req)
            fallback = clean_query(req.message)
            if fallback and fallback not in keywords:
                keywords.append(fallback)
            if not keywords:
                return ""

            junk_kws = ["汉典", "汉语国学", "汉语查", "新华字典", "字典", "的拼音",
                        "的意思", "部首", "释义", "读音", "笔画", "组词", "词典"]

            def _is_junk(r: Dict[str, str]) -> bool:
                blob = (r.get("title", "") + r.get("snippet", ""))
                return any(k in blob for k in junk_kws)

            all_results: List[Dict[str, Any]] = []
            seen: set = set()
            used_kws: List[str] = []
            for kw in keywords[:3]:
                used_kws.append(kw)
                for r in await web_search(kw, max_results=4):
                    url = r.get("url", "")
                    if url and url not in seen and not _is_junk(r):
                        seen.add(url)
                        all_results.append(r)
            # 结果全被过滤（如"禾迈电力电子"命中字典）→ 尝试简称变体（+股份/+集团）
            if not all_results:
                for kw in used_kws:
                    for suffix in ("股份", "集团", "科技"):
                        if kw.endswith(suffix) or len(kw) > 8:
                            continue
                        variant = kw + suffix
                        for r in await web_search(variant, max_results=3):
                            url = r.get("url", "")
                            if url and url not in seen and not _is_junk(r):
                                seen.add(url)
                                all_results.append(r)
            return format_results(all_results)
        except Exception as ex:
            logger.warning(f"联网搜索失败: {ex}")
            return ""

    async def _extract_search_keywords(self, req: Request) -> List[str]:
        """用快速模型从问题中提取 1-2 个搜索关键词（公司名/专有名词），失败返回空。"""
        try:
            agent = self._best_agent(AgentType.GENERAL)
            if agent is None:
                return []
            prompt = (
                "你是搜索关键词专家。从下面问题中提取 1-2 个最适合联网搜索的实体名称关键词。\n"
                "规则：\n"
                "1. 若是公司/机构，必须用其正式简称或股票简称（例如：杭州禾迈电力电子股份有限公司 → 禾迈股份；"
                "字节跳动有限公司 → 字节跳动）\n"
                "2. 排除'项目、公司、集团、科技、怎么样、是什么、有哪些'等通用词\n"
                "3. 只输出关键词本身，多个用英文逗号分隔，不要任何解释\n"
                "问题：\n" + req.message[:200]
            )
            text = await agent._client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.0,
                model=self._fast_model,
            )
            kws = [k.strip() for k in re.split(r"[,，]", text) if k.strip()]
            return kws[:2]
        except Exception as ex:
            logger.warning(f"搜索关键词提取失败: {ex}")
            return []

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"进度滞后且有风险"需要 tracker 同时处理，或"规划+排期"需要 planner。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        planning_kws = ["规划", "拆解", "分解", "里程碑", "计划", "排期", "安排", "工期", "plan", "schedule", "wbs"]
        tracking_kws = ["进度", "状态", "完成", "逾期", "延迟", "滞后", "跟踪", "进展", "progress", "status", "update", "风险", "阻塞", "卡住", "问题"]
        reporting_kws = ["周报", "月报", "汇报", "总结", "摘要", "复盘", "站会", "report", "summary", "recap"]

        if req.intent == IntentCategory.PLAN or req.intent == IntentCategory.SCHEDULE or any(kw in msg for kw in planning_kws):
            targets.append(AgentType.PLANNER)
        if req.intent in (IntentCategory.TRACK, IntentCategory.RISK) or any(kw in msg for kw in tracking_kws):
            targets.append(AgentType.TRACKER)
        if req.intent == IntentCategory.REPORT or any(kw in msg for kw in reporting_kws):
            targets.append(AgentType.REPORTER)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType, model: str) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req, model=model)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type != AgentType.GENERAL:
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req, model=model)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 tracker_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
