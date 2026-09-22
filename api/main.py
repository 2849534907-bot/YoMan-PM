"""
EchoMind-PM 项目管理 AI Agent — FastAPI 入口

由 EchoMind 智能客服系统改造而来，业务领域替换为项目管理。
启动时打印项目徽标。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
LLM 使用 OpenAI 兼容接口（OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL）。
"""
import asyncio
import json
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
# 这一行必须在所有项目内部 import 之前执行
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.llm import llm_env_config  # noqa: E402

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ╔══════════════════════════════╗
    ║   EchoMind-PM  v2.0          ║
    ║   项目管理 AI Agent          ║
    ╚══════════════════════════════╝
    Doubao 驱动 · 规划 / 跟踪 / 汇报
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_kb_use_chroma = False   # 知识库是否使用 ChromaDB 向量检索（False 时为关键词降级模式）


def _llm_cfg() -> Dict[str, Any]:
    return llm_env_config()


def _build_feishu_store():
    """根据环境变量构建飞书多维表格存储（未配置时返回 None）。"""
    app_id = os.getenv("LARK_APP_ID", "").strip()
    app_secret = os.getenv("LARK_APP_SECRET", "").strip()
    base_token = os.getenv("LARK_BASE_TOKEN", "").strip()
    table_id = os.getenv("LARK_TABLE_ID", "").strip()
    if not (app_id and app_secret and base_token and table_id):
        logger.info("未配置飞书多维表格（LARK_APP_ID/LARK_APP_SECRET/LARK_BASE_TOKEN/LARK_TABLE_ID），使用本地存储")
        return None
    from mcp.project_store import FeishuBaseStore
    logger.info("飞书多维表格同步已启用")
    return FeishuBaseStore(
        app_id=app_id,
        app_secret=app_secret,
        base_token=base_token,
        table_id=table_id,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager

    cfg = _llm_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    use_llm_intent = os.getenv("INTENT_USE_LLM", "true").lower() in {"1", "true", "yes"}

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        api_mode=cfg.get("api_mode", "chat"),
        use_llm=use_llm_intent,
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        api_mode=cfg.get("api_mode", "chat"),
        use_llm_intent=use_llm_intent,
        fast_model=cfg.get("fast_model") or None,
        skill_manager=_skill_manager,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        api_mode=cfg.get("api_mode", "chat"),
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        api_mode=cfg.get("api_mode", "chat"),
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    global _kb_use_chroma
    _kb_use_chroma = getattr(kb, "_use_chroma", False)
    logger.info(f"知识库已加载: {kb.doc_count} 个文档片段（向量检索: {'是' if _kb_use_chroma else '否，关键词降级模式'}）")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"项目数据暂时不可用，未能完成对“{query}”的检索。请稍后重试，或转项目负责人确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))

    # 项目管理数据服务（本地 JSON + 可选飞书多维表格同步）
    from mcp.project_store import FeishuBaseStore, ProjectService

    _project_service = ProjectService(
        store_path=os.getenv("PROJECT_STORE_PATH", "./data/projects.json"),
        feishu=_build_feishu_store(),
        auto_sync=os.getenv("ENABLE_FEISHU_SYNC", "false").lower() in {"1", "true", "yes"},
    )

    _tool_manager.register(Tool(
        name="project_create",
        description="创建项目（含描述、里程碑）",
        handler=_project_service.handle_create_project,
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "milestones": {"type": "array"},
            },
            "required": ["name"],
        },
    ))
    _tool_manager.register(Tool(
        name="project_list",
        description="列出所有项目",
        handler=_project_service.handle_list_projects,
        schema={"type": "object", "properties": {}},
    ))
    _tool_manager.register(Tool(
        name="task_add",
        description="向项目添加任务（name 必填，可含 owner/status/progress/priority/deadline/deliverable）",
        handler=_project_service.handle_add_task,
        schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "name": {"type": "string"},
                "owner": {"type": "string"},
                "status": {"type": "string"},
                "progress": {"type": "integer"},
                "priority": {"type": "string"},
                "deadline": {"type": "string"},
                "deliverable": {"type": "string"},
            },
            "required": ["project_id", "name"],
        },
    ))
    _tool_manager.register(Tool(
        name="task_update",
        description="更新任务（状态/进度/负责人/截止日期等）",
        handler=_project_service.handle_update_task,
        schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "task_id": {"type": "string"},
                "name": {"type": "string"},
                "owner": {"type": "string"},
                "status": {"type": "string"},
                "progress": {"type": "integer"},
                "priority": {"type": "string"},
                "deadline": {"type": "string"},
                "deliverable": {"type": "string"},
            },
            "required": ["project_id", "task_id"],
        },
    ))
    _tool_manager.register(Tool(
        name="task_list",
        description="查询项目任务列表（可按状态筛选）",
        handler=_project_service.handle_list_tasks,
        schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["project_id"],
        },
    ))
    _tool_manager.register(Tool(
        name="feishu_sync",
        description="把项目任务同步到飞书多维表格",
        handler=_project_service.handle_sync_feishu,
        schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
    ))
    _tool_manager.register(Tool(
        name="feishu_pull",
        description="从飞书多维表格拉取任务到本地",
        handler=_project_service.handle_pull_feishu,
        schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
    ))

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        api_mode=cfg.get("api_mode", "chat"),
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    logger.info("EchoMind-PM 已就绪")
    yield

    await _monitor.stop()
    logger.info("EchoMind-PM 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoMind-PM 项目管理助手",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 网页端静态入口（浏览器访问 http://127.0.0.1:8123 直接打开聊天界面）─────────
_WEB_DIR = pathlib.Path(__file__).resolve().parent.parent / "web"

@app.get("/", include_in_schema=False)
async def index():
    resp = FileResponse(str(_WEB_DIR / "index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatFile(BaseModel):
    """上传文件解析结果：kind=text 为文本内容，kind=image 为 data URL。"""
    filename:    str
    kind:        str = "text"
    content:     str = ""
    note:        str = ""


class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None
    model:       Optional[str] = None  # 手动指定模型（mini/evolving），不传则自动选择
    files:       Optional[List[ChatFile]] = None  # 上传文件（解析后内容）


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    agent_type:  str
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    model:       str = ""  # 实际使用的模型


def _apply_files(req: ChatRequest) -> tuple:
    """拆分用户消息与文件内容。

    返回 (用户原话, 图片 data URL 列表, 文件背景文本)：
    - 文件内容作为"背景信息"交给 LLM 阅读，不混入用户消息，
      避免污染意图识别与复杂度评估（防止误判、误切深度模型、误触发多 Agent）。
    """
    if not req.files:
        return req.message, None, ""
    images: List[str] = []
    file_parts = []
    for f in req.files:
        if f.kind == "image" and f.content:
            images.append(f.content)
        elif f.content:
            file_parts.append(f"《{f.filename}》\n{f.content[:50000]}")
    background = ""
    if file_parts:
        background = ("用户上传了以下文件，请结合文件内容回答问题：\n\n"
                      + "\n\n".join(file_parts))
    return req.message, (images or None), background


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/upload-file", tags=["文件"])
async def upload_file(file: UploadFile = File(...)):
    """
    上传并解析文件，返回结构化内容。
    - 文本/PDF/Word/Excel/PPT/压缩包：返回提取的文本内容
    - 图片：返回 base64 data URL（供视觉模型识别）
    - 视频/音频：返回说明文字（暂不支持内容解析）
    """
    from mcp.file_parser import parse_file
    data = await file.read()
    result = parse_file(file.filename or "unnamed", data)
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())

    # 1. 读取记忆上下文
    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

    # 2. 构建编排请求（含对话历史，用于意图识别上下文）
    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    knowledge_text, knowledge_used = await _build_knowledge_context(req.message)
    user_message, images, file_background = _apply_files(req)
    context_parts = ["【对话历史（上下文参考，不是用户主动上传的资料）】\n" + mem_ctx.to_prompt_text()]
    if file_background:
        context_parts.append("【用户上传的资料（回答以此为主要依据）】\n" + file_background)
    if knowledge_text:
        context_parts.append(knowledge_text)
    full_context = "\n\n".join(part for part in context_parts if part)

    orch_req = OrcReq(
        message=user_message,
        user_id=req.user_id,
        conv_id=conv_id,
        context=full_context,
        history=history,
        model_override=req.model,
        images=images,
    )

    # 3. 执行
    result = await _orchestrator.run(orch_req)

    # 4. 写入记忆
    attach_note = f"\n[已附加 {len(req.files)} 个文件]" if req.files else ""
    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message + attach_note)
    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)

    # 5. 异步更新用户画像（不阻塞响应）
    asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

    return ChatResponse(
        conv_id=conv_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        agent_type=result.agent_type.value,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used=knowledge_used,
        model=result.model,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口。逐块返回回复文本（SSE 风格的换行分隔 JSON），
    最后一条为包含元数据的 done 消息。

    响应格式（每行一个 JSON）：
      {"type": "token", "content": "你"}
      {"type": "token", "content": "好"}
      {"type": "done", "intent": "greeting", "agent_type": "general",
       "latency_ms": 15000, "escalated": false, "conv_id": "..."}
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())

    # 1. 读取记忆上下文
    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    knowledge_text, knowledge_used = await _build_knowledge_context(req.message)
    user_message, images, file_background = _apply_files(req)
    context_parts = ["【对话历史（上下文参考，不是用户主动上传的资料）】\n" + mem_ctx.to_prompt_text()]
    if file_background:
        context_parts.append("【用户上传的资料（回答以此为主要依据）】\n" + file_background)
    if knowledge_text:
        context_parts.append(knowledge_text)
    full_context = "\n\n".join(part for part in context_parts if part)

    orch_req = OrcReq(
        message=user_message,
        user_id=req.user_id,
        conv_id=conv_id,
        context=full_context,
        history=history,
        model_override=req.model,
        images=images,
    )

    async def generate():
        full_response = ""
        meta = {}
        try:
            async for event in _orchestrator.run_stream(orch_req):
                if event.get("type") == "token":
                    content = event.get("content", "")
                    full_response += content
                    yield json.dumps({"type": "token", "content": content}, ensure_ascii=False) + "\n"
                elif event.get("type") == "done":
                    meta = event
        except Exception as ex:
            logger.error(f"流式对话异常: {ex}")
            yield json.dumps({"type": "error", "message": str(ex)}, ensure_ascii=False) + "\n"

        # 流式结束后写入记忆
        try:
            await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
            await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, full_response)
            asyncio.create_task(_memory.update_profile(req.user_id, conv_id))
        except Exception as ex:
            logger.warning(f"流式对话记忆写入失败: {ex}")

        # 发送 done 元数据
        yield json.dumps({
            "type": "done",
            "conv_id": conv_id,
            "intent": meta.get("intent", "other"),
            "agent_type": meta.get("agent_type", "general"),
            "latency_ms": round(meta.get("latency_ms", 0), 1),
            "escalated": meta.get("escalated", False),
            "model": meta.get("model", ""),
            "knowledge_used": knowledge_used,
        }, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


async def _build_knowledge_context(message: str, top_k: int = 3) -> tuple[str, bool]:
    """
    为 /chat 主链路构建 RAG 知识上下文。

    这里复用 MCPToolManager 的查询改写、并行召回、重排、fallback 能力。
    """
    if _tool_manager is None:
        return "", False
    if not _should_use_knowledge(message):
        return "", False
    try:
        result = await _tool_manager.search_with_rewrite("knowledge_search", message, top_k=top_k)
        if not result.success or not isinstance(result.data, list) or not result.data:
            return "", False

        parts = ["[知识库检索结果]"]
        used = False
        for i, item in enumerate(result.data[:top_k], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("content", "")).strip()
            score = item.get("score", "")
            if not content:
                continue
            used = True
            parts.append(f"{i}. 标题: {title}\n   相关度: {score}\n   内容: {content[:600]}")

        if not used:
            return "", False
        parts.append("请优先依据以上项目数据回答；如果信息不足，再结合通用项目管理能力说明。")
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False


def _should_use_knowledge(message: str) -> bool:
    """
    跳过纯寒暄，业务类问题才检索知识库，避免无关 RAG 干扰回复。
    注意：ChromaDB 未安装时知识库为关键词降级模式，查询改写会额外消耗一次 LLM 调用，
    对深度思考模型（如 doubao-seed-evolving）响应延迟影响较大，因此降级模式下跳过知识库。
    """
    # 降级模式（无 ChromaDB）下跳过知识库检索，避免额外 LLM 调用导致超时
    if not _kb_use_chroma:
        return False
    msg = (message or "").strip().lower()
    if not msg:
        return False
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}
    if msg in greetings:
        return False
    business_keywords = [
        "任务", "项目", "进度", "排期", "工期", "里程碑", "周报", "风险", "阻塞",
        "负责人", "需求", "交付", "验收", "拆分", "规划", "延期", "逾期",
        "task", "project", "progress", "schedule", "milestone", "report", "risk", "deadline",
    ]
    return len(msg) >= 4 or any(kw in msg for kw in business_keywords)


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "项目周报模板", "content": "本周完成 / 进行中 / 风险阻塞 / 下一步计划..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = kb.add_documents([{"title": d.title, "content": d.content} for d in body.documents])
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": kb.doc_count}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = kb.add_documents(docs)
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": kb.doc_count}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("EchoMind-PM CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _llm_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nEchoMind-PM [{result.agent_type.value}]: {result.response}\n")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
