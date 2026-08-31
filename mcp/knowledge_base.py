"""
EchoMind-PM — RAG 知识库

由 EchoMind 客服知识库改造而来，内容替换为项目管理知识。

功能：
  1. 文档导入：将文本切片后存储（有 ChromaDB 时向量化，否则存内存做关键词检索）
  2. 语义检索：根据 query 返回最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

降级策略：
  - 安装并可用 ChromaDB 时：使用向量检索（语义匹配，效果最好）
  - 未安装 ChromaDB / 缺少 C++ 构建工具时：退化为内存关键词检索，
    保证知识库链路仍然可用，不影响 Agent 主流程。
"""
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    RAG 知识库。

    有 ChromaDB 时使用向量语义检索；否则使用内存关键词检索降级。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        self._use_chroma = False
        # 降级模式的内存文档片段：{"id": ..., "title": ..., "content": ..., "chunk_index": ...}
        self._fallback_docs: List[Dict[str, Any]] = []
        self._fallback_count = 0

        if CHROMADB_AVAILABLE:
            try:
                self._client = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )
                self._client.heartbeat()
                self._use_chroma = True
                logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
            except Exception:
                try:
                    logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
                    self._client = chromadb.PersistentClient(
                        path=chroma_path,
                        settings=chromadb.Settings(anonymized_telemetry=False),
                    )
                    self._use_chroma = True
                except Exception as ex:
                    self._use_chroma = False
                    logger.warning(f"ChromaDB 本地模式初始化失败，降级为关键词检索: {ex}")
        else:
            logger.warning("未安装 chromadb，知识库使用关键词检索降级模式")

        if self._use_chroma:
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "EchoMind-PM 项目管理 RAG 知识库"},
            )
        else:
            self._collection = None

        # 如果知识库为空，导入默认文档
        if self.doc_count == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks)})

        if not ids:
            return 0

        if self._use_chroma:
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
        else:
            for doc_id, content, meta in zip(ids, docs, metas):
                self._fallback_docs.append({
                    "id": doc_id,
                    "title": meta["title"],
                    "content": content,
                    "chunk_index": meta["chunk_index"],
                })
            self._fallback_count += len(ids)

        logger.info(f"知识库导入 {len(ids)} 个文档片段")
        return len(ids)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """语义检索：根据 query 返回最相关的文档片段。"""
        if self._use_chroma:
            return self._search_chroma(query, top_k)
        return self._search_fallback(query, top_k)

    def _search_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # 距离转相似度
                    "chunk":    meta.get("chunk_index", 0),
                })
        return items

    def _search_fallback(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """降级模式：关键词重合度打分检索。"""
        q_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
        if not q_tokens:
            return []

        scored = []
        for doc in self._fallback_docs:
            text_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", doc["content"].lower()))
            overlap = len(q_tokens & text_tokens)
            if overlap > 0:
                score = overlap / len(q_tokens)  # 召回覆盖率
                scored.append({
                    "title":   doc["title"],
                    "content": doc["content"],
                    "score":   round(score, 4),
                    "chunk":   doc["chunk_index"],
                })

        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]

    @property
    def doc_count(self) -> int:
        if self._use_chroma:
            return self._collection.count()
        return self._fallback_count

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return self.search(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（项目管理场景常用知识）。"""
        default_docs = [
            {
                "title": "WBS 工作分解方法",
                "content": (
                    "WBS 工作分解结构是把项目交付物和项目工作细分为更小、更易管理的组成部分的方法。"
                    "拆解遵循 MECE 原则：子任务相互独立、完全穷尽，不重叠不遗漏。"
                    "每个叶子节点任务要可执行、可验收、可分配。"
                    "拆解粒度到某个负责人能在 3-5 天内完成为宜，不要过细。"
                    "典型层级：第一层为阶段（需求、设计、开发、测试、上线），"
                    "第二层为模块或工作包，第三层为具体任务（含负责人、工期、交付物）。"
                ),
            },
            {
                "title": "里程碑管理",
                "content": (
                    "里程碑是项目中的关键时间节点，代表阶段性成果的验收点。"
                    "每个里程碑应对应可演示或可验收的成果，例如需求冻结、Beta 上线。"
                    "里程碑之间应留出缓冲时间，避免把关键路径排满到没有回旋余地。"
                    "里程碑一旦确立，变更需要走变更审批流程。"
                    "设定里程碑时要有明确的完成标准，例如所有 P0 需求通过评审并冻结。"
                ),
            },
            {
                "title": "任务状态口径",
                "content": (
                    "项目任务状态统一使用五类：未开始、进行中、已完成、已阻塞、已取消。"
                    "逾期判定标准：任务截止日期已过且状态不是已完成。"
                    "区分风险与问题：风险是可能发生但尚未发生的事项，问题是已经发生并正在影响的事项。"
                    "完成百分比只采用用户提供或系统记录的数值，不自行推算。"
                ),
            },
            {
                "title": "周报结构模板",
                "content": (
                    "项目周报推荐使用五段结构：本周完成、进行中、风险与阻塞、下周计划、需要协调事项。"
                    "周报数据要基于任务实际状态，可追溯到具体任务，不编造进度。"
                    "突出需要负责人或上级决策的事项。"
                    "进度汇报优先使用任务、状态、进度、负责人、截止、备注的表格或列表结构。"
                    "逾期任务用醒目方式标注，并给出跟进建议。"
                ),
            },
            {
                "title": "关键路径法",
                "content": (
                    "关键路径（CPM）是项目中最长的依赖链，决定项目最短工期。"
                    "关键路径上的任何任务延期，都会直接推迟项目整体完成时间。"
                    "排期时优先保障关键路径任务，对高风险任务（新技术、外部依赖、多人协作）建议加缓冲。"
                    "如果用户要求压缩工期，说明压缩哪些任务会影响质量或依赖，而不是直接承诺。"
                ),
            },
            {
                "title": "项目管理风险识别",
                "content": (
                    "项目管理中常见的风险类别包括：进度风险（延期）、范围风险（需求蔓延）、"
                    "资源风险（人员不足）、技术风险（新技术不确定）、外部依赖风险（供应商、第三方）。"
                    "每个风险应说明：是什么、影响什么、谁负责、什么时候需要决策。"
                    "区分风险与问题，给出可操作建议而不是只说有风险请注意。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
