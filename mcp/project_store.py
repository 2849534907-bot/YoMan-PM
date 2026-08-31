"""
EchoMind-PM — 项目数据存储层

用途：为项目管理 Agent 提供真实可读写的任务数据，而不是只靠对话。

架构：
  1. LocalProjectStore —— 本地 JSON 文件持久化（默认，开箱即用）
  2. FeishuBaseStore  —— 可选飞书多维表格同步（通过飞书开放平台 OpenAPI）
  3. ProjectService   —— 门面：Agent 统一通过它操作项目/任务，
                        配置飞书后每次写入会同步到多维表格，供用户在飞书里管理。

任务字段：
  id / name / owner / status / progress / priority / deadline / dependencies / deliverable
状态口径：未开始 / 进行中 / 已完成 / 已阻塞 / 已取消
"""
import asyncio
import json
import logging
import pathlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Task:
    id:          str
    name:        str
    owner:       str = "待定"
    status:      str = "未开始"      # 未开始/进行中/已完成/已阻塞/已取消
    progress:    int = 0             # 0-100
    priority:    str = "中"          # 高/中/低
    deadline:    str = ""            # YYYY-MM-DD
    dependencies: List[str] = field(default_factory=list)
    deliverable: str = ""
    updated_at:  str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Project:
    id:          str
    name:        str
    description: str = ""
    milestones:  List[str] = field(default_factory=list)
    created_at:  str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    tasks:       List[Task] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "milestones": self.milestones,
            "created_at": self.created_at,
            "tasks": [t.to_dict() for t in self.tasks],
        }


# ── 本地 JSON 存储 ────────────────────────────────────────────────────────────

class LocalProjectStore:
    """基于 JSON 文件的持久化存储。"""

    def __init__(self, path: str = "./data/projects.json"):
        self._path = pathlib.Path(path).expanduser().resolve()
        self._lock = asyncio.Lock()
        self._data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
            except Exception as ex:
                logger.warning(f"项目数据文件读取失败，使用空数据: {ex}")
        return {}

    async def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 项目 ──────────────────────────────────────────────────────────────

    async def create_project(self, name: str, description: str = "", milestones: Optional[List[str]] = None) -> Project:
        async with self._lock:
            pid = "p_" + uuid.uuid4().hex[:8]
            proj = Project(id=pid, name=name, description=description, milestones=list(milestones or []))
            self._data[pid] = proj.to_dict()
            await self._save()
            return proj

    async def get_project(self, project_id: str) -> Optional[Project]:
        raw = self._data.get(project_id)
        if not raw:
            return None
        return self._project_from_dict(raw)

    async def list_projects(self) -> List[Project]:
        return [self._project_from_dict(raw) for raw in self._data.values()]

    # ── 任务 ──────────────────────────────────────────────────────────────

    async def add_task(self, project_id: str, name: str, **kwargs) -> Optional[Task]:
        async with self._lock:
            raw = self._data.get(project_id)
            if raw is None:
                return None
            task = Task(
                id="t_" + uuid.uuid4().hex[:8],
                name=name,
                owner=kwargs.get("owner") or "待定",
                status=kwargs.get("status") or "未开始",
                progress=int(kwargs.get("progress") or 0),
                priority=kwargs.get("priority") or "中",
                deadline=kwargs.get("deadline") or "",
                dependencies=list(kwargs.get("dependencies") or []),
                deliverable=kwargs.get("deliverable") or "",
            )
            raw["tasks"].append(task.to_dict())
            await self._save()
            return task

    async def update_task(self, project_id: str, task_id: str, **fields) -> Optional[Task]:
        async with self._lock:
            raw = self._data.get(project_id)
            if raw is None:
                return None
            for t in raw["tasks"]:
                if t["id"] == task_id:
                    allowed = {"owner", "status", "progress", "priority", "deadline", "dependencies", "deliverable", "name"}
                    for k, v in fields.items():
                        if k in allowed:
                            t[k] = v
                    t["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    await self._save()
                    return Task(**t)
            return None

    async def list_tasks(self, project_id: str, status: Optional[str] = None) -> List[Task]:
        raw = self._data.get(project_id)
        if raw is None:
            return []
        tasks = [Task(**t) for t in raw.get("tasks", [])]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks

    # ── 辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _project_from_dict(raw: Dict[str, Any]) -> Project:
        return Project(
            id=raw["id"],
            name=raw["name"],
            description=raw.get("description", ""),
            milestones=list(raw.get("milestones", [])),
            created_at=raw.get("created_at", ""),
            tasks=[Task(**t) for t in raw.get("tasks", [])],
        )


# ── 飞书多维表格存储（可选）──────────────────────────────────────────────────

class FeishuBaseStore:
    """
    飞书多维表格同步（通过飞书开放平台 OpenAPI）。

    需要配置（环境变量）：
      LARK_APP_ID / LARK_APP_SECRET  飞书自建应用的凭证
      LARK_BASE_TOKEN                多维表格 app_token
      LARK_TABLE_ID                  数据表 table_id
      LARK_OWNER_FIELD / LARK_STATUS_FIELD 等字段名可自定义（默认英文名）

    任务记录字段（多维表格列名，可配置）：
      id / 任务名称 / 负责人 / 状态 / 进度 / 优先级 / 截止日期 / 交付物
    """

    API_BASE = "https://open.feishu.cn/open-apis"

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        base_token: str,
        table_id: str,
        field_names: Optional[Dict[str, str]] = None,
    ):
        self._app_id = app_id
        self._app_secret = app_secret
        self._base_token = base_token
        self._table_id = table_id
        self._fields = {
            "id": "id",
            "name": "任务名称",
            "owner": "负责人",
            "status": "状态",
            "progress": "进度",
            "priority": "优先级",
            "deadline": "截止日期",
            "deliverable": "交付物",
        }
        if field_names:
            self._fields.update(field_names)
        self._token: Optional[str] = None
        self._token_expire: float = 0.0

    async def _get_token(self) -> str:
        if self._token and time.time() < self._token_expire - 60:
            return self._token
        import httpx
        async with httpx.AsyncClient(timeout=15) as c:
            resp = await c.post(
                f"{self.API_BASE}/auth/v3/tenant_access_token/internal",
                json={"app_id": self._app_id, "app_secret": self._app_secret},
            )
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"获取飞书 token 失败: {data.get('msg')}")
            self._token = data["tenant_access_token"]
            self._token_expire = time.time() + int(data.get("expire", 7200))
            return self._token

    async def _request(self, method: str, path: str, **kw) -> Dict[str, Any]:
        import httpx
        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=20) as c:
            resp = await c.request(method, f"{self.API_BASE}{path}", headers=headers, **kw)
            data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"飞书 API 错误: {data.get('msg')} ({data.get('code')})")
        return data

    async def upsert_tasks(self, tasks: List[Task]) -> Dict[str, int]:
        """把任务列表写入多维表格（按 id 匹配更新，不存在则新增）。"""
        created = updated = 0
        existing = await self._list_records()
        id_to_record = {r["id"]: r["record_id"] for r in existing}

        for task in tasks:
            fields = {
                self._fields["id"]: task.id,
                self._fields["name"]: task.name,
                self._fields["owner"]: task.owner,
                self._fields["status"]: task.status,
                self._fields["progress"]: task.progress,
                self._fields["priority"]: task.priority,
                self._fields["deliverable"]: task.deliverable,
            }
            if task.deadline:
                fields[self._fields["deadline"]] = int(
                    datetime.strptime(task.deadline, "%Y-%m-%d").timestamp() * 1000
                )
            if task.id in id_to_record:
                await self._request(
                    "PUT",
                    f"/bitable/v1/apps/{self._base_token}/tables/{self._table_id}/records/{id_to_record[task.id]}",
                    json={"fields": fields},
                )
                updated += 1
            else:
                await self._request(
                    "POST",
                    f"/bitable/v1/apps/{self._base_token}/tables/{self._table_id}/records",
                    json={"fields": fields},
                )
                created += 1
        return {"created": created, "updated": updated}

    async def list_records(self) -> List[Dict[str, Any]]:
        """读取多维表格中的任务记录，映射回 Task 结构。"""
        records = await self._list_records()
        result = []
        for r in records:
            f = r["fields"]
            try:
                progress = int(f.get(self._fields["progress"], 0) or 0)
            except (TypeError, ValueError):
                progress = 0
            deadline = ""
            d = f.get(self._fields["deadline"])
            if isinstance(d, int):
                deadline = datetime.fromtimestamp(d / 1000).strftime("%Y-%m-%d")
            result.append({
                "id": f.get(self._fields["id"], ""),
                "name": f.get(self._fields["name"], ""),
                "owner": f.get(self._fields["owner"], ""),
                "status": f.get(self._fields["status"], ""),
                "progress": progress,
                "priority": f.get(self._fields["priority"], ""),
                "deadline": deadline,
                "deliverable": f.get(self._fields["deliverable"], ""),
            })
        return result

    async def _list_records(self) -> List[Dict[str, Any]]:
        items, page_token = [], ""
        while True:
            path = f"/bitable/v1/apps/{self._base_token}/tables/{self._table_id}/records?page_size=200"
            if page_token:
                path += f"&page_token={page_token}"
            data = await self._request("GET", path)
            items.extend(data["data"]["items"])
            if not data["data"].get("has_more"):
                break
            page_token = data["data"].get("page_token", "")
        return items


# ── 门面：统一入口 ────────────────────────────────────────────────────────────

class ProjectService:
    """
    项目管理数据服务门面。

    - 本地 JSON 始终作为操作层（快速、可靠）。
    - 配置飞书后，写操作同步到多维表格（ENABLE_FEISHU_SYNC=true 时）。
    - 提供 MCP 工具 handler，供 Agent 调用。
    """

    def __init__(
        self,
        store_path: str = "./data/projects.json",
        feishu: Optional[FeishuBaseStore] = None,
        auto_sync: bool = False,
    ):
        self.local = LocalProjectStore(store_path)
        self.feishu = feishu
        self.auto_sync = auto_sync

    async def _maybe_sync(self, project_id: str) -> None:
        if self.feishu is None or not self.auto_sync:
            return
        try:
            proj = await self.local.get_project(project_id)
            if proj:
                await self.feishu.upsert_tasks(proj.tasks)
        except Exception as ex:
            logger.warning(f"飞书同步失败: {ex}")

    # ── 工具 handler ──────────────────────────────────────────────────────

    async def handle_create_project(self, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        proj = await self.local.create_project(
            name=params.get("name", ""),
            description=params.get("description", ""),
            milestones=params.get("milestones"),
        )
        return proj.to_dict()

    async def handle_list_projects(self, params: Dict[str, Any], context: Any) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in await self.local.list_projects()]

    async def handle_add_task(self, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        fields = {k: v for k, v in params.items() if k not in ("project_id", "name") and v is not None}
        task = await self.local.add_task(
            project_id=params["project_id"],
            name=params["name"],
            **fields,
        )
        if task is None:
            raise ValueError(f"项目不存在: {params['project_id']}")
        await self._maybe_sync(params["project_id"])
        return task.to_dict()

    async def handle_update_task(self, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        # 过滤掉 None 值，避免误覆盖已有字段
        fields = {k: v for k, v in params.items() if k not in ("project_id", "task_id") and v is not None}
        task = await self.local.update_task(
            project_id=params["project_id"],
            task_id=params["task_id"],
            **fields,
        )
        if task is None:
            raise ValueError(f"任务不存在: {params['task_id']}")
        await self._maybe_sync(params["project_id"])
        return task.to_dict()

    async def handle_list_tasks(self, params: Dict[str, Any], context: Any) -> List[Dict[str, Any]]:
        tasks = await self.local.list_tasks(
            project_id=params["project_id"],
            status=params.get("status"),
        )
        return [t.to_dict() for t in tasks]

    async def handle_sync_feishu(self, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        if self.feishu is None:
            return {"synced": False, "reason": "未配置飞书多维表格（LARK_APP_ID / LARK_BASE_TOKEN 等）"}
        proj = await self.local.get_project(params["project_id"])
        if proj is None:
            raise ValueError(f"项目不存在: {params['project_id']}")
        result = await self.feishu.upsert_tasks(proj.tasks)
        return {"synced": True, **result}

    async def handle_pull_feishu(self, params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """从飞书多维表格拉取任务到本地。"""
        if self.feishu is None:
            return {"synced": False, "reason": "未配置飞书多维表格"}
        records = await self.feishu.list_records()
        project_id = params["project_id"]
        # 简单合并：按 id 更新或新增
        count = 0
        for r in records:
            existing = await self.local.list_tasks(project_id)
            if any(t.id == r["id"] for t in existing):
                await self.local.update_task(project_id, r["id"], **{k: v for k, v in r.items() if k != "id"})
            else:
                await self.local.add_task(project_id, r["name"], **{k: v for k, v in r.items() if k != "id" and k != "name"})
            count += 1
        return {"pulled": count}
