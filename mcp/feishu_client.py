"""
EchoMind-PM — 飞书 API 客户端封装

基于 lark-oapi SDK，封装项目管理机器人需要的飞书操作：
- 发送消息（私聊/群聊）
- 多维表格：查询/创建/更新记录
- 获取用户信息

所有方法均为同步调用，供飞书机器人事件回调中使用。
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
)
from lark_oapi.api.bitable.v1 import (
    AppTableRecord,
    CreateAppTableRecordRequest,
    ListAppTableRecordRequest,
    UpdateAppTableRecordRequest,
)

logger = logging.getLogger(__name__)


class FeishuClient:
    """飞书 API 客户端，封装消息发送和多维表格操作。"""

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        base_token: Optional[str] = None,
        table_id: Optional[str] = None,
    ):
        self.app_id = app_id or os.getenv("LARK_APP_ID", "")
        self.app_secret = app_secret or os.getenv("LARK_APP_SECRET", "")
        self.base_token = base_token or os.getenv("LARK_BASE_TOKEN", "")
        self.table_id = table_id or os.getenv("LARK_TABLE_ID", "")

        if not self.app_id or not self.app_secret:
            logger.warning("飞书 App ID / App Secret 未配置，飞书功能不可用")
            self._client = None
        else:
            self._client = lark.Client.builder() \
                .app_id(self.app_id) \
                .app_secret(self.app_secret) \
                .log_level(lark.LogLevel.INFO) \
                .build()

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── 消息发送 ──────────────────────────────────────────────────────────────

    def send_text_message(self, receive_id: str, text: str, receive_id_type: str = "open_id") -> bool:
        """
        发送文本消息给用户或群聊。

        receive_id: 用户 open_id 或群聊 chat_id
        receive_id_type: "open_id"（私聊）或 "chat_id"（群聊）
        """
        if not self.available:
            logger.warning("飞书客户端未初始化，无法发送消息")
            return False

        try:
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type("text")
                    .content(json.dumps({"text": text}, ensure_ascii=False))
                    .build()
                ) \
                .build()

            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(f"发送消息失败: code={response.code}, msg={response.msg}")
                return False
            logger.info(f"消息已发送给 {receive_id}")
            return True
        except Exception as ex:
            logger.error(f"发送消息异常: {ex}")
            return False

    # ── 多维表格操作 ──────────────────────────────────────────────────────────

    def bitable_list_records(
        self,
        page_size: int = 20,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询多维表格记录。"""
        if not self.available or not self.base_token or not self.table_id:
            logger.warning("多维表格未配置，无法查询记录")
            return []

        try:
            builder = ListAppTableRecordRequest.builder() \
                .app_token(self.base_token) \
                .table_id(self.table_id) \
                .page_size(page_size)
            if filter_expr:
                builder.filter(filter_expr)

            request = builder.build()
            response = self._client.bitable.v1.app_table_record.list(request)
            if not response.success():
                logger.error(f"查询记录失败: code={response.code}, msg={response.msg}")
                return []

            records = []
            if response.data and response.data.items:
                for item in response.data.items:
                    records.append({
                        "record_id": item.record_id,
                        "fields": item.fields,
                    })
            return records
        except Exception as ex:
            logger.error(f"查询记录异常: {ex}")
            return []

    def bitable_create_record(self, fields: Dict[str, Any]) -> Optional[str]:
        """
        创建多维表格记录，返回 record_id。

        fields: 字段名到值的映射，例如 {"任务名称": "需求评审", "状态": "待办"}
        """
        if not self.available or not self.base_token or not self.table_id:
            logger.warning("多维表格未配置，无法创建记录")
            return None

        try:
            request = CreateAppTableRecordRequest.builder() \
                .app_token(self.base_token) \
                .table_id(self.table_id) \
                .request_body(AppTableRecord.builder().fields(fields).build()) \
                .build()

            response = self._client.bitable.v1.app_table_record.create(request)
            if not response.success():
                logger.error(f"创建记录失败: code={response.code}, msg={response.msg}")
                return None

            record_id = response.data.record.record_id if response.data and response.data.record else None
            logger.info(f"已创建多维表格记录: {record_id}")
            return record_id
        except Exception as ex:
            logger.error(f"创建记录异常: {ex}")
            return None

    def bitable_update_record(self, record_id: str, fields: Dict[str, Any]) -> bool:
        """更新多维表格记录。"""
        if not self.available or not self.base_token or not self.table_id:
            logger.warning("多维表格未配置，无法更新记录")
            return False

        try:
            request = UpdateAppTableRecordRequest.builder() \
                .app_token(self.base_token) \
                .table_id(self.table_id) \
                .record_id(record_id) \
                .request_body(AppTableRecord.builder().fields(fields).build()) \
                .build()

            response = self._client.bitable.v1.app_table_record.update(request)
            if not response.success():
                logger.error(f"更新记录失败: code={response.code}, msg={response.msg}")
                return False
            logger.info(f"已更新多维表格记录: {record_id}")
            return True
        except Exception as ex:
            logger.error(f"更新记录异常: {ex}")
            return False

    # ── 项目管理快捷操作 ──────────────────────────────────────────────────────

    def create_task(
        self,
        task_name: str,
        project: str = "",
        assignee: str = "",
        status: str = "待办",
        priority: str = "中",
        due_date: str = "",
        progress: int = 0,
        remark: str = "",
    ) -> Optional[str]:
        """快捷创建项目任务到多维表格。"""
        fields: Dict[str, Any] = {"任务名称": task_name}
        if project:
            fields["项目"] = project
        if assignee:
            fields["负责人"] = assignee
        if status:
            fields["状态"] = status
        if priority:
            fields["优先级"] = priority
        if due_date:
            fields["截止日期"] = due_date
        if progress:
            fields["进度"] = progress
        if remark:
            fields["备注"] = remark

        return self.bitable_create_record(fields)

    def list_tasks(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """快捷查询任务列表，可按状态筛选。"""
        filter_expr = None
        if status_filter:
            filter_expr = f'CurrentValue.[状态]="{status_filter}"'
        return self.bitable_list_records(page_size=50, filter_expr=filter_expr)
