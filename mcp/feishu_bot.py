"""
EchoMind-PM — 飞书长连接机器人

通过飞书 WebSocket 长连接接收消息，调用 Agent 处理后回复到飞书。
不需要公网 IP/域名，本地直接运行即可接收飞书消息。

架构：
  飞书用户发消息 → 飞书 WebSocket 推送 → 事件回调 → 后台事件循环调用 Agent → 飞书 API 回复

使用方式：
  bot = FeishuBot(orchestrator=orchestrator, feishu_client=feishu_client)
  bot.start()  # 阻塞运行，Ctrl+C 停止
"""
import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import lark_oapi as lark
from lark_oapi.ws import Client as WSClient

from mcp.feishu_client import FeishuClient

logger = logging.getLogger(__name__)


class FeishuBot:
    """飞书长连接机器人，接收消息并调用 Agent 回复。"""

    def __init__(
        self,
        orchestrator: Any,
        feishu_client: Optional[FeishuClient] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
    ):
        self._orchestrator = orchestrator
        self._feishu = feishu_client or FeishuClient()
        self._app_id = app_id or os.getenv("LARK_APP_ID", "")
        self._app_secret = app_secret or os.getenv("LARK_APP_SECRET", "")

        # 后台事件循环，用于调用异步 Agent
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._ws_client: Optional[WSClient] = None
        self._running = False

        # 会话管理：user_id -> session_id，保持对话上下文
        self._sessions: Dict[str, str] = {}

    def _start_event_loop(self):
        """在后台线程启动 asyncio 事件循环。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout: float = 300.0):
        """在后台事件循环中运行协程，同步等待结果。"""
        if self._loop is None:
            raise RuntimeError("事件循环未启动")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _get_or_create_session(self, user_id: str) -> str:
        """获取或创建用户会话 ID。"""
        if user_id not in self._sessions:
            self._sessions[user_id] = f"feishu_{user_id}_{int(time.time())}"
        return self._sessions[user_id]

    # ── 消息处理 ──────────────────────────────────────────────────────────────

    def _handle_message_event(self, event_data: Dict[str, Any]):
        """处理飞书消息接收事件（同步回调，由 WSClient 线程调用）。"""
        try:
            # 解析事件数据
            event = event_data.get("event", event_data)
            message = event.get("message", {})
            sender = event.get("sender", {})

            msg_type = message.get("message_type", "")
            chat_type = message.get("chat_type", "p2p")
            sender_id = sender.get("sender_id", {})
            open_id = sender_id.get("open_id", "")
            chat_id = message.get("chat_id", "")

            if not open_id:
                logger.warning("无法获取发送者 open_id，忽略消息")
                return

            # 只处理文本消息
            if msg_type != "text":
                self._feishu.send_text_message(
                    open_id, "目前只支持文本消息哦，有什么项目管理问题可以文字告诉我～"
                )
                return

            # 解析文本内容
            content = message.get("content", "{}")
            try:
                text = json.loads(content).get("text", "")
            except (json.JSONDecodeError, AttributeError):
                text = str(content)

            # 群聊中去掉 @机器人 的提及
            if chat_type == "group":
                # 去掉 @_user_1 等提及标记
                import re
                text = re.sub(r"@_user_\d+", "", text).strip()
                if not text:
                    return  # 群聊中只 @ 没说话，忽略

            if not text:
                return

            logger.info(f"收到飞书消息 | 用户:{open_id} | 群聊:{chat_type == 'group'} | 内容:{text[:50]}")

            # 调用 Agent 处理
            session_id = self._get_or_create_session(open_id)
            response_text = self._process_with_agent(text, open_id, session_id)

            # 回复到飞书
            if response_text:
                receive_id = chat_id if chat_type == "group" else open_id
                receive_id_type = "chat_id" if chat_type == "group" else "open_id"
                self._feishu.send_text_message(receive_id, response_text, receive_id_type)

        except Exception as ex:
            logger.error(f"处理飞书消息异常: {ex}", exc_info=True)
            try:
                # 尝试通知用户
                event = event_data.get("event", event_data)
                sender = event.get("sender", {})
                open_id = sender.get("sender_id", {}).get("open_id", "")
                if open_id:
                    self._feishu.send_text_message(open_id, "抱歉，处理你的请求时出现了问题，请稍后重试。")
            except Exception:
                pass

    def _process_with_agent(self, message: str, user_id: str, session_id: str) -> str:
        """调用 Agent 编排器处理消息，返回回复文本。"""
        from agents.agent_orchestrator import Request as OrcReq

        try:
            orch_req = OrcReq(
                message=message,
                user_id=user_id,
                conv_id=session_id,
            )
            result = self._run_async(self._orchestrator.run(orch_req), timeout=300)
            return result.response or "抱歉，我没有理解你的问题，可以再说详细一点吗？"
        except Exception as ex:
            logger.error(f"Agent 处理异常: {ex}", exc_info=True)
            return "抱歉，处理你的请求时出现了问题，请稍后重试。"

    # ── 启动/停止 ─────────────────────────────────────────────────────────────

    def start(self):
        """启动飞书机器人（阻塞运行）。"""
        if not self._app_id or not self._app_secret:
            logger.error("飞书 App ID / App Secret 未配置，无法启动机器人")
            return

        if not self._feishu.available:
            logger.error("飞书客户端初始化失败，无法启动机器人")
            return

        # 启动后台事件循环
        self._loop_thread = threading.Thread(target=self._start_event_loop, daemon=True)
        self._loop_thread.start()
        # 等待事件循环就绪
        for _ in range(50):
            if self._loop is not None and self._loop.is_running():
                break
            time.sleep(0.1)

        # 注册事件处理器
        event_handler = lark.EventDispatcherHandler.builder() \
            .register_p2_im_message_receive_v1(self._handle_message_event) \
            .build()

        # 创建 WebSocket 长连接客户端
        self._ws_client = WSClient(
            app_id=self._app_id,
            app_secret=self._app_secret,
            event_handler=event_handler,
            auto_reconnect=True,
        )

        self._running = True
        logger.info("=" * 60)
        logger.info("EchoMind-PM 飞书机器人已启动（长连接模式）")
        logger.info(f"App ID: {self._app_id}")
        logger.info("在飞书中搜索机器人名称即可开始对话")
        logger.info("按 Ctrl+C 停止")
        logger.info("=" * 60)

        try:
            self._ws_client.start()
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭...")
        finally:
            self.stop()

    def stop(self):
        """停止机器人。"""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("飞书机器人已停止")
