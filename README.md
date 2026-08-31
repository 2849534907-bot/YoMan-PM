# EchoMind-PM 项目管理 AI 助手

基于多 Agent 架构的项目管理 AI 助手，支持自然语言完成任务规划、进度跟踪、风险识别、汇报生成等工作。已接入飞书机器人，团队成员可直接在飞书内对话管理项目。

> 基于开源 [EchoMind](https://github.com/) 多 Agent 框架二次开发，业务领域从智能客服改造为项目管理。

## 功能特性

### 多 Agent 专业分工

| Agent | 职责 | 触发场景 |
|---|---|---|
| 规划专家 | WBS 拆解、里程碑设定、排期、依赖关系 | "帮我规划一个项目" |
| 跟踪专家 | 任务状态查询、逾期预警、风险阻塞识别 | "项目进度怎么样" |
| 汇报专家 | 周报、项目摘要、站会要点、复盘 | "生成一份周报" |
| 通用助手 | 概念解答、需求引导、转介 | "你好"/"什么是关键路径" |

### 智能模型切换

- **简单查询**自动调用轻量模型（首字响应约 7 秒）
- **复杂规划**自动切换深度推理模型，保证回答质量
- 支持手动切换：自动 / mini / evolving

### 飞书机器人集成

- WebSocket 长连接接收消息，无需公网 IP
- 私聊 / 群聊 @机器人 均可使用
- 多维表格自动创建 / 更新任务记录
- 团队成员零门槛使用

### 其他能力

- 流式输出（逐字显示，感知响应更快）
- 对话记忆（多轮上下文）
- 知识库 RAG 检索
- 性能监控与 Agent 路由优化
- Web 端 + 飞书端双入口

## 技术架构

```
用户输入 → 意图识别 → Agent 路由 → 模型选择 → LLM 生成 → 工具执行 → 回复
                ↓            ↓           ↓
          关键词/LLM    4类专业Agent  轻量/深度双模型
```

- **后端**：Python + FastAPI
- **大模型**：豆包（火山方舟 Ark，OpenAI 兼容接口）
- **飞书集成**：lark-oapi SDK + WebSocket 长连接
- **前端**：原生 HTML/CSS/JS（单文件，无构建依赖）
- **数据存储**：本地 JSON（可扩展飞书多维表格）

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv-pm
.\.venv-pm\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入：
- `DOUBAO_API_KEY`：火山方舟 API Key
- `LARK_APP_ID` / `LARK_APP_SECRET`：飞书自建应用凭证（可选，启用飞书机器人时需要）

### 3. 启动服务

**仅启动 API 服务（Web 端对话）：**

```bash
.\.venv-pm\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8123
```

浏览器打开 `web/index.html` 即可对话。

**同时启动 API + 飞书机器人：**

```bash
.\.venv-pm\Scripts\python.exe run_all.py
```

## 项目结构

```
EchoMind-PM/
├── agents/              # Agent 编排器（路由、执行、降级）
├── api/                 # FastAPI 服务（对话、流式、监控）
├── core/                # 核心层（LLM 适配、意图识别、Skill 加载）
├── mcp/                 # 工具层（飞书客户端、飞书机器人、知识库、项目存储）
├── memory/              # 对话记忆管理
├── monitor/             # 性能监控
├── skills/              # 业务 Skill 规范（规划/跟踪/汇报/通用）
├── web/                 # Web 聊天界面
├── data/                # 数据存储（git 忽略）
├── run_all.py           # 一键启动（API + 飞书机器人）
├── run_bot.py           # 仅启动飞书机器人
└── requirements.txt     # 依赖清单
```

## 飞书机器人配置

1. 飞书开放平台创建自建应用，开通「机器人」能力
2. 事件订阅选择「长连接」模式，添加 `im.message.receive_v1` 事件
3. 开通权限：`im:message`、`im:message:send_as_bot`、`bitable:app`
4. 可用范围设为「全部成员」，创建版本并发布
5. `.env` 填入 `LARK_APP_ID` 和 `LARK_APP_SECRET`
6. 运行 `run_all.py`，在飞书搜索机器人名称开始对话

## License

MIT
