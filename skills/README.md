# EchoMind-PM Skills 文档

EchoMind-PM 启动时会从 `ECHOMIND_SKILLS_DIR` 读取 Skills，并在匹配用户请求时注入到对应 Agent 的 system prompt。Skills 适合维护项目管理规范、任务拆解规则、进度跟踪 SOP、汇报模板、升级规则和禁止事项。

当前内置四类 Skills：

```text
skills/planning/SKILL.md   # 项目规划：WBS 拆解、里程碑、排期
skills/tracking/SKILL.md   # 进度跟踪：状态、逾期、风险、阻塞
skills/reporting/SKILL.md  # 汇报总结：周报、摘要、站会、复盘
skills/general/SKILL.md    # 通用助手：接待、澄清、分流、升级
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill_name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: 进度跟踪规范
description: 适用于 TrackerAgent 的进度查询和风险处理规范
keywords: 进度,状态,逾期,风险,阻塞,progress,status,risk
agents: tracker
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称，会出现在注入给模型的 prompt 中。
- `description`：简短说明，方便 `/skills` 接口排查。
- `keywords`：触发关键词，用户消息命中后才注入；多个关键词用英文逗号或中文逗号分隔均可。
- `agents`：适用 Agent，可填 `general`、`planner`、`tracker`、`reporter`，多个值用逗号分隔。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 重要规则放在文档前半部分，因为过长内容会按 prompt 预算截断。
- 一类 Skill 只描述一类职责，不要把规划、跟踪、汇报规则混在一个文件里。
- 必须包含"角色定位""处理流程""升级条件""禁止事项"等稳定章节。
- 对无法保证的事项使用保守措辞，例如"通常""建议""待确认"。
- 对需要负责人审批、跨团队协调的场景要明确写出升级条件。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看加载结果和解析错误：

```bash
curl http://localhost:8000/skills
```
