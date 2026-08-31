"""
EchoMind-PM 飞书机器人启动脚本（独立运行）

单独启动飞书长连接机器人，不需要 API 服务。
用法：python run_bot.py
"""
import os
import sys
import pathlib
import logging

# 将项目根目录加入 sys.path
_ROOT = str(pathlib.Path(__file__).parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_bot")


def main():
    from core.llm import llm_env_config
    from core.skill_loader import SkillManager
    from agents.agent_orchestrator import AgentOrchestrator
    from mcp.feishu_client import FeishuClient
    from mcp.feishu_bot import FeishuBot

    # 1. LLM 配置
    cfg = llm_env_config()
    if not cfg["api_key"]:
        logger.error("未配置 DOUBAO_API_KEY，请在 .env 中设置")
        sys.exit(1)

    use_llm_intent = os.getenv("INTENT_USE_LLM", "false").lower() in {"1", "true", "yes"}

    # 2. 加载 Skills
    skill_manager = SkillManager(
        root_dir=os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    logger.info(f"已加载 {len(skill_manager._skills)} 个 Skill")

    # 3. 初始化 Agent 编排器
    orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        api_mode=cfg.get("api_mode", "chat"),
        use_llm_intent=use_llm_intent,
        fast_model=cfg.get("fast_model") or None,
        skill_manager=skill_manager,
    )
    logger.info(f"Agent 编排器已初始化 | 深度模型: {cfg['model']} | 快速模型: {cfg.get('fast_model', '未配置')}")

    # 4. 初始化飞书客户端
    feishu_client = FeishuClient()
    if not feishu_client.available:
        logger.error("飞书 App ID / App Secret 未配置，无法启动机器人")
        sys.exit(1)
    logger.info("飞书客户端已初始化")

    # 5. 启动机器人
    bot = FeishuBot(
        orchestrator=orchestrator,
        feishu_client=feishu_client,
    )
    bot.start()


if __name__ == "__main__":
    main()
