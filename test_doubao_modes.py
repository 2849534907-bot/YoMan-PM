"""
EchoMind-PM — 豆包双接口测试脚本

对照火山方舟官方示例，分别测试 Chat Completions 和 Responses 两种接口模式。
用法：
  1. 在 .env 中配置 DOUBAO_API_KEY（或 ARK_API_KEY / OPENAI_API_KEY）
  2. 运行：.venv-pm\\Scripts\\python.exe test_doubao_modes.py
  3. 也可指定模式：.venv-pm\\Scripts\\python.exe test_doubao_modes.py chat
                                          .venv-pm\\Scripts\\python.exe test_doubao_modes.py responses
"""
import asyncio
import os
import sys
import pathlib

# 加载 .env
from dotenv import load_dotenv
load_dotenv()

# 项目根目录加入 path
_ROOT = str(pathlib.Path(__file__).parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.llm import LLMClient, llm_env_config


async def test_chat_mode(cfg: dict) -> None:
    """Chat Completions 模式（/chat/completions），对应官方示例一。"""
    print("=" * 60)
    print("【模式 1】Chat Completions（/chat/completions）")
    print("=" * 60)
    client = LLMClient(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_mode="chat",
    )
    try:
        text = await client.chat(
            messages=[{"role": "user", "content": "用一句话介绍你自己，说明你能做什么。"}],
            max_tokens=1024,
            temperature=0.3,
        )
        print(f"模型回复:\n{text}\n")
    except Exception as ex:
        print(f"Chat 模式调用失败: {type(ex).__name__}: {ex}\n")


async def test_responses_mode(cfg: dict) -> None:
    """Responses 模式（/responses），对应官方示例二。"""
    print("=" * 60)
    print("【模式 2】Responses（/responses）")
    print("=" * 60)
    client = LLMClient(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_mode="responses",
    )
    try:
        text = await client.chat(
            messages=[{"role": "user", "content": "用一句话介绍你自己，说明你能做什么。"}],
            max_tokens=1024,
            temperature=0.3,
        )
        print(f"模型回复:\n{text}\n")
    except Exception as ex:
        print(f"Responses 模式调用失败: {type(ex).__name__}: {ex}\n")


async def main():
    cfg = llm_env_config()
    print(f"模型: {cfg['model']}")
    print(f"接口地址: {cfg['base_url']}")
    print(f"API Key: {cfg['api_key'][:8]}..." if cfg['api_key'] else "API Key: 未配置！")
    print()

    if not cfg["api_key"]:
        print("错误：未配置 API Key。请在 .env 中设置 DOUBAO_API_KEY。")
        return

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if mode in ("chat", "both"):
        await test_chat_mode(cfg)
    if mode in ("responses", "both"):
        await test_responses_mode(cfg)

    print("=" * 60)
    print("测试完成。两种模式返回内容应一致，只是调用接口不同。")
    print("在 .env 中设置 DOUBAO_API_MODE=chat 或 responses 可切换项目全局模式。")


if __name__ == "__main__":
    asyncio.run(main())
