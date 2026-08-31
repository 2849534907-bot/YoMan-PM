"""
EchoMind-PM 一键启动脚本

同时启动：
  1. API 服务（http://127.0.0.1:8123）—— 网页端对话接口
  2. 飞书机器人（长连接）—— 飞书内对话

用法：python run_all.py
按 Ctrl+C 停止全部服务。
"""
import multiprocessing
import os
import sys
import pathlib
import time

_ROOT = str(pathlib.Path(__file__).parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def run_api():
    """启动 API 服务。"""
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8123,
        log_level="info",
    )


def run_bot():
    """启动飞书机器人。"""
    from run_bot import main as bot_main
    bot_main()


def main():
    print("=" * 60)
    print("EchoMind-PM 项目管理 AI Agent")
    print("=" * 60)

    # 检查飞书配置
    from dotenv import load_dotenv
    load_dotenv()
    has_feishu = bool(os.getenv("LARK_APP_ID") and os.getenv("LARK_APP_SECRET"))

    processes = []

    # 启动 API 服务
    p_api = multiprocessing.Process(target=run_api, name="echomind-api", daemon=True)
    p_api.start()
    processes.append(p_api)
    print(f"[1/2] API 服务已启动: http://127.0.0.1:8123 (PID: {p_api.pid})")

    # 启动飞书机器人
    if has_feishu:
        time.sleep(3)  # 等 API 服务初始化
        p_bot = multiprocessing.Process(target=run_bot, name="echomind-feishu-bot", daemon=True)
        p_bot.start()
        processes.append(p_bot)
        print(f"[2/2] 飞书机器人已启动 (PID: {p_bot.pid})")
    else:
        print("[2/2] 飞书机器人未启动（未配置 LARK_APP_ID / LARK_APP_SECRET）")

    print("=" * 60)
    print("全部服务已启动，按 Ctrl+C 停止")
    print("=" * 60)

    try:
        while True:
            time.sleep(1)
            # 检查子进程是否存活
            for p in processes:
                if not p.is_alive():
                    print(f"[警告] 进程 {p.name} (PID: {p.pid}) 已退出")
    except KeyboardInterrupt:
        print("\n正在停止所有服务...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        print("所有服务已停止")


if __name__ == "__main__":
    main()
