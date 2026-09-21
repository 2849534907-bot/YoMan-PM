@echo off
chcp 65001 >nul
title YoMan-PM 项目管理 Agent
cd /d "%~dp0"

echo ============================================
echo   YoMan-PM 项目管理 Agent 启动中...
echo ============================================
echo.

REM 检查虚拟环境
if not exist ".venv-pm\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境，请先执行：python -m venv .venv-pm
    echo        然后：.venv-pm\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [1/2] 正在启动 API 服务 (http://127.0.0.1:8123)...
echo [2/2] 正在启动飞书机器人（如已配置）...
echo.
echo 启动完成后：
echo   - 网页端：用浏览器打开 web\index.html
echo   - 飞书端：在飞书内搜索机器人开始对话
echo.
echo 按 Ctrl+C 停止全部服务
echo ============================================
echo.

.venv-pm\Scripts\python.exe run_all.py

pause
