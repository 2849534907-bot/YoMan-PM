@echo off
chcp 65001 >nul
title YoMan-PM 项目管理 Agent
cd /d "%~dp0"

echo ============================================
echo   YoMan-PM 项目管理 Agent
echo   项目管理智能助手 - 一键启动
echo ============================================
echo.

REM 检查虚拟环境
if not exist ".venv-pm\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境，请先执行：
    echo   python -m venv .venv-pm
    echo   .venv-pm\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM 检查服务是否已在运行（探测 8123 端口）
powershell -NoProfile -Command "try { (Invoke-WebRequest 'http://127.0.0.1:8123/health' -TimeoutSec 2 -UseBasicParsing).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1

if %errorlevel% equ 0 (
    echo [提示] 服务已在运行，直接打开界面...
    goto open_browser
)

echo [1/2] 正在启动服务（窗口已最小化）...
powershell -NoProfile -Command "Start-Process -FilePath '%~dp0.venv-pm\Scripts\python.exe' -ArgumentList 'run_all.py' -WorkingDirectory '%~dp0' -WindowStyle Minimized"

echo [2/2] 等待服务就绪...
set /a tries=0
:wait_loop
set /a tries+=1
powershell -NoProfile -Command "try { (Invoke-WebRequest 'http://127.0.0.1:8123/health' -TimeoutSec 2 -UseBasicParsing).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 goto open_browser
if %tries% geq 30 (
    echo [错误] 服务启动超时，请检查 data\svc_err.log
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait_loop

:open_browser
echo 服务已就绪，正在打开界面...
start "" "http://127.0.0.1:8123"
echo.
echo ============================================
echo   已启动！浏览器打开后即可对话。
echo   使用完毕：在最小化的服务窗口中按 Ctrl+C，
echo   或直接关闭该窗口即可停止。
echo ============================================
echo.
pause
