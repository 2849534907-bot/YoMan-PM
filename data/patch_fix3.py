# -*- coding: utf-8 -*-
"""放宽 GeneralAgent 的 system prompt：允许解读上传的文件/图片。"""
import io

P = r"C:\Users\Administrator\Desktop\Agent开发\EchoMind-PM\agents\agent_orchestrator.py"
with io.open(P, encoding="utf-8") as f:
    t = f.read()
t = t.replace("\r\n", "\n")

old = """        "你是 EchoMind-PM 项目管理助手。友好、简洁地回答用户关于项目管理的问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业项目管理功能。"
        "可协助用户了解项目整体情况、解释项目管理概念。"
    )"""
new = """        "你是 EchoMind-PM 项目管理助手。友好、简洁地回答用户关于项目管理的问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业项目管理功能。"
        "可协助用户了解项目整体情况、解释项目管理概念。"
        "当用户上传文件或图片时，请先读取其中的内容，并尽量结合项目管理/工作场景给出解读、总结或建议；"
        "如果内容与工作完全无关，可简短说明后礼貌引导回项目管理话题。"
    )"""
cnt = t.count(old)
print("匹配次数:", cnt)
if cnt == 1:
    t = t.replace(old, new)
    t = t.replace("\n", "\r\n")
    with io.open(P, "w", encoding="utf-8", newline="") as f:
        f.write(t)
    print("已更新 GeneralAgent prompt")
else:
    print("未匹配，需人工检查")
