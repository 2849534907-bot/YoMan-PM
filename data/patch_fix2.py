# -*- coding: utf-8 -*-
"""修复 Request dataclass 中 images 字段被拼到注释同一行的问题。"""
import io

P = r"C:\Users\Administrator\Desktop\Agent开发\EchoMind-PM\agents\agent_orchestrator.py"
with io.open(P, encoding="utf-8") as f:
    t = f.read()
t = t.replace("\r\n", "\n")

old = "    model_override: Optional[str] = None  # 用户手动指定模型，跳过自动选择    images: Optional[List[str]] = None  # 图片 data URL 列表（多模态输入）"
new = ("    model_override: Optional[str] = None  # 用户手动指定模型，跳过自动选择\n"
       "    images: Optional[List[str]] = None  # 图片 data URL 列表（多模态输入）")
cnt = t.count(old)
print("匹配次数:", cnt)
if cnt == 1:
    t = t.replace(old, new)
    t = t.replace("\n", "\r\n")
    with io.open(P, "w", encoding="utf-8", newline="") as f:
        f.write(t)
    print("已修复")
else:
    print("未匹配")
