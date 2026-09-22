# -*- coding: utf-8 -*-
"""修复 agent_orchestrator.py 中 handle_stream 的多模态代码块缩进。"""
import io

P = r"C:\Users\Administrator\Desktop\Agent开发\EchoMind-PM\agents\agent_orchestrator.py"
with io.open(P, encoding="utf-8") as f:
    t = f.read()
t = t.replace("\r\n", "\n")

old = """            if req.images:
            content = [{"type": "text", "text": _clean(req.message)}]
            for _url in req.images:
                content.append({"type": "image_url", "image_url": {"url": _url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": _clean(req.message)})
"""
new = """            if req.images:
                content = [{"type": "text", "text": _clean(req.message)}]
                for _url in req.images:
                    content.append({"type": "image_url", "image_url": {"url": _url}})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": _clean(req.message)})
"""
cnt = t.count(old)
print("匹配次数:", cnt)
if cnt == 1:
    t = t.replace(old, new)
    t = t.replace("\n", "\r\n")
    with io.open(P, "w", encoding="utf-8", newline="") as f:
        f.write(t)
    print("已修复")
else:
    print("未精确匹配，需人工检查")
