# -*- coding: utf-8 -*-
"""
EchoMind-PM — 内置联网搜索模块

在豆包原生联网搜索插件未开通的情况下，提供零依赖的网页搜索能力：
- 使用必应（cn.bing.com）公开搜索页，抓取并解析标题/摘要/链接
- 供 Orchestrator 在"信息查询类"问题上自动调用，把结果作为背景交给 LLM

说明：解析基于公开搜索页 HTML，属于尽力而为的实现；
若后续在火山方舟控制台开通"联网内容插件"，可平滑切换到原生 web_search。
"""
import asyncio
import html as html_lib
import logging
import re
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

BING_URL = "https://cn.bing.com/search"
BING_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml",
}

_ALGO_RE = re.compile(r'<li class="b_algo".*?</li>', re.S)
_H2_RE = re.compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_P_RE = re.compile(r'<p[^>]*>(.*?)</p>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(s: str) -> str:
    return html_lib.unescape(_TAG_RE.sub("", s)).strip()


def _parse_bing(html_text: str, max_results: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for block in _ALGO_RE.findall(html_text)[:max_results]:
        m = _H2_RE.search(block)
        if not m:
            continue
        url, title_html = m.group(1), m.group(2)
        title = _strip_tags(title_html)
        p = _P_RE.search(block)
        snippet = _strip_tags(p.group(1)) if p else ""
        if url.startswith("http"):
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


async def web_search(query: str, max_results: int = 5, timeout: float = 12.0) -> List[Dict[str, str]]:
    """
    搜索并返回结构化结果列表：[{"title", "url", "snippet"}, ...]
    失败时返回空列表并记录日志（不抛异常，避免阻断主流程）。
    """
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=BING_HEADERS, follow_redirects=True) as client:
            resp = await client.get(BING_URL, params={"q": query, "mkt": "zh-CN", "setlang": "zh-CN"})
            resp.raise_for_status()
            results = _parse_bing(resp.text, max_results)
            if not results:
                logger.warning(f"必应搜索未解析到结果: {query}（页面可能被重定向到验证页）")
            else:
                logger.info(f"联网搜索成功: {query} → {len(results)} 条")
            return results
    except Exception as ex:
        logger.warning(f"联网搜索失败: {query} | {ex}")
        return []


# 口语/语气/分析性词汇清洗表：尽量只保留专有名词，提升必应命中率
STOP_WORDS = [
    "帮我", "请", "一下", "简单", "分析一下", "分析", "请问", "怎么样", "如何",
    "介绍", "了解", "关于", "搜索", "搜一下", "查一下", "呢", "吗", "吧", "啊",
    "的", "有什么", "啥", "哪些", "项目", "公司", "集团", "有限公司", "动态",
    "信息", "情况", "新闻", "最新", "报告", "资料", "内容", "数据", "业务",
    "产品", "简介", "相关", "看看", "给我", "一下", "具体", "详细", "说说",
]


def clean_query(q: str) -> str:
    """把口语化问题清洗成核心搜索词（专有名词优先）。"""
    for w in STOP_WORDS:
        q = q.replace(w, "")
    q = re.sub(r"\s+", " ", q).strip()
    return q or ""


def format_results(results: List[Dict[str, str]]) -> str:
    """把搜索结果格式化为 LLM 可读的背景文本。"""
    if not results:
        return ""
    lines = ["【网上参考信息（仅供参考，回答请以用户提供的资料和项目管理专业判断为主）】"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   链接: {r['url']}\n   摘要: {r['snippet'][:300]}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 本地自测
    async def main():
        q = "鸿擎科技 有什么项目"
        res = await web_search(q)
        print(f"搜索「{q}」共 {len(res)} 条：")
        for r in res:
            print("-", r["title"], "|", r["url"])
            print("  ", r["snippet"][:120])

    asyncio.run(main())
