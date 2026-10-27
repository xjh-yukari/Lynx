"""本地 HTML 文本提取，不执行脚本，不使用远程正文代理。"""

import re

from bs4 import BeautifulSoup

from lynx_agent.web.errors import WebError
from lynx_agent.web.http import WebDocument
from lynx_agent.web.models import FetchResponse


def clean_text(value: str) -> str:
    return " ".join(BeautifulSoup(value, "html.parser").get_text(" ", strip=True).split())


def extract(document: WebDocument, original_url: str, max_chars: int) -> FetchResponse:
    title = ""
    published = None
    if document.content_type in {"text/html", "application/xhtml+xml"}:
        soup = BeautifulSoup(document.body, "html.parser", from_encoding=document.charset)
        title = soup.title.get_text(" ", strip=True)[:300] if soup.title else ""
        date = soup.select_one('meta[property="article:published_time"], meta[name="date"], meta[itemprop="datePublished"]')
        if date and isinstance(date.get("content"), str):
            published = date["content"][:100]
        if soup.select_one("form[action*='anomaly'], #challenge-form, input[name='captcha']"):
            raise WebError("网页要求验证码或人工验证，未绕过访问限制")
        for node in soup.select("script,style,noscript,template,nav,footer,header,form,button,iframe,svg,head"):
            node.decompose()
        root = soup.select_one("article") or soup.select_one("main") or soup.body or soup
        text = root.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        try:
            text = document.body.decode(document.charset or "utf-8", errors="replace")
        except LookupError:
            text = document.body.decode("utf-8", errors="replace")
    text = text.strip()
    if not text:
        raise WebError("未提取到正文，页面可能需要脚本、登录或特殊解析")
    return FetchResponse(url=original_url, final_url=document.url, title=title, content=text[:max_chars],
                         content_type=document.content_type, published_at=published, truncated=len(text) > max_chars)
