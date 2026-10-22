"""输出前的脱敏：报告里引用的原文可能带上密钥、token、邮箱等。

证据句必须能在原文里核对，但账号类串一律替换掉——报告是给人读的，不是二次泄露。
"""

from __future__ import annotations

import re

PATTERNS = [
    (re.compile(r"\b(?:sk|gsk|ghp|gho|xoxb|xoxp|AKIA)[-_A-Za-z0-9]{8,}"), "<已隐去密钥>"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer <已隐去>"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "<已隐去邮箱>"),
    (re.compile(r"\b(?:api[_-]?key|token|secret|password)\s*[=:]\s*\S+", re.I), "<已隐去凭据>"),
    (re.compile(r"\b[0-9a-f]{32,}\b"), "<已隐去长串>"),
]


def scrub(text: str) -> str:
    if not text:
        return ""
    out = text
    for pat, repl in PATTERNS:
        out = pat.sub(repl, out)
    return out
