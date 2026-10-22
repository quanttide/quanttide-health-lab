"""模型调用与输出解析。

凭据解析顺序：环境变量 → ~/.bashrc 里的 export 行 → 报错。
（Hermes 的凭据池把 deepseek 记为 `env:DEEPSEEK_API_KEY`，而该变量只在交互式
shell 的 .bashrc 里 export，非交互进程读不到，所以这里自己解析一次。）
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("FATIGUE_API_BASE", "https://api.deepseek.com/v1")
MODEL = os.environ.get("FATIGUE_MODEL", "deepseek-chat")
KEY_ENV = "DEEPSEEK_API_KEY"


def resolve_key(key_env: str = KEY_ENV) -> str:
    key = os.environ.get(key_env)
    if key:
        return key.strip()
    for name in (".bashrc", ".profile", ".zshrc"):
        p = Path.home() / name
        if not p.exists():
            continue
        m = re.search(rf'^\s*export\s+{re.escape(key_env)}=["\']?([^"\'#\n]+)', p.read_text(errors="ignore"), re.M)
        if m:
            return m.group(1).strip()
    raise SystemExit(f"没有可用的 API key：环境变量与 ~/.bashrc 里都没有 {key_env}")


def chat(messages: list[dict], key: str, temperature: float = 0.0, max_tokens: int = 3000,
         retries: int = 3) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{API_BASE}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.load(r)
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(last) from e
        except Exception as e:  # 网络抖动
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"调用失败（重试 {retries} 次）：{last}")


KEY_ALIASES = {
    "signal": ("signal", "kind", "type", "类型", "信号"),
    "fatigue": ("fatigue", "疲惫", "是否疲劳", "is_fatigue", "疲劳"),
    "intensity": ("intensity", "强度", "等级"),
    "sources": ("sources", "来源", "来源短语", "疲劳来源"),
    "quotes": ("quotes", "evidence", "原句", "证据", "quote"),
    "implicit": ("implicit", "隐式", "隐含"),
}

SIGNAL_WORDS = {
    "fatigue": ("fatigue", "疲劳", "累", "疲", "心力", "耗竭"),
    "friction": ("friction", "摩擦", "烦", "返工", "失控", "不顺"),
}

TRUE_WORDS = {"true", "是", "yes", "1", "y", "有", "命中", "正确"}
FALSE_WORDS = {"false", "否", "no", "0", "n", "无", "未命中", "不"}


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in TRUE_WORDS:
            return True
        if s in FALSE_WORDS:
            return False
    return False


def _pick(obj: dict, field: str):
    for alias in KEY_ALIASES[field]:
        if alias in obj:
            return obj[alias]
    return None


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = re.split(r"[；;\n]", v)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v)]


def parse_items(text: str, expected_ids: list[str]) -> tuple[dict, list[str]]:
    """宽进严出：从模型输出里取出逐条结果。返回 ({id: item}, 失败说明)。"""
    out: dict[str, dict] = {}
    fails: list[str] = []
    blocks = _json_blocks(text)
    if not blocks:
        return {}, ["输出里没有可解析的 JSON"]
    items: list = []
    for block in blocks:
        if isinstance(block, list):
            items.extend(block)
        elif isinstance(block, dict):
            inner = block.get("results") or block.get("items")
            if isinstance(inner, list):
                items.extend(inner)
            else:
                items.append(block)
    for obj in items:
        if not isinstance(obj, dict):
            continue
        mid = None
        for alias in ("id", "msg_id", "消息id", "编号"):
            if alias in obj:
                mid = str(obj[alias]).strip()
                break
        if mid is None or mid not in expected_ids:
            continue
        raw_signal = _pick(obj, "signal")
        signal = ""
        if isinstance(raw_signal, str):
            s = raw_signal.strip().lower()
            for name, words in SIGNAL_WORDS.items():
                if any(w == s or w in s for w in words):
                    signal = name
                    break
        if not signal and _to_bool(_pick(obj, "fatigue")):
            signal = "fatigue"  # 兼容旧口径
        quotes = _as_list(_pick(obj, "quotes"))
        raw_sources = _pick(obj, "sources")
        sources: list[dict] = []
        for item in (raw_sources or []) if isinstance(raw_sources, list) else ([raw_sources] if raw_sources else []):
            if isinstance(item, dict):
                label = ""
                for k in ("label", "source", "来源", "name", "短语", "text"):
                    if item.get(k):
                        label = str(item[k]).strip()
                        break
                quote = ""
                for k in ("quote", "evidence", "原句", "证据", "text"):
                    if item.get(k):
                        quote = str(item[k]).strip()
                        break
            else:
                label, quote = str(item).strip(), ""
            if not label:
                continue
            if not quote:
                quote = quotes[0] if quotes else ""
            sources.append({"label": label, "quote": quote})
        # 没有原文依据的判定不算命中——这是硬口径
        if signal and not any(s["quote"] for s in sources):
            signal = ""
        if not signal:
            sources = []
        intensity = _pick(obj, "intensity")
        try:
            intensity = max(1, min(5, int(float(str(intensity)))))
        except (TypeError, ValueError):
            intensity = 3
        out[mid] = {
            "signal": signal,
            "fatigue": signal == "fatigue",
            "intensity": intensity if signal else 0,
            "sources": [s["label"] for s in sources],
            "source_quotes": sources,
            "quotes": quotes if signal else [],
            "implicit": _to_bool(_pick(obj, "implicit")) if signal else False,
        }
    missing = [i for i in expected_ids if i not in out]
    if missing:
        fails.append(f"未返回 {len(missing)} 条：{', '.join(missing[:5])}")
    return out, fails


def _json_blocks(text: str) -> list:
    """取出文本里所有完整的 JSON 块（数组或对象）。

    模型常见两种偏差：两个对象/数组连写、或先给一段话再给 JSON。整段 json.loads 会炸，
    所以逐个位置试 raw_decode，成功就跳到该块结尾继续找——不只看第一个，
    否则后半批的 id 会被判成「未返回」。
    """
    dec = json.JSONDecoder()
    blocks = []
    i = 0
    n = len(text)
    while i < n:
        start = min([p for p in (text.find("[", i), text.find("{", i)) if p != -1], default=-1)
        if start == -1:
            break
        try:
            obj, end = dec.raw_decode(text[start:])
        except json.JSONDecodeError:
            i = start + 1
            continue
        blocks.append(obj)
        i = start + end
    return blocks
