"""从 Hermes 会话库抽出用户消息，作为分析语料。

只取 role='user'——分析的是「说话人自己说了什么状态」，不是 AI 的回应。
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

DEFAULT_DB = Path.home() / ".hermes" / "state.db"
CST = timezone(timedelta(hours=8))


def extract(db_path: Path = DEFAULT_DB, out: Path | None = None, since: str | None = None,
            dedupe: bool = True) -> list[dict]:
    if not db_path.exists():
        raise SystemExit(f"会话库不存在：{db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql = (
        "select m.id, m.session_id, m.timestamp, m.content, s.source "
        "from messages m left join sessions s on s.id = m.session_id "
        "where m.role = 'user' and m.content is not null and length(trim(m.content)) > 0"
    )
    rows = con.execute(sql).fetchall()
    con.close()

    records = []
    for mid, sid, ts, content, source in rows:
        try:
            ts_f = float(ts)
        except (TypeError, ValueError):
            continue
        iso = datetime.fromtimestamp(ts_f, CST).isoformat(timespec="seconds")
        if since and iso < since:
            continue
        records.append(
            {
                "msg_id": str(mid),
                "session_id": sid or "",
                "ts": ts_f,
                "iso": iso,
                "date": iso[:10],
                "hour": int(iso[11:13]),
                "source": source or "",
                "chars": len(content),
                "content": content,
            }
        )
    records.sort(key=lambda r: r["ts"])
    # 同会话内完全相同的消息只算一条：Hermes 的消息表会留下被吸收/压缩的副本，
    # 不去重的话同一个「累」会被算两三次，命中数虚高。保留最早那条，记下重复次数。
    if dedupe:
        seen: dict[tuple[str, str], dict] = {}
        deduped = []
        for r in records:
            key = (r["session_id"], " ".join(r["content"].split()))
            if key in seen:
                seen[key]["dup_count"] = seen[key].get("dup_count", 1) + 1
                continue
            r["dup_count"] = 1
            seen[key] = r
            deduped.append(r)
        records = deduped
    if out:
        write_jsonl(out, records)
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
