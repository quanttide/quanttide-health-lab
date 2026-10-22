"""责任归属：逐条判「这个消耗是谁造成的」。

为什么不用关键词规则：第一版就是关键词（`labels.actor_of`），实测 14 条里 9 条判错，
而且错得有方向——「我对你的工作方向表示怀疑」被判成「用户目标未定」，
「技术细节和数据太多」被判成「用户同时推进太多」。他抱怨我的时候用的正是「方向」「太多」
「不知道」这些词，词表天然把这些话翻译成他的属性。这叫责任外推，不是精度问题。

所以改成读原句判：给模型原话 + 该来源对应的证据句，让它判断这句话的消耗该记在谁头上，
并要求给出一条判据（reason）。规则表只留作离线兜底（模型没跑或解析失败时）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from .corpus import read_jsonl, write_jsonl

ACTORS = ["助手", "用户", "外部", "不明"]

SYSTEM = """你在给对话记录做「责任归属」标注。每条记录是一句原话（说话人是用户），以及从这句话里
提炼出来的一个「消耗来源」短语。你的任务只有一件：这个消耗，该记在谁头上。

四选一，只能选一个：
- "助手"：助手（AI）造成的。包括：助手没照做、忘了、擅自改、把东西放错位置、跑偏方向、
  输出太复杂读不懂、输出太长、判据没给出来、做了多余的事。
  **只要这句话的主语指向助手或助手的产出（「你」「你的」「输出」「记录的信息」「这么写」），
  或抱怨的是「看不懂/太复杂/方向不对/怎么验证」，就是助手。**
- "用户"：说话人自己造成的。包括：他自己的目标还没定、在两个方案之间取舍未决、
  自己的设想本身不清楚、明知排期过载仍自己选择同时推进多条线。
  **必须是原话里能看出他自己承认尚未决定或自己安排过载，不能靠推测。**
  只是「他提出一个新问题要解决」不算。
- "外部"：数据、通道、第三方、客观条件造成的（客户信息不能接入、语音识别不到、
  等别人给结论、网络或环境不通）。
- "不明"：判不出来，或这句话同时指着两边而没有偏向。

判据纪律：
1. 拿不准就选 "不明"，不许硬派。硬派会把责任推给一方，比不判更糟。
2. reason 写你凭什么这么判，一句话、不超过 30 字，要能指回原话里的字眼。
3. 不要因为「他是老板/他是用户」就默认记在他头上；也不要因为「AI 容易被批评」就默认记在助手头上。

输出：只输出一个 JSON 数组，每条对应输入一条：
[{"id":"编号","actor":"助手","reason":"这句话主语是「你的输出」，抱怨对象是助手产出"}]
不要输出任何解释文字。"""


def build_prompt(batch: list[dict]) -> str:
    lines = []
    for b in batch:
        lines.append(
            f'<item id="{b["id"]}">\n'
            f'  来源短语：{b["label"]}\n'
            f'  证据原句：{b["quote"]}\n'
            f'  所在消息（节选）：{b.get("context", "")[:400]}\n'
            f'</item>'
        )
    return "逐条判断这些消耗该记在谁头上：\n\n" + "\n".join(lines)


def collect(annotations: list[dict]) -> list[dict]:
    """把标注里所有「来源-证据」配对取出来，作为待判条目。"""
    items = []
    for a in annotations:
        if not a.get("signal"):
            continue
        for p in a.get("source_quotes") or []:
            label = (p.get("label") or "").strip()
            quote = (p.get("quote") or "").strip()
            if not label:
                continue
            items.append(
                {
                    "id": f'{a["msg_id"]}::{label}',
                    "msg_id": a["msg_id"],
                    "signal": a["signal"],
                    "label": label,
                    "quote": quote,
                    "context": a.get("content") or "",
                }
            )
    return items


def attribute(items: list[dict], out_path: Path, batch_size: int = 20, workers: int = 4,
              limit: int | None = None) -> list[dict]:
    key = llm.resolve_key()
    done = {r["id"]: r for r in read_jsonl(out_path)}
    todo = [i for i in items if i["id"] not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        return list(done.values())

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"待判 {len(todo)} 条，分 {len(batches)} 批")

    def work(batch):
        ids = [b["id"] for b in batch]
        try:
            raw = llm.chat(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(batch)}],
                key,
                max_tokens=min(6000, 200 * len(batch) + 400),
            )
        except Exception as e:
            return {}, [f"调用失败：{e}"]
        return llm.parse_actors(raw, ids)

    rows = list(done.values())
    fails: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for items_out, f in pool.map(work, batches):
            fails.extend(f)
            for iid, v in items_out.items():
                src = next((x for x in todo if x["id"] == iid), None)
                if not src:
                    continue
                rows.append({**{k: src[k] for k in ("id", "msg_id", "signal", "label", "quote")}, **v})
    write_jsonl(out_path, rows)
    print(f"归属判定完成：{len(rows)} 条（本轮新增 {len(rows) - len(done)}，异常 {len(fails)}）")
    for f in fails[:5]:
        print("  ! " + f)
    return rows
