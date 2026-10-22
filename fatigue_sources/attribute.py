"""责任归属：这些消耗分别落在助手的哪个环节上。

口径由用户定死：**责任全部在助手一侧，不把任何一条推给用户，也不用「客观条件」「双方都有」
来含糊。** 助手负责把判据摆出来、把输出降到能判、把说过的记住、把选项收敛到能拍、
把外部条件的坑先探掉——这些没做到，就是助手的问题。

第一版问「是谁造成的」（四选一），会把「他不信任你」「你输出太长」判成他的目标未定；
第二版逐条读原句判，仍留了「用户」这一类。这一版不再问「谁的错」，只问「落在哪个环节」：

    A 没照做、忘了、擅自改        —— 说过的没记住、没按说的做、越权改东西、放错位置
    B 输出太复杂、看不懂          —— 密度高、太长、术语没解释、看不出主线
    C 没给判据、没法验证          —— 判不了对错、不知道怎么验收、概念对不上他的理解
    D 没把选项收敛到能拍          —— 他在两个方案之间纠结、不知道要写什么、目标悬着
    E 没把外部条件的坑先探掉      —— 数据接不上、通道不通、等别人的结论

D 和 E 是把旧版甩出去的那些翻回来：他没定，是因为助手没把选项收敛成一个能拍的方案；
客户数据接不上，助手该先探清并给绕法，而不是让他撞上去再发愁。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from .corpus import read_jsonl, write_jsonl

STAGES = [
    "A 没照做、忘了、擅自改",
    "B 输出太复杂、看不懂",
    "C 没给判据、没法验证",
    "D 没把选项收敛到能拍",
    "E 没把外部条件的坑先探掉",
]

STAGE_ALIASES = {
    "a": STAGES[0], "a 没照做": STAGES[0], "没照做": STAGES[0], "擅自改": STAGES[0],
    "b": STAGES[1], "b 输出太复杂": STAGES[1], "输出": STAGES[1], "看不懂": STAGES[1],
    "c": STAGES[2], "c 没给判据": STAGES[2], "判据": STAGES[2], "验证": STAGES[2],
    "d": STAGES[3], "d 没把选项收敛到能拍": STAGES[3], "收敛": STAGES[3], "取舍": STAGES[3],
    "e": STAGES[4], "e 没把外部条件的坑先探掉": STAGES[4], "外部": STAGES[4], "环境": STAGES[4],
}

SYSTEM = """你在给对话记录做「责任归属」标注。责任口径是定死的：**这些消耗全部记在助手（AI）头上，
不许推给用户，也不许用「客观条件」「双方都有」「不明」来含糊。**

每条记录是一句原话（说话人是用户）加一个「消耗来源」短语。你要判断它落在助手的哪个环节，
五选一：

- "A 没照做、忘了、擅自改"：说过的没记住、没按说的做、擅自多做或少做、越权改动、
  把东西放错位置、版本或分支搞错、该同步的没同步。
- "B 输出太复杂、看不懂"：密度高、太长、名词没解释、看不出主线、排版要来回翻、读起来吃力。
- "C 没给判据、没法验证"：他判不了对错、不知道怎么验收、结果无法核对、概念对不上他的理解。
- "D 没把选项收敛到能拍"：他在两个方案之间纠结、不知道要写什么、目标悬着、方向要找。
  **这也是助手的活——助手该把选项收敛成一个能拍的方案，而不是把一堆可能摆在他面前。**
- "E 没把外部条件的坑先探掉"：数据接不上、语音或通道不通、要等别人的结论、第三方并发干扰。
  **同样是助手的活——该在动手前把这些坑探出来并给绕法，而不是让他撞上去再发愁。**

纪律：
1. 必须选一个环节，不许选「不明」，不许把责任指回用户。
2. reason 一句话、不超过 30 字，要能指回原话里的字眼。
3. 不要因为是用户的用词或偏好就记成他的问题。

输出：只输出一个 JSON 数组，每条对应输入一条：
[{"id":"编号","stage":"C 没给判据、没法验证","reason":"他说不知道该怎么验证，助手没给验收判据"}]
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
    return "逐条判断这些消耗落在助手哪个环节上：\n\n" + "\n".join(lines)


def normalize_stage(raw: str) -> str:
    """把模型给的环节名收敛到五类；认不出的一律记到 C（没给判据）——那也是助手的活。"""
    text = (raw or "").strip()
    if not text:
        return STAGES[2]
    for stage in STAGES:
        if text.startswith(stage[:1]) or stage in text:
            return stage
    low = text.lower()
    for k, v in STAGE_ALIASES.items():
        if low.startswith(k) or k in low:
            return v
    return STAGES[2]


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
    by_id = {i["id"]: i for i in todo}

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
                src = by_id.get(iid)
                if not src:
                    continue
                rows.append(
                    {
                        **{k: src[k] for k in ("id", "msg_id", "signal", "label", "quote")},
                        "stage": normalize_stage(v.get("stage")),
                        "reason": v.get("reason") or "",
                    }
                )
    write_jsonl(out_path, rows)
    print(f"归属判定完成：{len(rows)} 条（本轮新增 {len(rows) - len(done)}，异常 {len(fails)}）")
    for f in fails[:5]:
        print("  ! " + f)
    return rows
