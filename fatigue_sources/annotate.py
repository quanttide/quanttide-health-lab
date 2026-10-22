"""逐条标注：这句话里有没有消耗信号，这股消耗从哪来，原文哪句能证明。

两类信号分开计，不混：
  fatigue  —— 说话人自己的精力/心力被消耗（累、撑不住、提不起劲）
  friction —— 事情反复、被替做决定、信息对不上、工具与协作不顺带来的烦躁与失控感
friction 是「消耗」但不是「累」，硬要合并会把「烦」当成「疲」，结论就偏了。

判定口径全部写在 prompt 里；改口径只改 prompt，不事后改数据。
断点续跑：已标的 msg_id 跳过，重跑只补缺。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from .corpus import read_jsonl, write_jsonl

SYSTEM = """你是对话语料的标注员。任务：在说话人（用户）自己说的话里，找出「消耗信号」，并判断这股消耗来自哪里。

signal 只有三个取值：
- "fatigue"：说话人自己的精力/心力被消耗——累、疲、撑不住、提不起劲、力不从心、心力不够、撑不下去。
- "friction"：事情反复、被替做决定、信息对不上、工具或协作不顺引发的烦躁、失控感、不耐烦。
  这类有消耗，但不是「我累了」。典型：同一件事已经说过又被漏掉、别人擅自做了决定、版本对不上。
- ""（空）：其余全部，包括技术提问、下命令、纠错、评价工具、单纯描述事实。

判定规则：
1. 只判说话人**自己**的状态。「这个报错很烦」是对工具的评价，除非他在说自己被它耗住了。
2. implicit=true 表示没有直接的疲劳/烦躁词，靠语气（回避、推迟、自嘲、放弃、懒得）看出来的；有明确词就 false。
3. intensity：1=顺口一提，3=明显，5=明确说撑不下去或崩了。
4. sources：1–3 条，「什么在消耗他」，具体口语短语，6–16 字，必须能对上原句。
   - 禁止「工作压力」「情绪问题」这类空泛词；要写「同一件事反复解释」「方案迟迟定不下来」。
   - 不许凭常识猜，只能是原句能支撑的推断。
   - 每条来源必须带一句**对应的** quote：就是支撑这条来源的那句话，原样截取 6–40 字，
     不得改写、拼接、概括。不同来源要引不同的句子，不要一条句子挂三个来源。
5. signal 非空却没有任何可核对的 quote 的，一律按空处理。
6. **主语必须是说话人自己。**「读者每处理一个点都要做一次判断」「用户会觉得累」「员工扛不住」
   这类是在评论文档或他人，不是他自己的状态——不算。只有他同时说了自己的感受（「我看不懂」
   「我判断不了」「我也觉得负担重」）才算。
7. 每条消息独立判断。

输出格式：只输出一个 JSON 数组，每条对应输入的一条：
[{"id":"消息id","signal":"fatigue","implicit":false,"intensity":3,
  "sources":[{"label":"来源短语","quote":"支撑这句话的原文片段"}]}]
signal 为空就写 {"id":"消息id","signal":""}。不要输出任何解释文字。"""


def build_batch_prompt(batch: list[dict]) -> str:
    lines = []
    for m in batch:
        lines.append(f'<msg id="{m["msg_id"]}" time="{m["iso"]}">{clip(m["content"], 900)}</msg>')
    return "以下是待判定的消息，逐条判定：\n\n" + "\n".join(lines)


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def annotate(cands: list[dict], out_path: Path, batch_size: int = 20, workers: int = 4,
             limit: int | None = None) -> list[dict]:
    key = llm.resolve_key()
    done = {r["msg_id"]: r for r in read_jsonl(out_path)}
    todo = [m for m in cands if m["msg_id"] not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        return list(done.values())

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"待标 {len(todo)} 条，分 {len(batches)} 批（每批 {batch_size} 条，并发 {workers}）")

    failures: list[str] = []

    def work(pair):
        idx, batch = pair
        ids = [m["msg_id"] for m in batch]
        try:
            raw = llm.chat(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": build_batch_prompt(batch)}],
                key,
                max_tokens=min(8000, 280 * len(batch) + 400),
            )
        except Exception as e:
            return idx, {}, [f"批 {idx} 调用失败：{e}"]
        items, fails = llm.parse_items(raw, ids)
        return idx, items, fails

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (idx, items, fails) in enumerate(pool.map(work, enumerate(batches)), 1):
            results.update(items)
            failures.extend(fails)
            if n % 10 == 0 or n == len(batches):
                print(f"  已处理 {n}/{len(batches)} 批，累计 {len(results)} 条，异常 {len(failures)} 条")

    by_id = {m["msg_id"]: m for m in todo}
    rows = list(done.values())
    for mid, item in results.items():
        src = by_id[mid]
        rows.append(
            {
                "msg_id": mid,
                "session_id": src.get("session_id"),
                "iso": src.get("iso"),
                "date": src.get("date"),
                "hour": src.get("hour"),
                "content": src.get("content"),
                "signal_hits": src.get("signal_hits", []),
                **item,
            }
        )
    rows.sort(key=lambda r: r.get("iso") or "")
    write_jsonl(out_path, rows)
    print(f"标注完成：{len(rows)} 条（本轮新增 {len(results)}，未返回 {len(todo) - len(results)}）")
    for f in failures[:10]:
        print("  ! " + f)
    return rows
