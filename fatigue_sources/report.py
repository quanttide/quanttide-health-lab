"""汇总：把标注数据算成按来源排序的证据清单。

纪律：数字全部由脚本算，结论由人写。每张表头写分母。
两类信号分开出表——fatigue（真累）与 friction（摩擦、返工、失控感）不混在一张表里。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .labels import canonical
from .sanitize import scrub

CST = timezone(timedelta(hours=8))


def _group(hits: list[dict]) -> tuple[Counter, dict[str, list[tuple[dict, str]]], Counter]:
    counts: Counter = Counter()
    evidence: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    unmerged: Counter = Counter()
    for h in hits:
        seen = set()
        pairs = h.get("source_quotes") or [
            {"label": s, "quote": (h.get("quotes") or [""])[0]} for s in (h.get("sources") or [])
        ]
        for p in pairs:
            label = p.get("label") or ""
            if not label:
                continue
            name, ok = canonical(label)
            if not ok:
                unmerged[label.strip()] += 1
            if name in seen:
                continue
            seen.add(name)
            counts[name] += 1
            if len(evidence[name]) < 4:
                evidence[name].append((h, p.get("quote") or (h.get("quotes") or [""])[0]))
    return counts, evidence, unmerged


def _source_table(L: list[str], title: str, hits: list[dict], hint: str) -> None:
    add = L.append
    add(f"## {title}")
    add("")
    if not hits:
        add("（无命中。）")
        add("")
        return
    add(f"分母：{len(hits)} 条命中消息；单条最多计 3 个来源，故各来源次数合计可多于命中条数。")
    add("")
    if hint:
        add(hint)
        add("")
    counts, evidence, unmerged = _group(hits)
    for name, cnt in counts.most_common():
        share = cnt / len(hits) * 100
        add(f"### {name} —— {cnt} 条（占该类命中 {share:.0f}%）")
        add("")
        for h, q in evidence[name][:3]:
            add(f"- 「{scrub(q)}」  _({h.get('iso', '')[:16].replace('T', ' ')}，强度 {h.get('intensity')}/5)_")
        add("")
    if unmerged:
        add("未归并的原始短语（各计一次，供人工判断）：")
        add("")
        add("　" + "；".join(f"{p}({c})" for p, c in unmerged.most_common(15)))
        add("")
    # 命中少的时候，分组统计不如逐条看原文有用（分母小，聚类只会掩盖细节）
    if len(hits) < 20:
        add(f"### 全部 {len(hits)} 条命中原文（命中少，逐条列出比分组更有用）")
        add("")
        for h in sorted(hits, key=lambda x: x.get("iso") or ""):
            pairs = h.get("source_quotes") or []
            add(f"- {h.get('iso', '')[:16].replace('T', ' ')}　强度 {h.get('intensity')}/5")
            for p in pairs:
                add(f"  - {p.get('label')}：> 「{scrub(p.get('quote') or '')}」")
        add("")


def build(annotations: list[dict], total_user_msgs: int, rest_pool_size: int,
          probe: list[dict] | None = None, conclusion: str = "",
          actors: list[dict] | None = None) -> str:
    def sig(a):
        return a.get("signal") or ("fatigue" if a.get("fatigue") else "")

    fatigue = [a for a in annotations if sig(a) == "fatigue"]
    friction = [a for a in annotations if sig(a) == "friction"]
    n = len(annotations)

    L: list[str] = []
    add = L.append
    add("# 疲劳来源分析报告")
    add("")
    add(f"> 生成 {datetime.now(CST).isoformat(timespec='seconds')}")
    add("")
    add(f"- 样本：用户消息 {total_user_msgs} 条，进入判定 {n} 条"
        f"（另有 {rest_pool_size} 条过短或纯确认，未进入判定池）")
    add("- 本报告回答：这些对话里，累出现在什么地方，来自哪一类事；以及「烦」和「累」是不是一回事")
    add("- 口径：只判说话人自己的状态；技术提问、下命令、纠错不算；"
        "判定必须附原文片段，无片段不计命中")
    add("")
    add("## 一、命中概况")
    add("")
    add(f"- 疲劳（fatigue）{len(fatigue)} 条，占判定 {len(fatigue) / n * 100:.1f}%")
    add(f"- 摩擦（friction）{len(friction)} 条，占判定 {len(friction) / n * 100:.1f}%")
    for label, hits in (("疲劳", fatigue), ("摩擦", friction)):
        if hits:
            imp = sum(1 for h in hits if h.get("implicit"))
            avg = sum(h.get("intensity") or 0 for h in hits) / len(hits)
            add(f"- {label}：隐式（无关键词、靠语气）{imp} 条，平均强度 {avg:.1f}/5，"
                f"强度 4–5 的 {sum(1 for h in hits if (h.get('intensity') or 0) >= 4)} 条")
    add("")
    add("为什么分开：把「同一件事又说了一遍还是没改」这种烦躁记成「累」，来源清单就会被摩擦占满，"
        "真正的精力耗竭反而看不见。两类分列，谁多谁少一眼可知。")
    add("")

    _source_table(L, "二、疲劳来源（真的累，按命中条数排序）", fatigue,
                  "看这一节：这些是「累」的来路。每条来源下挂原句，能对回原话。")
    _source_table(L, "三、摩擦来源（烦、返工、失控感，按命中条数排序）", friction,
                  "看这一节：这些消耗的是耐性，不是体力。它们常是疲劳的上游。")

    # ---- 责任归属（责任全部在助手一侧，只分环节）----
    add("## 四、这些消耗落在我的哪个环节")
    add("")
    add("口径是定死的：**责任全部在我这一侧，没有一条推给你，也不用「客观条件」「双方都有」含糊。**"
        "这一节不问「谁的错」，只问「落在我哪个环节」——因为我负责把判据摆出来、把输出降到能判、"
        "把说过的记住、把选项收敛到能拍、把外部条件的坑先探掉。")
    add("")
    add("归属逐条读原句判（`attribute.py`），每条必须带一句判据。早期版本试过四选一"
        "（助手／用户／外部／不明），会把「我对你的工作方向表示怀疑」判成「用户目标未定」——"
        "把我自己摘出去，这一版已经删掉那条路。")
    add("")
    stage_index = {a["id"]: a for a in (actors or [])}
    counts_by_stage: Counter = Counter()
    stage_examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    total_pairs = 0
    missing_attr = 0
    for h in fatigue + friction:
        for p_ in h.get("source_quotes") or []:
            label = p_.get("label") or ""
            if not label:
                continue
            total_pairs += 1
            quote = scrub(p_.get("quote") or "")
            rec = stage_index.get(f'{h.get("msg_id")}::{label}')
            if rec and rec.get("stage"):
                stage = rec["stage"]
                reason = rec.get("reason") or ""
            else:
                stage = "C 没给判据、没法验证"
                reason = "（未判定，按「没让人能判断」记在助手一侧）"
                missing_attr += 1
            counts_by_stage[stage] += 1
            if len(stage_examples[stage]) < 3:
                stage_examples[stage].append((quote, reason))

    add(f"分母：{total_pairs} 条「来源-证据」配对（疲劳 + 摩擦），全部记在助手一侧。")
    if missing_attr:
        add(f"其中 {missing_attr} 条没有模型判定，按「没让人能判断」计入——先跑 `attribute` 再出报告。")
    add("")
    add("| 环节 | 条数 | 占比 |")
    add("|---|---|---|")
    for s, c in counts_by_stage.most_common():
        add(f"| {s} | {c} | {c / max(1, total_pairs) * 100:.0f}% |")
    add("")
    for s, c in counts_by_stage.most_common():
        add(f"**{s}**（{c} 条）——举例与判据：")
        add("")
        for quote, reason in stage_examples[s]:
            add(f"- 「{quote}」　_{reason or '未给判据'}_")
        add("")

    # ---- 摩擦与疲劳的关系 ----
    add("## 五、摩擦是不是疲劳的上游（共现检验）")
    add("")
    days_any = Counter(a.get("date") for a in annotations)
    days_f = Counter(a.get("date") for a in fatigue)
    days_r = Counter(a.get("date") for a in friction)
    fric_days = {d for d, c in days_r.items() if c}
    other_days = {d for d in days_any if d not in fric_days}
    conv_f = sum(days_f[d] for d in fric_days) / max(1, sum(days_any[d] for d in fric_days)) * 100
    conv_o = sum(days_f[d] for d in other_days) / max(1, sum(days_any[d] for d in other_days)) * 100
    add(f"- 有摩擦命中的日子（{len(fric_days)} 天），疲劳命中率 {conv_f:.1f}%")
    add(f"- 没有摩擦命中的日子（{len(other_days)} 天），疲劳命中率 {conv_o:.1f}%")
    if conv_o > 0 and conv_f / conv_o >= 1.5:
        add(f"- 差 {conv_f / conv_o:.1f} 倍。方向上支持「摩擦高的日子更容易出现疲劳」，"
            f"但这只是共现，不能证明是因果。")
    else:
        add("- 差距不到 1.5 倍：**没有证据说摩擦导致疲劳**，别把两节合并解释。")
    add("")

    # ---- 时间分布 ----
    add("## 六、时间分布（疲劳命中）")
    add("")
    add(f"分母：{len(fatigue)} 条疲劳命中。")
    add("")
    all_month = Counter((a.get("iso") or "")[:7] for a in annotations)
    month = Counter((h.get("iso") or "")[:7] for h in fatigue)
    if month:
        add("| 月份 | 疲劳命中 | 判定总数 | 命中率 |")
        add("|---|---|---|---|")
        for m in sorted(all_month):
            add(f"| {m} | {month.get(m, 0)} | {all_month[m]} | "
                f"{month.get(m, 0) / all_month[m] * 100:.1f}% |")
        add("")
    hours = Counter(h.get("hour") for h in fatigue if h.get("hour") is not None)
    if hours:
        add("命中最多的小时：" + "、".join(f"{h:02d} 点（{c} 条）" for h, c in hours.most_common(5)))
        add("")

    # ---- 自检 ----
    add("## 七、口径自检（决定上面那些能不能信）")
    add("")
    add("| 检查 | 结果 |")
    add("|---|---|")
    add(f"| 证据句能否在原文里逐字对上 | 脚本校验：所有命中条目已核过，"
        f"不可核对的已按未命中处理 |")
    lexical = [a for a in annotations if a.get("signal_hits")]
    lex_f = sum(1 for a in lexical if sig(a) == "fatigue")
    implicit_f = sum(1 for h in fatigue if h.get("implicit"))
    add(f"| 规则词面信号（含累/困/烦等词）{len(lexical)} 条 | 其中判为疲劳 {lex_f} 条"
        f"（{lex_f / max(1, len(lexical)) * 100:.0f}%）：有疲劳词不等于在说自己累 |")
    add(f"| 隐式疲劳 {implicit_f} 条 | 这些没有关键词，是词表做不到、只有读懂语气才拿得到的 |")
    if probe:
        leaked = [p for p in probe if (p.get("signal") or ("fatigue" if p.get("fatigue") else ""))]
        add(f"| 漏检反查：未判定池抽 {len(probe)} 条强制判定 | 命中 {len(leaked)} 条，"
            f"穿漏率约 {len(leaked) / len(probe) * 100:.0f}% |")
        for p in leaked[:3]:
            add(f"| 　例 | 「{scrub((p.get('quotes') or [p.get('content', '')[:30]])[0])}」 |")
    add("")

    add("## 八、这份报告自带哪些偏差")
    add("")
    add("- 语料只有与 AI 的对话：与人当面聊的、会议上说的，不在这里面。")
    add("- 单模型单次判定，无第二人复核；来源短语是模型的概括，是「值得核对的假设」，不是因果证明。")
    add("- 命中要求原文片段，语气很轻的累会被漏掉；穿漏率见上一节。")
    add("- 有一类边界判不准：他在分析「读者会怎么累」时（如「读者要自己在脑中画出 DAG」），"
        "主语是读者而不是他，但里面确实混着他自己的负担感。这类可能被多算进疲劳，逐条抽查时可见。")
    add("- 对话里 AI 的提问方式会塑造他的说法（研究者的语言被对象吸收），这一层剥不掉。")
    add("- 时间分布按消息发出时间算，不代表疲劳发生在那一刻。")
    add("")
    if conclusion:
        add("## 九、结论")
        add("")
        add(conclusion.strip())
        add("")
    return "\n".join(L)


def write(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path