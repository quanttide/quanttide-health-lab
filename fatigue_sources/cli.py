"""命令行入口。

    python -m fatigue_sources extract          # 会话库 → data/messages.jsonl
    python -m fatigue_sources screen           # → data/candidates.jsonl + data/rest_pool.jsonl
    python -m fatigue_sources annotate         # → data/annotations.jsonl（断点续跑）
    python -m fatigue_sources probe --n 60     # 漏检反查：从未入选池抽样判定
    python -m fatigue_sources report           # → reports/<日期>.md
    python -m fatigue_sources run              # 串起来跑一遍
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from . import annotate as annotate_mod
from . import attribute as attribute_mod
from . import corpus, report as report_mod
from .screen import screen as screen_fn

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
MESSAGES = DATA / "messages.jsonl"
CANDS = DATA / "candidates.jsonl"
REST = DATA / "rest_pool.jsonl"
ANNOT = DATA / "annotations.jsonl"
PROBE = DATA / "probe.jsonl"
ACTORS = DATA / "actors.jsonl"


def cmd_extract(args) -> None:
    recs = corpus.extract(Path(args.db).expanduser(), MESSAGES, since=args.since)
    print(f"抽出用户消息 {len(recs)} 条 → {MESSAGES}")
    if recs:
        print(f"时间跨度 {recs[0]['iso'][:10]} → {recs[-1]['iso'][:10]}，会话 "
              f"{len({r['session_id'] for r in recs})} 个")


def cmd_screen(args) -> None:
    msgs = corpus.read_jsonl(MESSAGES)
    cands, rest = screen_fn(msgs)
    corpus.write_jsonl(CANDS, cands)
    corpus.write_jsonl(REST, rest)
    print(f"候选 {len(cands)} 条 / 全部 {len(msgs)} 条；未入选池 {len(rest)} 条")


def cmd_annotate(args) -> None:
    cands = corpus.read_jsonl(CANDS)
    if not cands:
        raise SystemExit("没有候选，先跑 extract + screen")
    annotate_mod.annotate(cands, ANNOT, batch_size=args.batch, workers=args.workers, limit=args.limit)


def cmd_probe(args) -> None:
    rest = corpus.read_jsonl(REST)
    if not rest:
        raise SystemExit("未入选池是空的，先跑 screen")
    rnd = random.Random(args.seed)
    sample = rnd.sample(rest, min(args.n, len(rest)))
    for m in sample:
        m.setdefault("signal_hits", [])
    print(f"从未入选池抽 {len(sample)} 条做漏检反查（seed={args.seed}）")
    annotate_mod.annotate(sample, PROBE, batch_size=args.batch, workers=args.workers)


def cmd_attribute(args) -> None:
    anns = corpus.read_jsonl(ANNOT)
    if not anns:
        raise SystemExit("没有标注数据，先跑 annotate")
    items = attribute_mod.collect(anns)
    if not items:
        raise SystemExit("标注里没有可归属的来源条目")
    attribute_mod.attribute(items, ACTORS, batch_size=args.batch, workers=args.workers,
                            limit=args.limit)


def cmd_report(args) -> None:
    anns = corpus.read_jsonl(ANNOT)
    if not anns:
        raise SystemExit("没有标注数据，先跑 annotate")
    msgs = corpus.read_jsonl(MESSAGES)
    rest = corpus.read_jsonl(REST)
    probe = corpus.read_jsonl(PROBE)
    actors = corpus.read_jsonl(ACTORS)
    conclusion = Path(args.conclusion).read_text(encoding="utf-8") if args.conclusion else ""
    text = report_mod.build(anns, len(msgs), len(rest), probe, conclusion, actors)
    out = REPORTS / f"{args.name or 'fatigue-sources'}.md"
    report_mod.write(text, out)
    print(f"报告 → {out}（命中 {sum(1 for a in anns if a.get('fatigue'))}/{len(anns)}）")


def cmd_run(args) -> None:
    cmd_extract(args)
    cmd_screen(args)
    cmd_annotate(args)
    cmd_attribute(args)
    if args.probe:
        cmd_probe(args)
    cmd_report(args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fatigue", description="从对话语料定位疲劳并归因")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--db", default=str(corpus.DEFAULT_DB), help="Hermes 会话库路径")
        p.add_argument("--since", default=None, help="只取该日期之后（ISO，如 2026-08-01）")
        p.add_argument("--batch", type=int, default=20)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--limit", type=int, default=None, help="本轮最多标多少条（试跑用）")
        p.add_argument("--probe", action="store_true", help="run 时是否做漏检反查")
        p.add_argument("--n", type=int, default=60, help="漏检反查抽样条数")
        p.add_argument("--seed", type=int, default=20261022)
        p.add_argument("--conclusion", default=None, help="结论文件的路径，拼进报告")
        p.add_argument("--name", default=None, help="报告文件名（不含扩展名）")

    for name in ("extract", "screen", "annotate", "attribute", "probe", "report", "run"):
        p = sub.add_parser(name)
        common(p)
        p.set_defaults(func=globals()[f"cmd_{name}"])

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
