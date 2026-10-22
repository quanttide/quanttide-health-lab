"""宽筛：把明显不含自身状态的短消息剔出去，只留候选集。

判据要宽——目标是少漏，不是省调用。凡是「像在说自己状态」的都留下：
命中弱信号词，或长度够长（长句通常有内容，短句多是命令/确认）。

未入选的消息不丢弃，单独留池子，供漏检反查（report 会报抽样穿漏率）。
"""

from __future__ import annotations

import re

# 弱信号词：出现即入选候选。只求宽，不求准——准由模型那一步给。
SIGNAL_WORDS = [
    "累", "疲", "乏", "困", "撑不住", "撑不", "扛不", "精力", "力气", "提不起",
    "烦", "崩", "焦虑", "压力", "耗", "不想", "懒得", "没劲", "脑", "心力",
    "难受", "低落", "情绪", "失眠", "睡不着", "头疼", "头痛", "焦虑", "慌",
    "忙", "赶", "来不及", "拖", "熬", "熬夜", "困死", "累死", "心累", "厌倦",
    "抵触", "逃避", "放弃", "做不到", "搞不", "弄不", "转不动", "卡住",
    "好累", "太累", "真累", "疲了", "倦",
]

# 纯操作类的短句，即使含词也不值得单独看（但长句不排除）
BARE_ACK = re.compile(r"^(?:同意|好的|好|嗯+|ok|OK|继续|行|可以|对|是的|收到|谢谢|辛苦了?|是|否|不要|算了)[。！.!~]*$")
MIN_CHARS = 15


def screen(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """返回 (候选, 未入选池)。两条都保留分母信息。"""
    cands, rest = [], []
    for m in messages:
        text = (m.get("content") or "").strip()
        hit = [w for w in SIGNAL_WORDS if w in text]
        m = dict(m)
        m["signal_hits"] = hit
        if BARE_ACK.match(text) and not hit:
            rest.append(m)
            continue
        if hit or len(text) >= MIN_CHARS:
            cands.append(m)
        else:
            rest.append(m)
    return cands, rest
