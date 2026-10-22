"""离线自测：不联网，只验解析容错、归并、宽筛、脱敏这几处容易错的地方。

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fatigue_sources.attribute import STAGES, normalize_stage
from fatigue_sources.labels import canonical
from fatigue_sources.llm import parse_items
from fatigue_sources.sanitize import scrub
from fatigue_sources.screen import screen


class TestParse(unittest.TestCase):
    def test_plain_array_with_paired_quote(self):
        raw = json.dumps([{"id": "a1", "signal": "fatigue", "intensity": 4,
                           "sources": [{"label": "输出太长", "quote": "这个输出还是太长了"}]}])
        items, fails = parse_items(raw, ["a1"])
        self.assertFalse(fails)
        self.assertEqual(items["a1"]["signal"], "fatigue")
        self.assertEqual(items["a1"]["source_quotes"][0]["quote"], "这个输出还是太长了")

    def test_no_quote_means_no_hit(self):
        raw = json.dumps([{"id": "a1", "signal": "fatigue",
                           "sources": [{"label": "输出太长"}]}])
        items, _ = parse_items(raw, ["a1"])
        self.assertEqual(items["a1"]["signal"], "")

    def test_two_objects_back_to_back(self):
        raw = ('[{"id":"a1","signal":"","sources":[]}]'
               ' 说明：[{"id":"a2","signal":"friction","sources":[{"label":"返工","quote":"又改错了"}]}]')
        items, fails = parse_items(raw, ["a1", "a2"])
        self.assertEqual(items["a1"]["signal"], "")
        self.assertEqual(items["a2"]["signal"], "friction")
        self.assertFalse(fails)  # 两个块都要吃掉，不能只看第一个

    def test_missing_id_reported(self):
        raw = json.dumps([{"id": "a1", "signal": ""}])
        items, fails = parse_items(raw, ["a1", "a2"])
        self.assertTrue(any("未返回" in f for f in fails))

    def test_chinese_keys_and_string_bool(self):
        raw = json.dumps([{"id": "a1", "信号": "摩擦", "强度": "5",
                           "来源": [{"来源": "被替做决定", "原句": "谁让你手动触发的"}],
                           "隐式": "是"}])
        items, _ = parse_items(raw, ["a1"])
        it = items["a1"]
        self.assertEqual(it["signal"], "friction")
        self.assertEqual(it["intensity"], 5)
        self.assertTrue(it["implicit"])

    def test_string_sources_fall_back_to_quotes(self):
        raw = json.dumps([{"id": "a1", "signal": "fatigue", "sources": ["太复杂"],
                           "quotes": ["看不懂，太复杂了"]}])
        items, _ = parse_items(raw, ["a1"])
        self.assertEqual(items["a1"]["source_quotes"][0]["quote"], "看不懂，太复杂了")

    def test_intensity_clamped(self):
        raw = json.dumps([{"id": "a1", "signal": "fatigue", "intensity": 99,
                           "sources": [{"label": "x", "quote": "y"}]}])
        items, _ = parse_items(raw, ["a1"])
        self.assertEqual(items["a1"]["intensity"], 5)


class TestLabels(unittest.TestCase):
    def test_longest_keyword_wins(self):
        # 「围绕」里含「绕」，不能把这条吸进「太复杂」（那是短词误伤的老坑）
        self.assertEqual(canonical("反复围绕同一个概念纠缠")[0], "反复搞不对、整段重来")

    def test_group_hits(self):
        self.assertEqual(canonical("被替做决定")[0], "被替做决定、擅自改动")
        self.assertEqual(canonical("同一件事已经说过又被漏掉")[0], "说的没被记住、没照做")
        self.assertEqual(canonical("名词密度极高且不解释")[0], "太复杂、看不懂、看得吃力")

    def test_unmerged_passthrough(self):
        name, ok = canonical("玩家只能随便选")
        self.assertFalse(ok)
        self.assertEqual(name, "玩家只能随便选")


class TestScreen(unittest.TestCase):
    def test_bare_ack_dropped(self):
        cands, rest = screen([{"msg_id": "1", "content": "好的"}])
        self.assertEqual(len(cands), 0)
        self.assertEqual(len(rest), 1)

    def test_signal_word_kept(self):
        cands, _ = screen([{"msg_id": "1", "content": "累"}])
        self.assertEqual(len(cands), 1)

    def test_long_message_kept(self):
        cands, _ = screen([{"msg_id": "1", "content": "这一段话足够长并且没有任何疲劳方面的词汇表达"}])
        self.assertEqual(len(cands), 1)


class TestSanitize(unittest.TestCase):
    def test_keys_and_emails_scrubbed(self):
        text = "key gsk_abcdefgh12345678 和 a@b.com 还有 token=xyz"
        out = scrub(text)
        self.assertNotIn("gsk_abcdefgh12345678", out)
        self.assertNotIn("a@b.com", out)
        self.assertIn("已隐去", out)


class TestStage(unittest.TestCase):
    """责任归属只分环节，且全部记在助手一侧：认不出也必须落到某一环节，不能回到用户身上。"""

    def test_leading_letter(self):
        self.assertEqual(normalize_stage("A 没照做"), "A 没照做、忘了、擅自改")
        self.assertEqual(normalize_stage("B"), "B 输出太复杂、看不懂")

    def test_full_name(self):
        self.assertEqual(normalize_stage("D 没把选项收敛到能拍"), "D 没把选项收敛到能拍")

    def test_unknown_still_assistant_side(self):
        self.assertEqual(normalize_stage(""), "C 没给判据、没法验证")
        self.assertEqual(normalize_stage("用户"), "C 没给判据、没法验证")

    def test_stages_are_all_assistant_side(self):
        self.assertEqual(len(STAGES), 5)
        self.assertTrue(all(s[0] in "ABCDE" for s in STAGES))


if __name__ == "__main__":
    unittest.main()
