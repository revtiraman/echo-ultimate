"""
ECHO ULTIMATE — Robust <confidence><answer> parser.
Handles 15+ edge cases. NEVER crashes. Always returns a ParseResult.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────
_CONF_TAG_RE = re.compile(r"<confidence>\s*([^<]*?)\s*</confidence>", re.IGNORECASE | re.DOTALL)
_ANS_TAG_RE  = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_NUM_RE      = re.compile(r"-?\d+(?:\.\d+)?")
_QUOTES_RE   = re.compile(r'^["\'](.+)["\']$', re.DOTALL)

# Verbal confidence map
_VERBAL_MAP = {
    "very sure": 90, "very certain": 90, "extremely sure": 95, "absolutely sure": 98,
    "certain": 88, "confident": 78, "sure": 75, "fairly sure": 70,
    "somewhat sure": 60, "unsure": 35, "uncertain": 30, "not sure": 25,
    "very unsure": 15, "very uncertain": 15, "no idea": 5, "no clue": 5,
    "high": 85, "medium": 50, "low": 25, "moderate": 55,
    "probably": 65, "likely": 65, "unlikely": 30, "doubtful": 20,
}

DEFAULT_CONFIDENCE = 50


@dataclass
class ParseResult:
    """Result of parsing one LLM response."""
    confidence: int          = DEFAULT_CONFIDENCE
    answer: str              = ""
    parse_success: bool      = False
    confidence_source: str   = "default"   # "tag"|"default"|"clipped"|"inferred"|"verbal"
    answer_source: str       = "empty"     # "tag"|"last_sentence"|"full_text"|"empty"
    is_abstention: bool      = False       # True if answer is "I don't know"
    raw: str                 = ""


# ── Confidence extraction ─────────────────────────────────────────────────────

def _extract_confidence(text: str) -> tuple[int, str]:
    """Return (confidence_int, source_label). Never raises."""
    matches = _CONF_TAG_RE.findall(text)
    if not matches:
        return DEFAULT_CONFIDENCE, "default"

    raw = matches[0].strip()  # use first match only (edge case 8)

    if not raw:
        return DEFAULT_CONFIDENCE, "default"

    # Edge case 6: verbal confidence
    raw_lower = raw.lower()
    for phrase, val in _VERBAL_MAP.items():
        if phrase in raw_lower:
            return val, "verbal"

    # Edge case 7 + 10 + 11: float / out-of-range number
    nums = _NUM_RE.findall(raw.replace(",", ""))
    if nums:
        try:
            val = round(float(nums[0]))
            clipped = max(0, min(100, val))
            source = "clipped" if clipped != val else "tag"
            return clipped, source
        except ValueError:
            pass

    return DEFAULT_CONFIDENCE, "default"


# ── Answer extraction ─────────────────────────────────────────────────────────

def _extract_answer(text: str) -> tuple[str, str]:
    """Return (answer_str, source_label). Never raises."""
    matches = _ANS_TAG_RE.findall(text)
    if matches:
        raw_ans = matches[0].strip()

        # Edge case 13: strip surrounding quotes
        m = _QUOTES_RE.match(raw_ans)
        if m:
            raw_ans = m.group(1).strip()

        return raw_ans, "tag"

    # No answer tag — fall back to text after </confidence>
    after_conf = re.split(r"</confidence>", text, flags=re.IGNORECASE, maxsplit=1)
    if len(after_conf) > 1:
        tail = after_conf[1].strip()
        # Remove any remaining tags
        tail = re.sub(r"<[^>]+>", " ", tail).strip()
        if tail:
            return tail, "full_text"

    # Last sentence fallback
    clean = re.sub(r"<[^>]+>.*?</[^>]+>", " ", text, flags=re.DOTALL)
    clean = re.sub(r"<[^>]+>", " ", clean).strip()
    sentences = [s.strip() for s in re.split(r"[.!?]", clean) if s.strip()]
    if sentences:
        return sentences[-1], "last_sentence"

    return "", "empty"


# ── Main parse function ───────────────────────────────────────────────────────

def parse_response(text) -> ParseResult:
    """
    Parse an LLM response into confidence and answer.

    Handles edge cases:
    1.  Perfect format
    2.  Reversed tags
    3.  No confidence tag → default 50
    4.  No answer tag → extract from remaining text
    5.  Confidence out of range → clip to [0,100]
    6.  Verbal confidence ("high", "low", "very sure") → mapped to int
    7.  Float confidence → rounded
    8.  Multiple tags → first occurrence
    9.  Nested tags → regex extracts correctly
    10. Confidence > 100 → clipped to 100
    11. Negative confidence → clipped to 0
    12. Empty answer → empty string
    13. Answer with quotes → stripped
    14. "I don't know" → is_abstention=True, confidence=5
    15. None / non-string input → safe defaults
    """
    if text is None:
        return ParseResult(raw="")

    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ParseResult(raw="")

    conf, conf_src = _extract_confidence(text)
    ans, ans_src   = _extract_answer(text)

    # Edge case 14: abstention detection
    is_abstention = False
    if ans and any(phrase in ans.lower() for phrase in
                   ["i don't know", "i do not know", "i'm not sure", "no idea", "don't know"]):
        is_abstention = True
        conf = min(conf, 10)
        conf_src = "inferred"

    parse_success = (conf_src == "tag" or conf_src == "verbal") and ans_src == "tag"

    return ParseResult(
        confidence=conf,
        answer=ans,
        parse_success=parse_success,
        confidence_source=conf_src,
        answer_source=ans_src,
        is_abstention=is_abstention,
        raw=text,
    )


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_prompt(
    question: str,
    domain: str,
    difficulty: str = "medium",
    show_difficulty: bool = True,
) -> str:
    """
    Build a formatted prompt combining the system instruction + question.

    Args:
        show_difficulty: Phase 1 shows difficulty; Phase 2+ hides it.
    """
    from config import cfg

    domain_hints = {
        "math":     "This is a math problem. Give a numeric answer.",
        "logic":    "This is a logic/reasoning question. Give the letter (A/B/C/D).",
        "factual":  "This is a factual question. Give a concise text answer.",
        "science":  "This is a science question. Give the letter or a concise answer.",
        "medical":  "This is a medical question. Give the letter (A/B/C/D).",
        "coding":   "This is a coding question. Give a concise answer.",
        "creative": "This is a creative question. Give a short text answer.",
    }
    hint = domain_hints.get(domain, "Give a concise answer.")

    diff_str = f" [{difficulty.upper()}]" if show_difficulty else ""
    header = f"Domain: {domain.capitalize()}{diff_str}\n{hint}\n\n"

    return f"{cfg.SYSTEM_PROMPT}\n\n{header}Question: {question}"


# ── Self-tests ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    failures = []

    def check(text, exp_conf, exp_ans, label, exp_abst=False):
        r = parse_response(text)
        ok = True
        if exp_conf is not None and r.confidence != exp_conf:
            failures.append(f"[{label}] confidence: expected {exp_conf}, got {r.confidence}")
            ok = False
        if exp_ans is not None and r.answer != exp_ans:
            failures.append(f"[{label}] answer: expected '{exp_ans}', got '{r.answer}'")
            ok = False
        if r.is_abstention != exp_abst:
            failures.append(f"[{label}] is_abstention: expected {exp_abst}, got {r.is_abstention}")
            ok = False
        if ok:
            print(f"  ✅  {label}")

    print("Running ECHO Ultimate parser tests…")

    check("<confidence>75</confidence><answer>Paris</answer>", 75, "Paris", "1. perfect format")
    check("<answer>Paris</answer><confidence>75</confidence>", 75, "Paris", "2. reversed tags")
    check("<answer>London</answer>", DEFAULT_CONFIDENCE, "London", "3. no confidence tag")
    check("<confidence>55</confidence>", 55, None, "4. no answer tag")
    check("<confidence>150</confidence><answer>x</answer>", 100, "x", "5. confidence clipped high")
    check("<confidence>high</confidence><answer>Paris</answer>", 85, "Paris", "6. verbal 'high'")
    check("<confidence>very sure</confidence><answer>yes</answer>", 90, "yes", "6b. verbal 'very sure'")
    check("<confidence>73.6</confidence><answer>42</answer>", 74, "42", "7. float confidence")
    check("<confidence>80</confidence><answer>A</answer><confidence>30</confidence>", 80, "A", "8. multiple tags")
    check("<confidence>95</confidence><answer>Rome</answer>", 95, "Rome", "9. normal nested")
    check("<confidence>200</confidence><answer>x</answer>", 100, "x", "10. > 100 clipped")
    check("<confidence>-5</confidence><answer>x</answer>", 0, "x", "11. negative clipped")
    check("<confidence>50</confidence><answer></answer>", 50, "", "12. empty answer")
    check('<confidence>70</confidence><answer>"Paris"</answer>', 70, "Paris", "13. quoted answer")
    r14 = parse_response("<confidence>80</confidence><answer>I don't know</answer>")
    assert r14.is_abstention, "14. abstention flag"
    assert r14.confidence <= 10, "14. abstention confidence"
    print("  ✅  14. I don't know → abstention=True, conf≤10")
    check(None, DEFAULT_CONFIDENCE, "", "15. None input")
    check(42, DEFAULT_CONFIDENCE, None, "15b. int input")
    check("", DEFAULT_CONFIDENCE, "", "15c. empty string")
    check("  <confidence>  60  </confidence>  <answer>  Berlin  </answer>  ", 60, "Berlin", "whitespace trimmed")
    check("<CONFIDENCE>80</CONFIDENCE><ANSWER>Rome</ANSWER>", 80, "Rome", "uppercase tags")
    check("<confidence>50</confidence><answer>The Eiffel Tower</answer>", 50, "The Eiffel Tower", "multi-word answer")

    if failures:
        print("\n❌ FAILURES:")
        for f in failures:
            print(f"   {f}")
    else:
        print("\n✅  All parser tests passed.")
