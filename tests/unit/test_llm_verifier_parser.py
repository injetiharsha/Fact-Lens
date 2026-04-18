from pipeline.verdict.llm_verifier import LLMVerifier


def _mk() -> LLMVerifier:
    return LLMVerifier(provider="openai", model="gpt-4o-mini")


def test_parse_json_with_code_fence():
    v = _mk()
    content = """```json
{"verdict":"support","confidence":0.91,"reason":"matched evidence"}
```"""
    parsed = v._parse_json_from_content(content)
    assert parsed["verdict"] == "support"
    assert abs(float(parsed["confidence"]) - 0.91) < 1e-9


def test_salvage_malformed_json_with_unescaped_reason():
    v = _mk()
    content = """
{
  "verdict": "refute",
  "confidence": 87%,
  "reason": "The claim is false because "source A" contradicts it
}
"""
    parsed = v._parse_json_from_content(content)
    assert parsed["verdict"] == "refute"
    assert 0.86 <= float(parsed["confidence"]) <= 0.88
    assert "false" in parsed["reason"].lower()


def test_fallback_plaintext_verdict_not_supported():
    v = _mk()
    parsed = v._extract_plaintext_verdict_with_mode(
        "Result: not supported by provided evidence. confidence: 0.73",
        allow_lenient=False,
    )
    assert parsed["verdict"] == "refute"
    assert 0.72 <= float(parsed["confidence"]) <= 0.74


def test_parse_keyed_nonstandard_fields():
    v = _mk()
    content = 'classification = "contradicted", confidence=76, explanation="sources disagree"'
    parsed = v._parse_json_from_content(content)
    assert parsed["verdict"] == "refute"
    assert 0.75 <= float(parsed["confidence"]) <= 0.77
